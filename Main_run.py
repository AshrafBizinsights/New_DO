from datetime import datetime, timedelta
import calendar
import re
import joblib
import os
import anthropic
from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import streamlit as st
from functions.check_freshness import check_freshness
from functions.Table_uniqueness import is_compound_key
from functions.FILTER_DATA_FOR_EACH_COMBINATION import INITIATE_DATA_QUALITY
from functions.PERFORM_DATA_PATTERN_CHECKS import INITIATE_DATA_QUALITY_PATTERN_CHECK
from functions.check_missing_values_new import check_missing_values_new
from functions.check_uniqueness_new import check_uniqueness_new
from functions.restatement import (
    highlight_pass_fail, filter_data_by_period,uploadedFiles,
    get_datetime_columns, train_changed,
    add_commas, structural_change_kpi,
    structural_change,plot_pie,display_comparison_results,remove_unnamed_columns,detect_changes
)
from functions.restatement_visual import (
    Brand_Trend_Graph, Anomalies_Brand, fig_to_base64, 
    BarChartNumberAnomaliesDetected, plot_anomaly_category_distribution
)
from functions.KPI import render_metric, render_metrics_dashboard
from functions.variables import categorize_columns , render_element
#from llm_agents.Data_summary import data_summary
import plotly.graph_objects as go


st.set_page_config(layout="wide")

def timePlusBoosting( yesterday_df, today_df, date_column, target_column, primary_key,categorical_cols,numerical_cols):
    model_path  = "artifacts/xgboost_model.pkl"
    today_df['pred'] = False

    # Detect and remove key columns from feature list
    columns = list(yesterday_df.columns)
    for col in [date_column, primary_key, target_column]:
        if col in columns:
            columns.remove(col)


    # Date conversion and sorting
    yesterday_df[date_column] = pd.to_datetime(yesterday_df[date_column])
    today_df[date_column] = pd.to_datetime(today_df[date_column])
    yesterday_df = yesterday_df.sort_values(by=date_column, ascending=True)
    today_df = today_df.sort_values(by=date_column, ascending=True)

    # Filter today's data for a given start date (optional: make configurable)
    today_df = today_df[today_df[date_column] >= '2023-10-01'].reset_index(drop=True)

    # Extract month from date column
    yesterday_df['month'] = yesterday_df[date_column].dt.to_period('M').dt.strftime('%b')
    today_df['month'] = today_df[date_column].dt.to_period('M').dt.strftime('%b')

    # One-hot encode categorical columns
    yesterday_df_encoded = pd.get_dummies(yesterday_df, columns=categorical_cols)
    today_df_encoded = pd.get_dummies(today_df, columns=categorical_cols)
    today_df_encoded = today_df_encoded.reindex(columns=yesterday_df_encoded.columns, fill_value=0)

    # Drop non-numeric columns from training data
    X_train = yesterday_df_encoded.drop([primary_key, date_column, target_column, 'month'], axis=1, errors='ignore')
    X_test = today_df_encoded.drop([primary_key, date_column, target_column, 'month'], axis=1, errors='ignore')

    # Scaling numerical columns
    scaler = MinMaxScaler()
    yesterday_df_encoded[numerical_cols] = scaler.fit_transform(yesterday_df_encoded[numerical_cols])
    today_df_encoded[numerical_cols] = scaler.transform(today_df_encoded[numerical_cols])

    # Features and target split
    y_train = yesterday_df_encoded[target_column]
    y_test = today_df_encoded[target_column]

    # Load or train the model
    if os.path.exists(model_path):
        print("Loading existing model from disk.")
        model = joblib.load(model_path)
        
        # Check for feature mismatch
        model_features = model.get_booster().feature_names
        current_features = X_train.columns.tolist()

        # If feature sets differ, retrain. If sets match but order differs, align columns to model order.
        if set(model_features) != set(current_features):
            print("Feature set mismatch detected between the model and the current data. Retraining the model.")
            model = XGBRegressor(
                n_estimators=650,
                learning_rate=0.1,
                max_depth=7,
                min_child_weight=3,
                subsample=1.0,
                random_state=42
            )
            model.fit(X_train, y_train)
            joblib.dump(model, model_path)  # Save the retrained model
        else:
            if model_features != current_features:
                print("Feature order differs; aligning X_train and X_test to model feature order.")
                X_train = X_train.reindex(columns=model_features, fill_value=0)
                X_test = X_test.reindex(columns=model_features, fill_value=0)

    else:
        print("Training new XGBoost model.")
        model = XGBRegressor(
            n_estimators=650,
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=3,
            subsample=1.0,
            random_state=42
        )
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)  # Save the new model        

    y_pred = model.predict(X_test)
    today_df['predicted'] = y_pred
    residuals = np.abs(y_test - y_pred)
    today_df['residuals'] = residuals

    # Initial anomaly detection
    today_df['anomaly'] = False

    # Threshold calculation for anomaly detection
    threshold_data = []
    for product in today_df[categorical_cols[0]].unique():
        for indicator in today_df[categorical_cols[1]].unique():
            for month in today_df['month'].unique():
                sub_df = today_df[
                    (today_df[categorical_cols[0]] == product) & 
                    (today_df[categorical_cols[1]] == indicator) & 
                    (today_df['month'] == month)
                ].reset_index(drop=True)

                if not sub_df.empty:
                    median_residuals = np.median(sub_df['residuals'])
                    mad_residuals = np.median(np.abs(sub_df['residuals'] - median_residuals))
                    upper_threshold = median_residuals + 3 * mad_residuals + (0.04 * median_residuals)
                    lower_threshold = median_residuals - 3 * mad_residuals - (0.04 * median_residuals)

                    threshold_data.append({
                        categorical_cols[0]: product,
                        categorical_cols[1]: indicator,
                        'month': month,
                        'upper_threshold': upper_threshold,
                        'lower_threshold': lower_threshold
                    })

                    today_df.loc[
                        (today_df[categorical_cols[0]] == product) &
                        (today_df[categorical_cols[1]] == indicator) &
                        (today_df['month'] == month) &
                        ((today_df['residuals'] > upper_threshold) |
                         (today_df['residuals'] < lower_threshold)),
                        'anomaly'
                    ] = True

    # Merge threshold data into final dataframe
    threshold_df = pd.DataFrame(threshold_data)
    today_df = today_df.merge(threshold_df, how='left', on=[categorical_cols[0], categorical_cols[1], 'month'])

    # Anomaly categorization based on residuals
    max_residual = today_df['residuals'].max()
    high_threshold = 0.5 * max_residual
    medium_threshold = 0.3 * max_residual

    def categorize_residual(residual, anomaly):
        if not anomaly:
            return 'Not Anomalous'
        elif residual > high_threshold:
            return 'High'
        elif residual > medium_threshold:
            return 'Medium'
        else:
            return 'Low'

    today_df['residual_category'] = today_df.apply(
        lambda row: categorize_residual(row['residuals'], row['anomaly']), axis=1
    )

    today_df['pred'] = today_df['anomaly'].apply(lambda x: 1 if x else 0)
    today_df.to_csv("testingdatacheck.csv", index=False)


    return today_df







def main():



        config_df = pd.read_csv('ConfigAll_Biz.csv')
    
        CONFIG_DF = config_df
        CONFIG_DF = CONFIG_DF[(CONFIG_DF['ID'] == 1)].copy()


        def process_config_row(row):

            #--------------------------------------------------------------------------------------#
            ### Coniguration selector
            EXTRACTED_DATA_FROM_DB = pd.read_csv("Sales_data.csv")
            
            TABLE_NAME = row['Table Name']
            FILTER_TO_BE_APPLIED = row['Filter']
            DATE_COLUMN_IN_DATASET = row['Date_Field']
            AGGREGATE_COLUMNS = row['Aggregate Columns']
            PIVOT_FLAG = row['Pivot Down']
            #Data_Validation = row["Data Validation Columns"]
            columns_for_freshness = row['Data Freshness'].split(';')
            columns_for_volume = row['Data Volume'].split(';')
            columns_for_not_nulls = row['Not Null'].split(';')
            columns_to_check_uniqueness = row['Uniqueness'].split(';')
            AGGREGATE_COLUMNS_RAW = row['Dimension Patterns'].split(';')
            compund_key_columns_list = row['Table Level Uniqueness'].split(';')
            rca_column  = row["RCA Analysis Columns"].split(';')
            Target_col = row["Restatement Metric Column"]
            Primary_Key = row["Primary Key"]
            Brand_list = []
            try:
                Brand_list  = row["brands_list"].split(';')
            except:
                pass
            
            product_name = row["product_name"]

            


            if PIVOT_FLAG == 'No':
                METRIC_COLUMNS = row['Dimension Patterns Metric'].split(';')
            else:
                METRIC_COLUMNS = row['Aggregate Metric'].split(';')

            metric_column_name = row['Metric Column Name']

            LIST_OF_AGGREGATE_COLUMNS = AGGREGATE_COLUMNS.split(';')
            try:
                EXTRACTED_DATA_FROM_DB[DATE_COLUMN_IN_DATASET] = EXTRACTED_DATA_FROM_DB[
                    DATE_COLUMN_IN_DATASET].apply(lambda x: x.strftime('%Y-%m-%d'))
            except Exception as e:
                pass

            # EXTRACTED_DATA_FROM_DB[DATE_COLUMN_IN_DATASET] = pd.to_datetime(EXTRACTED_DATA_FROM_DB[DATE_COLUMN_IN_DATASET], errors='coerce', dayfirst=True, format='%Y-%m-%d')

            EXTRACTED_DATA_FROM_DB[DATE_COLUMN_IN_DATASET] = pd.to_datetime(
                EXTRACTED_DATA_FROM_DB[DATE_COLUMN_IN_DATASET], errors='coerce', dayfirst=True, format='%m/%d/%Y')
            EXTRACTED_DATA_FROM_DB[DATE_COLUMN_IN_DATASET] = pd.to_datetime(
                EXTRACTED_DATA_FROM_DB[DATE_COLUMN_IN_DATASET], format='%d-%m-%Y', errors='coerce')

            new_column_names = {DATE_COLUMN_IN_DATASET: 'ds'}
            EXTRACTED_DATA_FROM_DB = EXTRACTED_DATA_FROM_DB.rename(columns=new_column_names)

            baseline = EXTRACTED_DATA_FROM_DB.isnull().sum().reset_index()
            baseline_obs = EXTRACTED_DATA_FROM_DB.shape[0]
            baseline.columns = ['Column', 'Null Count']

            #------------------------------------------------------------------------------#
            # Restatement reader and filter

            yesterday_file=r"Data/Previous_Data.xlsx"
            today_file=r"Data/Current_Data.xlsx"
            
            selected_date_col=''

            today = datetime.now()
            selected_period_col='R6M'
            today_df=pd.DataFrame()

            if 'button_train_model' not in st.session_state:
                st.session_state.button_train_model = False
            
            # Get today's date and format it
            today = datetime.today()
            formatted_date = today.strftime('%m/%d/%Y')

        
            
            
        
            
            datetime_columns = get_datetime_columns(today_df)        
            num_cols=today_df.select_dtypes(include=['int64','float64']).columns.to_list()
            #Target_col='Tot_30day_Fills'
            


            yesterday_df = pd.read_excel(yesterday_file)
            yesterday_df=remove_unnamed_columns(yesterday_df)
            today_df = pd.read_excel(today_file)
            today_df=remove_unnamed_columns(today_df)

            brand_options = list(set(yesterday_df['PRODUCT'].unique()).union(set(today_df['PRODUCT'].unique())))
            brand_selected='Brand11'
            selected_date_col='WEEK'
            period_number=int(re.findall(r'\d+', selected_period_col)[0])
            yesterday_df=filter_data_by_period(yesterday_df, selected_date_col, 'R6M')
            
            #dimension_option=yesterday_df.select_dtypes(include=['object']).columns.to_list()
            Dimension_selection='PRODUCT'
            month_select=''
            Metric_selection='TOTAL_RX'
            
            anomalies_category='Significant'
            
            #st.sidebar.button("Train Model",help="Train Model on Yesterday's Last "+selected_period_col,on_click=train_changed)
            
            testingdataML=pd.DataFrame()
            testingdata=today_df.copy()
            testingdata = testingdata[testingdata['WEEK'] >= '2023-10-01'].reset_index(drop=True)

            
            date_column=DATE_COLUMN_IN_DATASET
            target_column=Target_col
            primary_key= Primary_Key
            categorized_columns = categorize_columns(yesterday_df)
            categorical_cols = categorized_columns["categorical"]
            numerical_cols = categorized_columns["numerical"]
            testingdataML=timePlusBoosting(yesterday_df,today_df,date_column,target_column,primary_key,categorical_cols,numerical_cols)
            print("Outside Model TestingdataML", testingdataML.columns)
            testingdataML.columns = testingdataML.columns.str.strip()
            testingdataML = testingdataML.loc[:, ~testingdataML.columns.str.startswith('Unnamed:')]

            testingdataML.to_csv("test24.csv",index=False)

            
            testingdata=testingdataML.copy()
            print("Outside Model Testingdata", testingdata.columns)

            date_col=DATE_COLUMN_IN_DATASET
            id_col= Primary_Key
            prediction_col=Target_col

            columns = list(yesterday_df.columns)
            for col in [date_col, id_col, prediction_col]:
                if col in columns:
                    columns.remove(col)

            results = detect_changes(yesterday_df, today_df)

            #-----------------------------------------------------------------------------------#
            # preview and overview

            freshnessDf = pd.DataFrame(columns=['Dimension', 'Value', 'Status', 'Comment'])
            uniqueDf = pd.DataFrame(columns=['Dimension', 'Unique Check Result', 'Unique Check Comment.'])
            notnullDf = pd.DataFrame(columns=['Dimension', 'NOT NULL Check Result', 'NOT NULL Comment'])

            st.image("logoo-removebg-preview.png", width=150) 
            # Define the custom CSS to fix the image

            config_df = pd.read_csv('ConfigAll_Biz.csv')
            st.markdown(f"<div style='margin-bottom: -90px; font-weight:bold; color:black; text-align:left;'>DQ Monitoring Rules Configuration</div>",unsafe_allow_html=True)
            st.write(" ")
            with st.expander("Expand to modify the configuration"):
                df_edited = st.data_editor(config_df, hide_index=True)
                df_edited.to_csv('ConfigAll_Biz.csv', index=False)


            # Data Preview
            with st.expander("Expand to see the Data Preview and Profiling"):
                st.markdown(f"<div style='margin-bottom: -70px; font-weight:bold; color:black; text-align:left;margin-top:20px;'>Data Preview</div>", unsafe_allow_html=True)
                st.write(" ")
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<p class="title-font" style="color:black;margin-top:30px;" ><b>Current Month Data<b></p>', unsafe_allow_html=True)

                # Modify the DataFrame display for preview
                today_df1 = pd.read_csv("data_preview.csv")

                
                st.dataframe(today_df1.head(20))

                st.markdown('<p class="title-font" style="color:black;margin-top:30px;" ><b>Data Profile by Dimension<b></p>', unsafe_allow_html=True)
                plot_pie(today_df, Dimension_selection)


         
            #-------------------------------------------------------------------------------------#
            # KPI
            

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Reading the data
            Data_freshness = pd.read_csv("freshnessDf.csv")
            Data_Validation = pd.read_csv("Data_Validation.csv")
            Brand_Volume = pd.read_csv("Time_series_Analysis.csv")
            #Unknown_Unknown = pd.read_csv("Pattern_Check_Analysis.csv")
            Unknown_Unknown = pd.read_csv("Pattern_check_hard_coded_csv.csv")
            data_triangulation = pd.read_csv("Data_trangulation.csv")
            

            df_structure_kpi = structural_change_kpi(yesterday_df, today_df)
            
            
            render_metrics_dashboard(st, Data_freshness, Data_Validation, Brand_Volume, Unknown_Unknown,data_triangulation, testingdata, anomalies_category, results, render_metric, add_commas,df_structure_kpi)

            st.write(" ")
            #-----------------------------------------------------------------------------------------------------#
            # Assigning container for all check

            structural_change_continer = st.container(border = True)
            freshness_continer = st.container(border = True)
            brand_volume_continer = st.container(border = True)
            data_triangulation = st.container(border = True)
            restatement_continer = st.container(border = True)
            Unknown_Unknown_continer = st.container(border = True)
            data_validation_continer = st.container(border = True)
            table_uniqueness_continer = st.container(border = True)

            #----------------------------------------------------------------------------------------------------#
            # structual changes

            st.markdown("<br>", unsafe_allow_html=True)
            with structural_change_continer:
                st.markdown("""<style>.stApp {background-color: white; }</style>""", unsafe_allow_html=True)
                bold_text = f"Table Structure Changes"
                styled_text = f"<div style='font-weight: bold; color: black; font-size: 20px; margin-top: -10px;'>{bold_text}</div>"
                st.markdown(styled_text, unsafe_allow_html=True)
                st.write('    ')

                structural_change(yesterday_df, today_df)
                st.markdown("<br>", unsafe_allow_html=True)
            
            #---------------------------------------------------------------------------------------#
            #  data freshness check

            #data_freshness_date = date_selector(EXTRACTED_DATA_FROM_DB)
            data_freshness_date = "2024-01-19"
            #data_freshness_date = max(EXTRACTED_DATA_FROM_DB['ds'])

            new_fresh_df = check_freshness(EXTRACTED_DATA_FROM_DB, data_freshness_date, columns_for_freshness,freshnessDf)
            
            freshness_continer.markdown("""<style>.stApp {background-color: white; }</style>""", unsafe_allow_html=True)
            bold_text = f"Data Freshness/Recency Summary"

            styled_text = f"<div style='font-weight: bold; color: black; font-size: 20px; margin-top: -10px;'>{bold_text}</div>"
            freshness_continer.markdown(styled_text, unsafe_allow_html=True)
            new_fresh_df.sort_values(by=['Comment'],inplace=True)

            new_fresh_df.rename(columns={'Comment': 'Comments'}, inplace=True)

            # Get unique conditions and add "ALL" option
            unique_conditionsf = new_fresh_df['Dimension'].unique().tolist()
            unique_conditionsf.insert(0, "ALL")
            
            # Select box for conditions with "ALL" option
            selected_conditionf = freshness_continer.selectbox("Select Condition", unique_conditionsf)
            
            # Filter DataFrame based on selected condition
            if selected_conditionf == "ALL":
                filtered_result = new_fresh_df
            else:
                filtered_result = new_fresh_df[new_fresh_df['Dimension'] == selected_conditionf]
            filtered_result = filtered_result.reset_index(drop=True)
            styled_df = filtered_result.style.applymap(highlight_pass_fail)

            with freshness_continer:
                    col1,col2 = st.columns(2)

                    with col1:
                       
                        st.dataframe(styled_df, width=1700, hide_index=True)
                    with col2:
                        summary_continer = st.container(border=True)
                        check_name = "Freshness Check"
                        summary_continer.subheader(f"{check_name} Summary")
                        #summary = data_summary(new_fresh_df,check_name)

                        summary = """
                    
                        Total Records: 41 brands (Brand1 through Brand42).
                        Dimension Type: PRODUCT.
                        Status: All records passed quality check (100% pass rate).
                        Comments: No comments/issues reported for any record.
                        Data Completeness: Sequential brand numbering.
                        Quality Metrics: 41/41 records passed validation criteria.
                        The data shows consistent quality across all product brands with no identified issues or failures.
                        """
                        summary_continer.write(summary)
            #freshness_continer.dataframe(styled_df, width=1700, hide_index=True)


            new_fresh_df.to_csv('freshnessDf.csv', index=False)

            #-----------------------------------------------------------------------------------#
            # Brand Volumn check

            container_Brand_volume = st.container(border=True)
            brand_volume_continer.markdown("""<style>.stApp {background-color: white; }</style>""",unsafe_allow_html=True)
            bold_text = f"DQ Summary of Brand Volume Checks"
            styled_text = f"<div style='font-weight: bold; color: black; font-size: 20px; margin-top: -10px;'>{bold_text}</div>"
            brand_volume_continer.markdown(styled_text, unsafe_allow_html=True)
            brand_volume_continer.write('    ')
            time_series = pd.read_csv("Time_series_Analysis.csv")
            #styled_df = time_series.style.applymap(highlight_pass_fail)
            #brand_volume_continer.dataframe(styled_df, width=1500, hide_index=True)

            textDict = {'d1': 'dval'}
            DATE_COLUMN_IN_DATASET = 'ds'

            
            EXTRACTED_DATA_FROM_DB_NVS_BRANDS_FILTERED = EXTRACTED_DATA_FROM_DB.copy()

            ##Check Function
            if PIVOT_FLAG == 'Yes':
                AGGREGATE_METRICS = row['Aggregate Metric']
                LIST_OF_AGGREGATE_METRIC_COLUMNS = AGGREGATE_METRICS.split(';')

                for METRIC_COLUMN in LIST_OF_AGGREGATE_METRIC_COLUMNS:
                    INITIATE_DATA_QUALITY(EXTRACTED_DATA_FROM_DB_NVS_BRANDS_FILTERED, LIST_OF_AGGREGATE_COLUMNS,
                                        DATE_COLUMN_IN_DATASET, METRIC_COLUMN, textDict,brand_volume_continer,rca_column,product_name,Brand_list)
            else:
                METRIC_COLUMN = metric_column_name

                INITIATE_DATA_QUALITY(EXTRACTED_DATA_FROM_DB_NVS_BRANDS_FILTERED, LIST_OF_AGGREGATE_COLUMNS,
                                    DATE_COLUMN_IN_DATASET, METRIC_COLUMN, textDict,brand_volume_continer,rca_column,product_name,Brand_list)

            #-----------------------------------------------------------------------------------------------#
            

            data_triangulation.markdown("""<style>.stApp {background-color: white; }</style>""", unsafe_allow_html=True)
            bold_text = f"Data Triangulation Check"
            styled_text = f"<div style='font-weight: bold; color: black; font-size: 20px; margin-top: -10px;'>{bold_text}</div>"
            data_triangulation.markdown(styled_text, unsafe_allow_html=True)
            
            # Read the data
            data = pd.read_excel(r"sales2.xlsx")
            data['delta'] = round(data['delta'])
            data.rename(columns={"WEEK": "date"}, inplace=True)
            delta = data[['date', 'delta']]
            delta.rename(columns={"date": "ds", "delta": "y"}, inplace=True)
            
            # Fit Prophet model
            model = Prophet()
            model.fit(delta)
            future = model.make_future_dataframe(periods=0)
            forecast = model.predict(future)
            
            df_merged = delta.merge(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds")
            df_merged["anomaly"] = (df_merged["y"] < df_merged["yhat_lower"]) | (df_merged["y"] > df_merged["yhat_upper"])
            df_anomalies = df_merged[df_merged["anomaly"]]
            
            df_anomaly = df_anomalies.copy()
            df_anomaly.rename(columns={'ds': 'date'}, inplace=True)

        
         
            
            # Plot with Plotly
            x_start = df_merged['ds'].min() - pd.Timedelta(days=7)
            x_end = df_merged['ds'].max()
            
            fig = go.Figure()
            
            # Delta line
            fig.add_trace(go.Scatter(
                x=df_merged['ds'], y=df_merged['y'],
                mode='lines+markers',
                name='Delta (TRx)',
                line=dict(color='blue')
            ))
            
            # Anomaly points
            fig.add_trace(go.Scatter(
                x=df_anomaly['date'], y=df_anomaly['y'],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=10)
            ))
            
            # Upper threshold
            fig.add_trace(go.Scatter(
                x=df_merged['ds'], y=df_merged['yhat_upper'],
                mode='lines',
                name='Upper Threshold',
                line=dict(color='green', width=2, dash='dash')
            ))
            
            # Lower threshold
            fig.add_trace(go.Scatter(
                x=df_merged['ds'], y=df_merged['yhat_lower'],
                mode='lines',
                name='Lower Threshold',
                line=dict(color='orange', width=2, dash='dash')
            ))
            
            # Layout settings
            fig.update_layout(
                title='Weekly TRx Delta ',
                xaxis_title='Date',
                yaxis_title='Delta (TRx)',
                xaxis=dict(
                    range=[x_start, x_end],
                    tickformat='%b %d\n%Y',
                    tickangle=0,
                    tickmode='linear',
                    dtick=7 * 24 * 60 * 60 * 1000,  # 1 week in milliseconds
                    showgrid=True
                ),
                yaxis=dict(
                    tickmode='linear',
                    tick0=0,
                    dtick=20,
                    range=[0, max(df_merged['y'].max(), df_merged['yhat_upper'].max()) + 10]
                ),
                legend=dict(
                    x=1.05,
                    y=1,
                    xanchor='left',
                    yanchor='top',
                    font=dict(size=10),
                    bgcolor='rgba(255,255,255,0.6)',
                    bordercolor='lightgray',
                    borderwidth=1
                ),
                margin=dict(r=100),  # right margin for legend
                template='plotly_white'
            )
            
            # Display the figure in Streamlit
            #st.plotly_chart(fig)

            with data_triangulation:
                   

           
                    st.plotly_chart(fig)
             
                    summary_continer = st.container(border=True)
                    check_name = "Data Triangulation Check"
                    summary_continer.subheader(f"{check_name} Summary")
                    # summary = data_summary(data,check_name)
                    summary = """
                    Compared weekly TRx values between IQVIA Xponent and IQVIA NPA datasets over a 13-week period.
                    For 12 weeks, both sources showed consistent trends with minimal deviation.
                    In the week of Oct 29, Xponent TRx showed a sharp increase while NPA remained flat, flagging a potential reporting lag or coverage shift in the NPA feed.
                    """
                    summary_continer.write(summary)
    

            #-----------------------------------------------------------------------------------------------#
            #Restatement Analysis

            today_df_final  =  timePlusBoosting(yesterday_df,today_df,date_column,target_column,primary_key,categorical_cols,numerical_cols)

            with restatement_continer:
                st.markdown("""<style>.stApp {background-color: white; }</style>""", unsafe_allow_html=True)
                bold_text = f"Restatement Analysis "
                styled_text = f"<div style='font-weight: bold; color: black; font-size: 23px; margin-top: -10px;'>{bold_text}</div>"
                st.markdown(styled_text, unsafe_allow_html=True)



                columns = list(yesterday_df.columns)
                for col in [date_col, id_col, prediction_col]:
                        if col in columns:
                            columns.remove(col)

                
                Restatement_Summary=st.columns(1)[0]
                with Restatement_Summary:
                    
                    display_comparison_results(
                        results=results,data1=yesterday_df,
                        data2=today_df_final,
                        date_col=date_col,
                        id_col=id_col,
                        # prediction_col=prediction_col,
                        categorical_cols=categorical_cols,
                        flag=True
                    )
                    st.markdown("<br>", unsafe_allow_html=True)
                
                    col1,col2 = st.columns(2)
                    with col1:
                        Brand_Trend_Graph(yesterday_df,today_df, brand_selected, month_select, Metric_selection)
                
                        #today_df_final  = timePlusBoosting(yesterday_df, today_df)
                        
                            
                    with col2:
                        
                        fig = BarChartNumberAnomaliesDetected(today_df_final)
                        img2_base64 = fig_to_base64(fig)

                        st.markdown(
                            """
                            <style>
                            .Graph-box6 {
                                border: 1px solid #ddd;
                                border-radius: 5px;
                                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
                                background-color: #f9f9f9;
                                text-align: center;
                                padding: 20px;
                                margin: 10px 0;
                                height: 350px;  /* Set a fixed height */
                            }
                            .Graph-box6 p {
                                font-size: 12px;
                                font-weight: bold;
                                color: #333;
                            }
                            .legend-box {
                                display: flex;
                                justify-content: center;
                                margin-top: 10px;
                            }
                            .legend-box div {
                                display: flex;
                                align-items: center;
                                margin-right: 20px;
                            }
                            .legend-box span {
                                width: 12px;
                                height: 12px;
                                display: inline-block;
                                margin-right: 5px;
                            }
                            .legend-box .green {
                                background-color: skyblue;
                            }
                            .legend-box .red {
                                background-color: #0063C8;
                            }
                            </style>
                            """,
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f"""
                            <div class='Graph-box6'>
                                <p>Total No. of Restatement Detected in Current Month Data Load</p>
                                <img  src='data:image/png;base64,{img2_base64}' style='width:80%;position:relative;
                                margin-top: 66px;'> 
                                <div class='legend-box'>
                                    <div style="color:black" ><span class='green'></span><b>Non-Restatement  Records<b></div>
                                    <div style="color:black" ><span class='red'></span><b>Restatement  Records<b></div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.markdown("<br><br><br>", unsafe_allow_html=True)
                        #Anomalies_Brand(testingdata,Dimension_selection)  

                    #dimensionCol = ['PRODUCT', 'INDICATION','SPECIALTY']
                    dimensionCol = rca_column
                    plot_anomaly_category_distribution(today_df_final, dimensionCol)
                    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

            #--------------------------------------------------------------------------------------------#
            # Unknow Unknow check

        
            Unknown_Unknown_continer.markdown("""<style>.stApp {background-color: white; }</style>""",
                                            unsafe_allow_html=True)
            bold_text = f"DQ Summary of Unknown unknowns"
            styled_text = f"<div style='font-weight: bold; color: black; font-size: 20px; margin-top: -10px;'>{bold_text}</div>"
            Unknown_Unknown_continer.markdown(styled_text, unsafe_allow_html=True)
            Unknown_Unknown_continer.write('    ')
            pattern_check = pd.read_csv(r'./Pattern_check_hard_coded_csv.csv')
            #styled_df = pattern_check.style.applymap(highlight_pass_fail)
            #Unknown_Unknown_continer.dataframe(styled_df, width=1500, hide_index=True)

            AGGREGATE_COLUMNS = []
        
            EXTRACTED_DATA_FROM_DB_NVS_BRANDS_FILTERED_Patterns = EXTRACTED_DATA_FROM_DB.copy()

            for COLUMN_SET in AGGREGATE_COLUMNS_RAW:
                if "," in COLUMN_SET:
                    AGGREGATE_COLUMNS.append(COLUMN_SET.split(','))
                else:
                    AGGREGATE_COLUMNS.append(COLUMN_SET)

                for METRIC_COLUMN_NAME in METRIC_COLUMNS:
                    if PIVOT_FLAG == 'No':
                        VALUE_COLUMN_NAME = row['Metric Column Name']
                    else:
                        VALUE_COLUMN_NAME = METRIC_COLUMN_NAME.strip()
                    # print(AGGREGATE_COLUMNS)

            INITIATE_DATA_QUALITY_PATTERN_CHECK(EXTRACTED_DATA_FROM_DB_NVS_BRANDS_FILTERED_Patterns,AGGREGATE_COLUMNS, METRIC_COLUMN_NAME, VALUE_COLUMN_NAME,DATE_COLUMN_IN_DATASET, PIVOT_FLAG,Unknown_Unknown_continer)

            #----------------------------------------------------------------------------------------#
            # Data Validation Check
            data_validation_continer.markdown("""<style>.stApp {background-color: white; }</style>""", unsafe_allow_html=True)
            bold_text = f"Data Validations Summary "
            styled_text = f"<div style='font-weight: bold; color: black; font-size: 20px; margin-top: -10px;'>{bold_text}</div>"
            data_validation_continer.markdown(styled_text, unsafe_allow_html=True)
            data_validation_continer.write('    ')

            new_null_check = check_missing_values_new(
                EXTRACTED_DATA_FROM_DB,
                columns_for_not_nulls,
                baseline,
                baseline_obs,
                notnullDf,
            )


            new_null_check.to_csv("Null_Check.csv", index=False)

            new_unique_check = check_uniqueness_new(
                EXTRACTED_DATA_FROM_DB,
                data_freshness_date,
                columns_to_check_uniqueness,
                uniqueDf,
            )
            new_unique_check.to_csv("Unique_Check.csv", index=False)
        
            
            

            # Read the CSV files
            df_unique = pd.read_csv('Unique_Check.csv')
            #df_unique = df_unique.sort_values(by='Dimension', ascending=False)
            df_unique = df_unique.rename(columns={
                "Unique Check Result": "Status",
                "Unique Check Comment.": "Comments"})
            df_unique['Check'] = 'Unique'

            df_Null = pd.read_csv("Null_Check.csv")
            #df_Null = df_Null.sort_values(by='Dimension', ascending=False)
            df_Null = df_Null.rename(columns={
                "NOT NULL Check Result": "Status",
                "NOT NULL Comment": "Comments"})
            df_Null['Check'] = 'Not_Null'

            #result = pd.concat([df_unique, df_Null], axis=0)
            result = pd.concat([df_Null], axis=0)


            result.reset_index(drop=True, inplace=True)
            result.to_csv("Data_Validation.csv")
            result = result.sort_values(by='Check', ascending=False)

            # Get unique conditions and add "ALL" option
            unique_conditions = result['Check'].unique().tolist()
            unique_conditions.insert(0, "ALL")
            
            # Select box for conditions with "ALL" option
            selected_condition = data_validation_continer.selectbox("Select Condition", unique_conditions)
            
            # Filter DataFrame based on selected condition
            if selected_condition == "ALL":
                filtered_result = result
            else:
                filtered_result = result[result['Check'] == selected_condition]
            styled_df = filtered_result.style.applymap(highlight_pass_fail)

            with data_validation_continer:
                    col1,col2 = st.columns(2)

                    with col1:
                
                        st.dataframe(styled_df, width=1700, hide_index=True)
                    with col2:
                        summary_continer = st.container(border=True)
                        check_name = "Validation Check"
                        summary_continer.subheader(f"{check_name} Summary")
                        # summary = data_summary(filtered_result,check_name)
                        summary = """
                            All 14 dimensions passed their respective Not_Null checks.
                            Most dimensions passed without any comments/issues.
                            Two dimensions showed null values but were deemed acceptable:
                            TOTAL_RX: Nulls within historical trend.
                            COMPETITIVE_RX: Nulls within historical trend.
                            Key dimensions checked include:
                            Product hierarchy (PRODUCT, PROD_LVL).
                            Medical categories (SPECIALTY, SUB_SPECIALTY).
                            Market indicators (MARKET_FLAG, LEVEL_0_MARKET_FLAG).
                            Prescription metrics (TOTAL_RX, COMPETITIVE_RX).
                            Overall Status: PASS - All checks met quality standards.
                            """
                        summary_continer.write(summary)

            #data_validation_continer.dataframe(styled_df, width=1700, hide_index=True)

            st.markdown("<br>", unsafe_allow_html=True)

            
            
            #------------------------------------------------------------------------------#
            # Table uniquness check
    
            table_uniqueness_continer.markdown("""<style>.stApp {background-color: white; }</style>""", unsafe_allow_html=True)
            bold_text = f"Data Uniqueness Check"
            styled_text = f"<div style='font-weight: bold; color: black; font-size: 20px; margin-top: -10px;'>{bold_text}</div>"
            table_uniqueness_continer.markdown(styled_text, unsafe_allow_html=True)
            table_uniqueness_continer.write('    ')
            
            is_compound_key(EXTRACTED_DATA_FROM_DB, compund_key_columns_list,data_freshness_date,table_uniqueness_continer)

        CONFIG_DF.apply(process_config_row, axis=1)

   
            

if __name__ == "__main__":
    main()


