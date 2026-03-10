from functions.custom_rca_for_each_week import custom_rca_for_each_week
import seaborn as sns
import pandas as pd
from datetime import timedelta
from prophet import Prophet
import matplotlib.pyplot as plt
import altair as alt
import streamlit as st
import numpy as np
from functions.restatement import highlight_pass_fail 
from llm_agents.Data_summary import data_summary

def INITIATE_DATA_QUALITY(EXTRACTED_DATA_FROM_DB, LIST_OF_AGGREGATE_COLUMNS, DATE_COLUMN_IN_DATASET,
                        METRIC_COLUMN, textDict,brand_volume_continer,rca_column,product_name,Brand_list):
    
   
    
    if len(Brand_list) > 0:
        EXTRACTED_DATA_FROM_DB = EXTRACTED_DATA_FROM_DB[EXTRACTED_DATA_FROM_DB[product_name].isin(Brand_list)]
    else:
        EXTRACTED_DATA_FROM_DB = EXTRACTED_DATA_FROM_DB

    product_sum_df = EXTRACTED_DATA_FROM_DB.groupby(LIST_OF_AGGREGATE_COLUMNS[0])[METRIC_COLUMN].sum().reset_index()
    percentile_90_value = np.percentile(product_sum_df[METRIC_COLUMN], 0)
    filtered_products = product_sum_df[product_sum_df[METRIC_COLUMN] >= percentile_90_value][LIST_OF_AGGREGATE_COLUMNS[0]]
    filtered_products_list = filtered_products.tolist()

    EXTRACTED_DATA_FROM_DB = EXTRACTED_DATA_FROM_DB[EXTRACTED_DATA_FROM_DB[LIST_OF_AGGREGATE_COLUMNS[0]].isin(filtered_products_list)]

    UNIQUE_COMBINATIONS = EXTRACTED_DATA_FROM_DB[LIST_OF_AGGREGATE_COLUMNS].drop_duplicates().apply(tuple,
                                                                                                    axis=1).tolist()
  
    # UNIQUE_COMBINATIONS = [('HS', 'COMPETITIVE', 'COSENTYX', 'MOTRX'),('PSO', 'COMPETITIVE', 'CIMZIA', 'MOTRX')]
    # UNIQUE_COMBINATIONS = [('COSENTYX', 'NBRX'),('COSENTYX SC', 'NBRX')]
    #appendReal = pd.DataFrame(columns=["ds", "yhat_lower", "yhat_upper", "y"])
    appendReal = pd.DataFrame(
                columns=["ds", "yhat_lower", "yhat_upper", "y", "Granularity"]
            )
        
    for CURRENT_COMBINATION in UNIQUE_COMBINATIONS:
        # st.write(CURRENT_COMBINATION)
        metric_val = CURRENT_COMBINATION[-1]
        try:
            rca_text = textDict[CURRENT_COMBINATION[-1]]
        except Exception as e:
            rca_text = ''
        brand_volume_call = FILTER_DATA_FOR_EACH_COMBINATION(CURRENT_COMBINATION, LIST_OF_AGGREGATE_COLUMNS,
                                           EXTRACTED_DATA_FROM_DB, METRIC_COLUMN, rca_text, metric_val,brand_volume_continer,rca_column)

        # Concatenate the result with the main DataFrame
        appendReal = pd.concat([appendReal, brand_volume_call], ignore_index=True)



    # Convert certain columns to numeric, handle errors gracefully
    appendReal["yhat_lower"] = pd.to_numeric(appendReal["yhat_lower"], errors="coerce")
    appendReal["yhat_upper"] = pd.to_numeric(appendReal["yhat_upper"], errors="coerce")

    appendReal["yhat_lower"] = appendReal["yhat_lower"].round(0).astype(int)
    appendReal["yhat_upper"] = appendReal["yhat_upper"].round(0).astype(int)
    
    appendReal["metric_name"] = METRIC_COLUMN
    
    # Add a 'Status' column based on whether the value falls within the threshold
    appendReal["Status"] = appendReal.apply(
        lambda row: (
            "Pass"
            if row["y"] >= row["yhat_lower"] and row["y"] <= row["yhat_upper"]
            else "Fail"
        ),
        axis=1,
    )

    # Rename columns for clarity
    new_column_names = {
        "ds": "Date",
        "yhat_lower": "Lower Threshold",
        "yhat_upper": "Upper Threshold",
        "y": "Value",
        "metric_name": "Metric Name"
    }
    appendReal["y"] = pd.to_numeric(appendReal["y"], errors="coerce").round(0)
    try:
        appendReal["y"] = appendReal["y"].astype("Int64")
    except Exception:
        appendReal["y"] = appendReal["y"].fillna(0).astype(int)
    appendReal = appendReal.rename(columns=new_column_names)
    appendReal_new = appendReal[["Metric Name","Granularity" , "Lower Threshold", "Upper Threshold", "Value","Status" ]]
    appendReal_new.rename(columns={'Value': 'Actuals'}, inplace=True)
    appendReal_new = pd.read_csv("Time_series_Analysis.csv")
    styled_df = appendReal_new.style.applymap(highlight_pass_fail)
    
    with brand_volume_continer:
        col1,col2 = st.columns(2)

        with col1:
            
            st.dataframe(styled_df, width=1500, hide_index=True)
        with col2:
            summary_continer = st.container(border=True)
            check_name = "Brand Volume Check"
            summary_continer.subheader(f"{check_name} Summary")
            #summary = data_summary(appendReal_new,check_name)
            summary = """
                      Total Metrics Analyzed : 
                        9 (all for COMPETITIVE_RX).
                        Pass Rate: 8/9 (88.89%).
                      Failed Check :
                        Brand2 | NBRX (Actual: 2302, exceeded Upper Threshold: 2162).
                      Brands Overview :
                        Brand1: All checks passed (TRX, NBRX, MOTRX).
                        Brand2: 2/3 passed (TRX, MOTRX passed; NBRX failed).
                        Brand3: All checks passed (TRX, NBRX, MOTRX).
                      Prescription Types :
                        TRX: All passed.
                        NBRX: 2/3 passed.
                        MOTRX: All passed.
                      
                      """
            summary_continer.write(summary)
    
  
    #appendReal_new.to_csv('data\Time_series_Analysis.csv', index=False)




def FILTER_DATA_FOR_EACH_COMBINATION(CURRENT_COMBINATION, LIST_OF_AGGREGATE_COLUMNS, EXTRACTED_DATA_FROM_DB,
                                                 METRIC_COLUMN, rca_text, metric_val,brand_volume_continer,rca_column):
 

    FILTERED_DATA_DF = EXTRACTED_DATA_FROM_DB.copy()

    #filtered_data = filtered_data[filtered_data[LIST_OF_AGGREGATE_COLUMNS[0]].isin(filtered_products)]

    FILTER_CONDITION = None
    for col, val in zip(LIST_OF_AGGREGATE_COLUMNS, CURRENT_COMBINATION):
        CONDITION = f"(FILTERED_DATA_DF['{col}'] == '{val}')"
        if FILTER_CONDITION is None:
            FILTER_CONDITION = CONDITION
        else:
            FILTER_CONDITION = FILTER_CONDITION + ' & ' + CONDITION

    FILTERED_DATA_FOR_MODEL = FILTERED_DATA_DF[eval(FILTER_CONDITION)].copy()

    brand = RUN_TIME_SERIES_ANALYSIS(FILTERED_DATA_FOR_MODEL, CURRENT_COMBINATION, LIST_OF_AGGREGATE_COLUMNS,
                                METRIC_COLUMN, rca_text, metric_val,brand_volume_continer,rca_column)
    return brand
    
def RUN_TIME_SERIES_ANALYSIS(filtered_data, selected_combination, LIST_OF_AGGREGATE_COLUMNS, METRIC_COLUMN,
                                         rca_text, metric_val,brand_volume_continer,rca_column):
    try:
    
        graphVal = 10
        
        graphNumber = 0
        appendReal = pd.DataFrame(
                columns=["ds", "yhat_lower", "yhat_upper", "y", "Granularity"]
            )
        
        # Step 1: Grouping and summing
        

        
        filtered_data = filtered_data.loc[filtered_data['ds'] > '2023-01-01'].copy()
        sns_c = sns.color_palette(palette='deep')
        filtered_data.rename(columns={'Date': 'ds', METRIC_COLUMN: 'y'}, inplace=True)
        filtered_data_all = filtered_data.copy()
        
        filtered_data['ds'] = pd.to_datetime(filtered_data['ds'], errors='coerce', dayfirst=True,
                                                format='%Y-%m-%d')
        # st.write('test2')
        
        max_date = filtered_data['ds'].max()
        start_date_52_weeks_ago = max_date - timedelta(weeks=216)
        filtered_data = filtered_data[filtered_data['ds'] > start_date_52_weeks_ago]
        filtered_data_historical = filtered_data_all[filtered_data_all['ds'] >= start_date_52_weeks_ago]

        # LIST_OF_AGGREGATE_COLUMNS.append('ds') if 'ds' not in LIST_OF_AGGREGATE_COLUMNS else None

        if 'ds' not in LIST_OF_AGGREGATE_COLUMNS:
            LIST_OF_AGGREGATE_COLUMNS.append('ds')

        filtered_data_historical = filtered_data_historical.groupby(LIST_OF_AGGREGATE_COLUMNS).agg(
            {'y': 'sum'}).reset_index()
        
        

        if len(filtered_data) == 0:
            return 'Data not available for selected combination'

        # filtered_data['ds'] = filtered_data['ds'].dt.tz_localize(None)
        # excel_path = './Data/BeforeSum.xlsx'
        # filtered_data.to_excel(excel_path, index=False)
        
        filtered_data = filtered_data.groupby(LIST_OF_AGGREGATE_COLUMNS).agg({'y': 'sum'}).reset_index()
        


        df_holidays_manual = pd.read_csv('holidays_1.csv')
        df_holidays_manual = df_holidays_manual.dropna()

        #model = Prophet(interval_width=0.95, holidays=df_holidays_manual ,changepoint_prior_scale=0.05,seasonality_mode='multiplicative',seasonality_prior_scale=10.0,changepoint_range=0.8,yearly_seasonality=True,weekly_seasonality=True)

        model = Prophet(interval_width=0.95, holidays=df_holidays_manual)
        #model.add_seasonality(name='daily', period=1, fourier_order=10)

        #model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        #model = Prophet(interval_width=0.95, holidays=df_holidays_manual)

        model.fit(filtered_data)


        #future = model.make_future_dataframe(periods=1, freq='W-Fri')
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)
        forecast = pd.merge(forecast, filtered_data_historical[['ds', 'y']], on='ds', how='inner')
        forecast = forecast.sort_values(by='ds')


        forecast['y'] = pd.to_numeric(forecast['y'], errors='coerce')
        forecast['5%_increase'] = forecast['y'].shift(1) * 1.05
        forecast['5%_decrease'] = forecast['y'].shift(1) * 0.95

        result_string = ' | '.join(str(x) for x in selected_combination)
        # Split the result_string based on the delimiter '|'
        parts = result_string.split(' | ')




        # Save the two parts into separate lists
        Latest_Week = []
        Trend_Break = []

        if len(parts) == 2:
            Latest_Week.append(parts[0])
            Trend_Break.append(parts[1])
        else:
            print("The result_string does not contain exactly two parts.")



        if graphNumber < graphVal:
            ###############################################################################
            # GRAPH CODE STARTS HERE
            type_of_graphs = ['Manual Thresholds', 'ML Based Thresholds']

            type_of_graphs = ['ML Based Thresholds']
            for graph_type in type_of_graphs:
                fig, ax = plt.subplots(figsize=(10, 8))

                latest_data_point_index = -1
                latest_data_point_color = 'r'
                prior_data_points_color = 'gray'

                forecast = forecast.merge(df_holidays_manual[['ds', 'holiday']], on='ds', how='left')

                forecast['Legend'] = forecast.apply(lambda row: 'Holiday' if isinstance(row['holiday'], str)else ('Pass' if row['y'] >= row['yhat_lower'] and row['y'] <= row['yhat_upper'] else 'Fail'),axis=1)

            

                holidays_data = forecast[forecast['Legend'] == 'Holiday']
                pass_data = forecast[forecast['Legend'] == 'Pass']
                fail_data = forecast[forecast['Legend'] == 'Fail']

                holidays_data['rca_text'] = holidays_data['holiday'].copy()
                fail_data['rca_text'] = ''

                ax.plot(pass_data['ds'].values, pass_data['y'].values, 'o', color=prior_data_points_color,
                        markersize=2,
                        label='Pass Data')
                ax.plot(fail_data['ds'].values, fail_data['y'].values, 'o', color=latest_data_point_color,
                        markersize=4,
                        label='Fail Data', zorder=5)
                ax.plot(holidays_data['ds'].values, holidays_data['y'].values, 'o', color='blue', markersize=4,
                        label='Holidays', zorder=5)

                fail_data = pd.concat([fail_data, holidays_data], ignore_index=True)

                for index, row in fail_data.iterrows():
                    if not row['rca_text']:
                        filtered_data.rename(columns={'Date': 'ds', METRIC_COLUMN: 'y'}, inplace=True)
                        max_date = row['ds']
                        rca_input_df = filtered_data_all.copy()
                        if row['y'] > row['yhat_upper']:
                            failureType = 'Upper'
                        else:
                                failureType = 'Lower'



                        rca_input_df.rename(columns={'y': METRIC_COLUMN}, inplace=True)
                        
                        rca_text = custom_rca_for_each_week(max_date, rca_input_df, METRIC_COLUMN, metric_val,failureType,rca_column)
                        fail_data.at[index, 'rca_text'] = rca_text

                # convert the list into string
                trend_break_string = ", ".join(Trend_Break)
                latest_week_string = ", ".join(Latest_Week)

                # set the color scale
                color_scale = alt.Scale(domain=['Fail', 'Pass', 'Holiday'], range=['red', 'grey', 'blue'])

                # Plot the data for rca
                rac = alt.Chart(fail_data).mark_circle().encode(
                    x='ds:T',
                    y='y:Q',
                    color=alt.Color('Legend:N', scale=color_scale),
                    tooltip=[alt.Tooltip('rca_text:N', title="  ")]
                ).properties(
                    width=800,
                    height=400,
                    title={
                        'text': f"{trend_break_string} Trend Break has been observed for {latest_week_string} in latest week",
                        'anchor': 'start'
                    }
                )
                # Plot the data for holiday
                holiday = alt.Chart(holidays_data).mark_circle().encode(
                    x='ds:T',
                    y='y:Q',
                    color=alt.Color('Legend:N', scale=color_scale),
                    tooltip=[alt.Tooltip('holiday:N', title='Holiday')]
                )

                # Plot the data for pass
                pass_chart = alt.Chart(pass_data).mark_circle().encode(
                    x='ds:T',
                    y='y:Q',
                    color=alt.Color('Legend:N', scale=color_scale),
                    tooltip=[alt.Tooltip('ds:T', title='Date'), alt.Tooltip('y:Q', title='Value')]
                )

                # Add ML-based bounds
                ml_based_bounds = alt.Chart(forecast).mark_area(
                    color='blue',
                    opacity=0.2
                ).encode(
                    x=alt.X('ds:T', title='Date',axis=alt.Axis(format='%b %d')),
                    y=alt.Y('yhat_lower:Q', title='Metric'),
                    y2=alt.Y2('yhat_upper:Q', title='Volume')
                )


                combined_chart = alt.layer(rac, ml_based_bounds, holiday, pass_chart).configure_axis(grid=False)
                

                if graph_type == 'ML Based Thresholds':
                    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                                    color='blue', alpha=0.2, label='ML Based Bounds')
                elif graph_type == 'Manual Thresholds':
                    ax.fill_between(forecast['ds'], forecast['5%_decrease'], forecast['5%_increase'],
                                    color='red', alpha=0.2, label='Calc Interval')

                    # ax.plot(forecast['ds'], forecast['yhat'], color='grey', label='Trend')
                ax.set_xlabel('Date')
                ax.set_ylabel('Value')

                lower_buffer = 100
                buffer = 200

                columns_to_check = ['yhat_upper', 'y']
                max_value_specific_columns = forecast[columns_to_check].max().max()

                ax.set_ylim(min(forecast['yhat_lower']) - lower_buffer, max_value_specific_columns + buffer)

                # ax.legend()

                legend = ax.legend(loc='lower right')

                # Set legend font size
                font_size = 5  # Change font size as needed
                for text in legend.get_texts():
                    text.set_fontsize(font_size)

                plt.title(str(result_string))
            # gRaph code ends here
            ################################################################################

            graphNumber = graphNumber + 1

        count_outside_percent_interval = \
            forecast[
                (forecast['y'] < forecast['5%_decrease']) | (forecast['y'] > forecast['5%_increase'])].shape[0]
        count_outside_forecast = \
            forecast[(forecast['y'] < forecast['yhat_lower']) | (forecast['y'] > forecast['yhat_upper'])].shape[
                0]
        selected_row = forecast.loc[forecast['ds'].idxmax()]

        # append to new dataframe
        forecastAppend = forecast[['ds', 'yhat_lower', 'yhat_upper', 'y']].copy()

        forecastAppend['Granularity'] = result_string
        sr2 = forecastAppend.iloc[forecastAppend['ds'].idxmax()]

        appendReal = pd.concat([appendReal, sr2.to_frame().transpose()], ignore_index=True)

        if selected_row['yhat_lower'] <= selected_row['y'] <= selected_row['yhat_upper']:

            latest_data_point_color = 'g'
            print(f"Completed Time series Analysis for Combination - {selected_combination} with Result: ",
                    end="");
            display(HTML("<font color='green'><b>PASS</b></font>"))


        else:
            latest_data_point_color = 'r'
      
        if selected_row['yhat_lower'] <= selected_row['y'] <= selected_row['yhat_upper']:
            pass
        else:
            brand_volume_continer.write(result_string)
            brand_volume_continer.altair_chart(combined_chart, use_container_width=True)
            # container_Brand_volume.write(rca_text)
            # st.write(result_string)
            # st.altair_chart(combined_chart, use_container_width=True)
 
       








        return appendReal
    except:
        pass

def unique_sorted_values_plus_ALL(array):
    unique = array.unique().tolist()
    unique.sort()
    return unique

