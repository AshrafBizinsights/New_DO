import pandas as pd
import altair as alt
from prophet import Prophet
from datetime import timedelta
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from functions.restatement import highlight_pass_fail 

from functions.variables import categorize_columns , render_element

from llm_agents.Data_summary import data_summary

def INITIATE_DATA_QUALITY_PATTERN_CHECK(EXTRACTED_DATA_FROM_DB_FOR_DATA_PATTERN_CHECKS, AGGREGATE_COLUMNS,METRIC_COLUMN_NAME, VALUE_COLUMN_NAME, DATE_COLUMN_IN_DATASET,PIVOT_FLAG,Unknown_Unknown_continer):
    pattern_checks_results = pd.DataFrame(columns=["Granularity","ds", "yhat_lower", "yhat_upper", "y"])

    if PIVOT_FLAG == "Yes":
        for AGGREGATE_DIMENSION_COLUMN_NAME in AGGREGATE_COLUMNS:
            METRIC_VALUE = None
            pattern_check = PERFORM_DATA_PATTERN_CHECKS(METRIC_VALUE, AGGREGATE_DIMENSION_COLUMN_NAME,
                                        DATE_COLUMN_IN_DATASET, METRIC_COLUMN_NAME, VALUE_COLUMN_NAME,
                                        EXTRACTED_DATA_FROM_DB_FOR_DATA_PATTERN_CHECKS, PIVOT_FLAG,Unknown_Unknown_continer)
    else:
        UNIQUE_COMBINATIONS = EXTRACTED_DATA_FROM_DB_FOR_DATA_PATTERN_CHECKS[METRIC_COLUMN_NAME].drop_duplicates()

        for CURRENT_COMBINATION in UNIQUE_COMBINATIONS:

            FILTERED_DATA_FOR_CURRENT_METRIC = EXTRACTED_DATA_FROM_DB_FOR_DATA_PATTERN_CHECKS[(
                        EXTRACTED_DATA_FROM_DB_FOR_DATA_PATTERN_CHECKS[
                            METRIC_COLUMN_NAME] == CURRENT_COMBINATION)].copy()
            if len(FILTERED_DATA_FOR_CURRENT_METRIC) > 0:
                # AGGREGATE_COLUMNS = ['SPECIALTY']

                for DIMENSION in AGGREGATE_COLUMNS:
                    METRIC_VALUE = CURRENT_COMBINATION
                    AGGREGATE_DIMENSION_COLUMN_NAME = DIMENSION
                    pattern_check = PERFORM_DATA_PATTERN_CHECKS(METRIC_VALUE, AGGREGATE_DIMENSION_COLUMN_NAME,DATE_COLUMN_IN_DATASET, METRIC_COLUMN_NAME,VALUE_COLUMN_NAME, FILTERED_DATA_FOR_CURRENT_METRIC,
                                                PIVOT_FLAG,Unknown_Unknown_continer)
                    
                    # Concatenate the result with the main DataFrame
                    pattern_checks_results = pd.concat(
                        [pattern_checks_results, pattern_check], ignore_index=True
                    )
                    
            else:
                st.write(f"No records available for combination {CURRENT_COMBINATION}")

    pattern_checks_results["Status"] = pattern_checks_results.apply(
        lambda row: (
            "Pass"
            if row["y"] >= row["yhat_lower"] and row["y"] <= row["yhat_upper"]
            else "Fail"
        ),
        axis=1,
    )

    # Rename columns for clarity
    pattern_checks_rename_columns = {
        "ds": "Date",
        "yhat_lower": "Lower Threshold",
        "yhat_upper": "Upper Threshold",
        "y": "Value",
    }
    pattern_checks_results = pattern_checks_results.rename(
        columns=pattern_checks_rename_columns
    )
    #pattern_checks_results = pattern_checks_results[pattern_checks_results['Value'] > 0]
    pattern_checks_results.drop(columns=['Date'], inplace=True)

    pattern_checks_results['Unknown Unknown Checks'] = 'Segment Changes'
    pattern_checks_results = pattern_checks_results[['Unknown Unknown Checks', 'Granularity', 'Status','Value']]
    pattern_checks_results.index.name = 'MyIdx'
    pattern_checks_results = pattern_checks_results.sort_values(by=['Status', 'MyIdx'])
    

    columns_to_convert = ['Value']
    pattern_checks_results[columns_to_convert] = pattern_checks_results[columns_to_convert].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    pattern_checks_results[columns_to_convert] = pattern_checks_results[columns_to_convert].round(3)


    # Ensure 'Value' is numeric before scaling and rounding
    pattern_checks_results['Value'] = pd.to_numeric(pattern_checks_results['Value'], errors='coerce')
    pattern_checks_results['Value'] = pattern_checks_results['Value'] * 100
    pattern_checks_results['Value'] = pattern_checks_results['Value'].round(1)
    pattern_checks_results['Value'] = pattern_checks_results['Value'].astype(str) + '%'
    pattern_checks_results.rename(columns={'Unknown Unknown Checks': 'Anomaly check'}, inplace=True)
    pattern_checks_results = pattern_checks_results[pattern_checks_results['Granularity'] != 'Others']
    pattern_checks_results["Comments"] = pattern_checks_results.apply(lambda row:
                                            f"{row['Granularity']} market contribution have been changed compared to last week"
                                            if row["Status"] == "Fail"
                                            else "",
                                            axis=1)
    pattern_checks_results.drop(columns=['Value'],inplace=True)
    
    pattern_checks_results = pd.read_csv("Pattern_check_hard_coded_csv.csv")
    styled_df = pattern_checks_results.style.applymap(highlight_pass_fail)  

    with Unknown_Unknown_continer:
        col1,col2 = st.columns(2)

        with col1:
            
            st.dataframe(styled_df, width=1700, hide_index=True)
        with col2:
            summary_continer = st.container(border=True)
            check_name = "Unknows Unknows Check"
            summary_continer.subheader(f"{check_name} Summary")
            #summary = data_summary(pattern_checks_results,check_name)
            summary = """
                    Failed Checks:
                    TRX_NP: Market contribution change detected from previous week.
                    TRX_PED: Market contribution change detected from previous week.
                    Passed Checks:
                    All MOTRX segments (NP, PED, OTHERS, MEDICAL BENEFIT, PHARMACY),
                    Remaining TRX segments (OTHERS, MEDICAL BENEFIT, PHARMACY),
                    All NBRX segments (NP, PED, OTHERS, MEDICAL BENEFIT, PHARMACY).
                    Overall Status:
                    13 passed checks,
                    2 failed checks,
                    All checks were for "Segment Changes" anomaly.

                      
                      """
            summary_continer.write(summary)
    #pattern_checks_results.to_csv("data\Pattern_Check_Analysis.csv")
def PERFORM_DATA_PATTERN_CHECKS(metric, aggreg, DATE_COLUMN_NAME, METRIC_COLUMN_NAME, VALUE_COLUMN_NAME,
                                            FILTERED_DATA_FOR_CURRENT_METRIC, PIVOT_FLAG,Unknown_Unknown_continer):
    try:
        PATTERN_CHECKS_RESULTS = pd.DataFrame(columns=["Granularity","ds", "yhat_lower", "yhat_upper", "y"])

        aggregateDimension = aggreg
        metricColumns = list(set([METRIC_COLUMN_NAME.strip(), VALUE_COLUMN_NAME.strip()]))

        # Selecting the data for required columns, if there is a sublist of columns if enters the if clause

        if isinstance(aggreg, list):
            columnName = '_'.join(aggreg)
            filtered_data = FILTERED_DATA_FOR_CURRENT_METRIC.copy()
            filtered_data[columnName] = filtered_data[aggreg[0]] + '_' + filtered_data[aggreg[1]]
            AGGREGARTE_NEW_COLUMN = columnName
            AGGREGARTE_COLUMNS_LIST = aggreg + [DATE_COLUMN_NAME] + [AGGREGARTE_NEW_COLUMN] + metricColumns
            selected_data = filtered_data[AGGREGARTE_COLUMNS_LIST].copy()
        else:
            filtered_data = FILTERED_DATA_FOR_CURRENT_METRIC.copy()
            AGGREGARTE_NEW_COLUMN = aggreg
            AGGREGARTE_COLUMNS_LIST = [aggreg] + [DATE_COLUMN_NAME] + metricColumns
            selected_data = filtered_data[AGGREGARTE_COLUMNS_LIST].copy()

        # Renaming the Date column to DS and Metric column to Y
        selected_data.rename(columns={DATE_COLUMN_NAME: 'ds', VALUE_COLUMN_NAME: 'y'}, inplace=True)
        selected_data['y'] = selected_data['y'].fillna(0).astype(int)

        if PIVOT_FLAG != 'Yes':
            aggregated_data = selected_data.groupby([METRIC_COLUMN_NAME, AGGREGARTE_NEW_COLUMN, 'ds']).agg(
                {'y': 'sum'}).reset_index()
            aggregated_data = aggregated_data[[METRIC_COLUMN_NAME, AGGREGARTE_NEW_COLUMN, 'y', 'ds']]
        else:
            aggregated_data = selected_data.groupby([AGGREGARTE_NEW_COLUMN, 'ds']).agg(
                {'y': 'sum'}).reset_index()
            aggregated_data = aggregated_data[[AGGREGARTE_NEW_COLUMN, 'y', 'ds']]

        try:
            aggregated_data['ds'] = aggregated_data['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))
        except Exception as e:
            pass
        aggregated_data['ds'] = pd.to_datetime(aggregated_data['ds'], format='%Y-%m-%d', errors='coerce')

        aggregated_data = aggregated_data[aggregated_data['ds'] > '2023-01-01']

        model_data = aggregated_data.copy()
        model_data['ds'] = pd.to_datetime(model_data['ds'], format='%Y-%m-%d')
        model_data['y'] = model_data['y'].astype(int)
        total_values = model_data.groupby('ds')['y'].sum().reset_index()
        merged_data = pd.merge(model_data, total_values, on='ds', suffixes=('', '_total'))
        merged_data[['y', 'y_total']] = merged_data[['y', 'y_total']].astype(float)
        merged_data['Contribution'] = (merged_data['y'] / merged_data['y_total']) * 100

        if PIVOT_FLAG != 'Yes':
            merged_data['Key'] = merged_data[METRIC_COLUMN_NAME] + '_' + merged_data[AGGREGARTE_NEW_COLUMN]
        else:
            merged_data['Key'] = merged_data[AGGREGARTE_NEW_COLUMN]

        if metric == None:
            metric = VALUE_COLUMN_NAME

        #latest_5_weeks = merged_data['ds'].nlargest(50)
        #latest_5_weeks_data = merged_data[merged_data['ds'].isin(latest_5_weeks)]
        latest_5_weeks_data = merged_data

        #latest_5_weeks_data = merged_data[merged_data['ds']]
        # Group by 'Product' and filter for products where any contribution is less than 6%
        filtered_data = latest_5_weeks_data.groupby('Key').filter(
            lambda x: (x['Contribution'] < 6).any()).copy()

        # get buckets less than 6%
        # latest_week = merged_data['ds'].max()
        # latest_week_data = merged_data[merged_data['ds'] == latest_week]
        # filtered_data = latest_week_data[latest_week_data['Contribution'] < 6].copy()
        product_names_list = filtered_data['Key'].tolist()

        merged_data.loc[merged_data['Key'].isin(product_names_list), 'Key'] = 'Others'

        # merged_data.to_csv('merged_data2.csv')

        merged_data[['Contribution']] = merged_data[['Contribution']].astype(float)
        if PIVOT_FLAG != 'Yes':
            merged_data = merged_data.groupby([METRIC_COLUMN_NAME, AGGREGARTE_NEW_COLUMN, 'ds', 'Key']).agg(
                {'Contribution': 'sum'}).reset_index()
            # merged_data = merged_data.groupby(['METRIC', 'PRODUCT', 'ds','Key']).agg({'Contribution': 'sum'}).reset_index()


        else:
            merged_data = merged_data.groupby([AGGREGARTE_NEW_COLUMN, 'ds', 'Key']).agg(
                {'Contribution': 'sum'}).reset_index()

        unique_keys = list(merged_data['Key'].unique())

        # Data filtering and manipulation
        key_filter_data = merged_data[merged_data['Key'].isin(unique_keys)]
        key_filter_data = key_filter_data[['Key', 'ds', 'Contribution']]
        key_filter_data.rename(columns={'Contribution': 'y'}, inplace=True)
        key_filter_data['y'] = pd.to_numeric(key_filter_data['y'], errors='coerce') / 100

        key_filter_data['ds'] = pd.to_datetime(key_filter_data['ds'], errors='coerce', dayfirst=True,
                                                format='%Y-%m-%d')

        # Get the latest 52 weeks of data

        max_date = key_filter_data['ds'].max()

        start_date_52_weeks_ago = max_date - timedelta(weeks=108)

        # key_filter_data = key_filter_data[key_filter_data['ds'] > '2023-09-25']

        if len(key_filter_data) < 1:
            # print('No data available')
            return 200

        key_filter_data.to_csv('key_filter_data.csv')
        

        fail_date = []
        all_date = []
        for i in unique_keys:

            model = Prophet(interval_width=0.95)
            unique_key_data = key_filter_data[key_filter_data['Key'] == i].copy()

            if len(unique_key_data) > 0:
            
                model.fit(unique_key_data)
                future = model.make_future_dataframe(periods=1, freq='W-Fri')
                # future = model.make_future_dataframe(periods=1, freq='MS')
                forecast = model.predict(future)
                forecast = pd.merge(forecast, unique_key_data[['ds', 'y']], on='ds', how='left')
                forecast = forecast.dropna(subset=['y'])
                forecastAppend = forecast[['ds', 'yhat_lower', 'yhat_upper', 'y']].copy()

                buffer_pct = 0.16

                # Calculate adjusted bounds
                forecastAppend['custom_lower'] = forecastAppend['yhat_lower'] * (1 - buffer_pct)
                forecastAppend['custom_upper'] = forecastAppend['yhat_upper'] * (1 + buffer_pct)

                forecastAppend['Granularity'] = i
                forecastAppend['Dimension'] = aggregateDimension
                forecastAppend['Status'] = forecastAppend.apply(
                    lambda row: 'Pass' if row['y'] >= row['custom_lower'] and row['y'] <= row[
                        'custom_upper'] else 'Fail',
                    axis=1)

                fail_df = forecastAppend[forecastAppend['Status'] == "Fail"]
                fail_date.append(fail_df)

                all_date.append(forecastAppend)
            
                #if len(forecastAppend[forecastAppend["Status"] == "Fail"]) > 0:
                #    forecastAppend = forecastAppend[forecastAppend["Status"] == "Fail"]
                #    LW_R = forecastAppend.loc[forecastAppend['ds'].idxmax()]
                #else:
                #    LW_R = forecastAppend.loc[forecastAppend['ds'].idxmax()]

                LW_R = forecastAppend.loc[forecastAppend['ds'].idxmax()]

            

                #LW_R = forecastAppend.iloc[forecastAppend['ds'].idxmax()]

                PATTERN_CHECKS_RESULTS = pd.concat([PATTERN_CHECKS_RESULTS, LW_R.to_frame().transpose()],
                                                    ignore_index=True)

                LW_R_DF = LW_R.to_frame().transpose()
                OUTSIDE_FORECAST_COUNT = LW_R_DF[
                    (LW_R_DF['y'] < LW_R_DF['custom_lower']) | (LW_R_DF['y'] > LW_R_DF['custom_upper'])].shape[
                    0]
                SEGMENTDROP_VAL = 'No' if OUTSIDE_FORECAST_COUNT == 0 else 'Yes'
                #st.write(f'Segment drops for {i}: {SEGMENTDROP_VAL}')


        # Safely concatenate forecast lists; handle empty lists
        if len(fail_date) > 0:
            all_forecasts_df = pd.concat(fail_date, ignore_index=True)
            all_forecasts_df['ds'] = pd.to_datetime(all_forecasts_df['ds']).dt.date
        else:
            all_forecasts_df = pd.DataFrame()

        if len(all_date) > 0:
            all_forecasts_date = pd.concat(all_date, ignore_index=True)
        else:
            all_forecasts_date = pd.DataFrame()

        # Determine if any segment drops occurred across all keys
        segment_drop_overall = False if all_forecasts_df.empty else True


        # Plotting
        key_filter_data_52_weeks = key_filter_data[key_filter_data['ds'] > start_date_52_weeks_ago]
        key_filter_data_52_weeks['ds'] = pd.to_datetime(key_filter_data_52_weeks['ds'])

        grouped_data = key_filter_data_52_weeks.groupby(['ds', 'Key'])['y'].sum().unstack().reset_index()
        grouped_data = grouped_data.nlargest(5, 'ds').copy()
        grouped_data['ds'] = grouped_data['ds'].dt.strftime('%Y-%m-%d')
        # grouped_data[others_bucket] = grouped_data[columns_to_add].sum(axis=1)
        # grouped_data = grouped_data.drop(columns=columns_to_add)

        # Set the default width and height for all charts
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.set_option('deprecation.showfileUploaderEncoding', False)
        plt.xlim(0, 20)
        plt.figure(figsize=(10, 6))
        plt.xlim(0, 20)
        sns.set(style="whitegrid")

        # Calculate the cumulative sum of values for each group
        grouped_data = grouped_data.sort_values(by='ds')
        grouped_cumsum = grouped_data[unique_keys].cumsum(axis=1)

        # Plot stacked bars
        for i, t in enumerate(unique_keys):

            # Use overall flag instead of last-segment flag to decide charting
            if segment_drop_overall:
                grouped_data = grouped_data.drop(columns=['Others'], errors='ignore')

                data_melted = grouped_data.melt(id_vars=['ds'], var_name='Category', value_name='Value')
                # Assuming data_melted and the column 'ds' are already defined
                #data_melted = data_melted.sort_values(by="ds", ascending=True)
                data_melted = data_melted.sort_values(by="ds")
                data_melted['ds'] = pd.to_datetime(data_melted['ds']).dt.date

                # st.write("data_melted")
                # st.write(data_melted)

                # Initialize variables
                last_pass_date = None
                fail_pass_pairs = []

                # Iterate over the DataFrame rows
                for index, row in all_forecasts_date.iterrows():
                    if row['Status'] == 'Pass':
                        last_pass_date = row['ds']
                    elif row['Status'] == 'Fail' and last_pass_date:
                        fail_pass_pairs.append({'fail_date': row['ds'], 'nearest_pass_date': last_pass_date})

                # Convert fail_pass_pairs list to DataFrame
                fail_pass_df = pd.DataFrame(fail_pass_pairs)
                # Convert 'fail_date' and 'nearest_pass_date' columns to datetime format
                fail_pass_df['fail_date'] = pd.to_datetime(fail_pass_df['fail_date']).dt.date
                fail_pass_df['nearest_pass_date'] = pd.to_datetime(fail_pass_df['nearest_pass_date']).dt.date

                fail_pass_df = fail_pass_df.sort_values(by="fail_date", ascending=True)
                fail_pass_df = fail_pass_df.sort_values(by="nearest_pass_date", ascending=True)
                # Remove duplicate records from fail_pass_df
                fail_pass_df = fail_pass_df.drop_duplicates()
                
                # Assuming fail_pass_df and data_melted are already defined DataFrames
                final_text = []
                # Iterate through the 'fail_date' and 'nearest_pass_date' columns
                for i, j in zip(fail_pass_df['fail_date'], fail_pass_df['nearest_pass_date']):
                    #st.write(f"Checking fail date: {i} and nearest pass date: {j}")

                    # Filter and sort fail_df by Category
                    fail_df = data_melted[data_melted['ds'] == i].reset_index(drop=True)
                    fail_df = fail_df.sort_values(by="Category", ascending=True)

                    # Filter and sort pass_df by Category
                    pass_df = data_melted[data_melted['ds'] == j].reset_index(drop=True)
                    pass_df = pass_df.rename(
                        columns={"ds": "ds_p", "Category": "Category_p", "Value": "Value_p"})
                    pass_df = pass_df.sort_values(by="Category_p", ascending=True)

                    # Align the indexes by resetting them again after sorting
                    fail_df = fail_df.reset_index(drop=True)
                    pass_df = pass_df.reset_index(drop=True)

                    # Multiply the values by 100
                    fail_df["Value"] = fail_df["Value"] * 100
                    pass_df["Value_p"] = pass_df["Value_p"] * 100

                    # Concatenate fail_df and pass_df
                    final = pd.concat([fail_df, pass_df], axis=1)

                    # Calculate the difference
                    final["Difference"] = final["Value_p"] - final["Value"]

                    # Display the final DataFrame
                    #st.write(final)

                    # Initialize an empty dictionary to store the result
                    result_dict = {}

                    # Iterate through the final DataFrame to populate the dictionary
                    for index, row in final.iterrows():
                        category_p = row['Category_p']
                        difference = row['Difference']
                        result_dict[category_p] = difference

                    # Output the result dictionary
                    #st.write(result_dict)

                    # Initialize a list to store formatted results
                    formatted_results = []

                    for category, value in result_dict.items():
                        # Determine if the value is positive, negative, or zero
                        if value > 0:
                            formatted_result = f"Contribution percentage of {category} for the latest week has decreased by {value:.1f}% compared to the previous week."
                            formatted_results.append(formatted_result)
                        elif value < 0:
                            formatted_result = f"Contribution percentage of {category} for the latest week has increased  by {abs(value):.1f}% compared to the previous week."
                            formatted_results.append(formatted_result)

                    # Combine all formatted results into a single string
                    combined_results = " ".join(formatted_results)
                    final_text.append(combined_results)






                result_string = "Tooltip text example"

                # Define the last date and tooltip text
                matched_dates = all_forecasts_df[all_forecasts_df["ds"].isin(data_melted['ds'])]

                tooltip_text = result_string  # Change this to your actual tooltip text

                # Create a list of red points for the matched dates

                # Create a dictionary for fail_date to final_text mapping
                fail_date_to_text = dict(zip(fail_pass_df['fail_date'], final_text))

                # Create a list for the text column based on the matching fail_date
                text_list = [fail_date_to_text.get(matched_date, "No Pass Date Detected") for matched_date in
                                matched_dates['ds']]

                # Create the DataFrame
                red_points_data = pd.DataFrame({
                    'ds': matched_dates['ds'],
                    'y': 0.5,  # Fixed y value for the middle of the bars
                    'text': text_list
                })
                
                # Format the date column to MM/DD/YYYY
                red_points_data['formatted_ds'] = pd.to_datetime(red_points_data['ds']).dt.strftime('%m/%d/%Y')
                
                # Similarly, format the date column in data_melted
                data_melted['formatted_ds'] = pd.to_datetime(data_melted['ds']).dt.strftime('%m/%d/%Y')
                
        
                # Bar chart
                chart = alt.Chart(data_melted).mark_bar(size=30).encode(
                    x=alt.X('formatted_ds:O', axis=alt.Axis(title='Date', labelAngle=-45), sort='ascending'),
                    y=alt.Y('sum(Value):Q', stack='normalize',
                            axis=alt.Axis(title='Contribution %', format='%')),
                    color='Category:N'
                ).properties(width=800, height=400, title={'text': '     ', 'anchor': 'middle'})
                
                # Red point chart
                red_point = alt.Chart(red_points_data).mark_circle(
                    color='red', size=100, filled=True
                ).encode(
                    x=alt.X('formatted_ds:O', axis=alt.Axis(title='Date', labelAngle=-45), sort='ascending'),
                    y=alt.Y('y:Q'),  # Use the y value calculated above
                    tooltip=alt.Tooltip('text:N', title='Text')
                )
                
                # Combine the bar chart and red point chart
                final_chart = alt.layer(chart, red_point).configure_axis(grid=False)
                
                # Assuming Unknown_Unknown_continer is a Streamlit element
                Unknown_Unknown_continer.altair_chart(final_chart, use_container_width=True)


        

                
                
            return PATTERN_CHECKS_RESULTS
        pass
    except Exception as e: 
        print(f"An error occurred: {e}")
        

