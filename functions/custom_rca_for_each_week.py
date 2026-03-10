
import pandas as pd
from datetime import timedelta
import streamlit  as st

def get_dimension_changes(filtered_data_8_weeks, metric_column_name, columnName, metric_rca_val,failureType,rca_column):
    global RCAdf
    columns_columnDimension = [columnName, 'ds', metric_column_name]
    filtered_data_columnDimension = filtered_data_8_weeks[columns_columnDimension]
    filtered_data_columnDimension[metric_column_name] = pd.to_numeric(
        filtered_data_columnDimension[metric_column_name], errors='coerce')

    filtered_data_columnDimension = \
        filtered_data_columnDimension.groupby(['ds', columnName], as_index=False)[metric_column_name].sum()
    

    grouped_data_columnDimension = filtered_data_columnDimension.groupby('ds')[
        metric_column_name].sum().reset_index()

    filtered_data_columnDimension = pd.merge(filtered_data_columnDimension, grouped_data_columnDimension,
                                                on='ds', suffixes=('', '_total'))
    

    metricTotal = metric_column_name + '_total'

    filtered_data_columnDimension[metricTotal] = pd.to_numeric(filtered_data_columnDimension[metricTotal],
                                                                errors='coerce')
    filtered_data_columnDimension['pct'] = filtered_data_columnDimension[metric_column_name] / \
                                            filtered_data_columnDimension[metricTotal] * 100
    

    filtered_data_columnDimension['ds'] = pd.to_datetime(filtered_data_columnDimension['ds'])

    most_recent_week_start = filtered_data_columnDimension['ds'].max() - pd.DateOffset(
        days=filtered_data_columnDimension['ds'].max().dayofweek)

    filtered_data_recent = filtered_data_columnDimension[
        filtered_data_columnDimension['ds'] >= most_recent_week_start]
    filtered_data_history = filtered_data_columnDimension[
        filtered_data_columnDimension['ds'] < most_recent_week_start]

    filtered_data_history = filtered_data_history.groupby([columnName], as_index=False)['pct'].mean()

    filtered_data_recent = filtered_data_recent[[columnName, 'pct']]


    filtered_data_final = pd.merge(filtered_data_recent, filtered_data_history, on=columnName,suffixes=('_recent', '_history'))

    filtered_data_final['abs'] = filtered_data_final['pct_recent'] - filtered_data_final['pct_history']

    if failureType == 'Upper':

        filtered_data_final = filtered_data_final.sort_values(by='abs', ascending=False)
    else:
        filtered_data_final = filtered_data_final.sort_values(by='abs', ascending=True)


    #st.write(filtered_data_final)

    filtered_data_columnDimension = filtered_data_final.head(1)
    # filtered_data_columnDimension = filtered_data_final.copy()

    filtered_data_columnDimension = filtered_data_columnDimension.rename(columns={columnName: 'List_Value'})

    filtered_data_columnDimension['Column Name'] = columnName

    filtered_data_columnDimension['Metric Val'] = metric_rca_val

    filtered_data_columnDimension['change_text'] = filtered_data_columnDimension.apply(change_text, axis=1)




    if len(filtered_data_columnDimension) > 0:
        for index, row in filtered_data_columnDimension.iterrows():
            return row['change_text'], abs(row['abs'])
    else: 
        return 'No Major changes detected', 0

    # failureText = ''
    # for text_val in filtered_data_columnDimension['change_text'].unique():
    #     if failureText:
    #         failureText = failureText + '\n' + text_val
    #     else:
    #         failureText = text_val
    # return failureText

    #return filtered_data_columnDimension['change_text'], row['abs']


def perform_root_cause(latest_8_weeks_data, metric_column_name, metric_rca_val,failureType,rca_column):

    textOpFinal = ''
    val1 = 0
    for column in rca_column:
        textOp, val = get_dimension_changes(latest_8_weeks_data, metric_column_name, column, metric_rca_val,failureType,rca_column)

        if val > val1:
            textOpFinal = textOp
            val1 = val
        

        


        # if textOpFinal:
        #     textOpFinal = textOpFinal + '\n' + '\n' + textOp
        # else:
        #     textOpFinal = textOp
    return textOpFinal

def custom_rca_for_each_week(max_date, failed_records, METRIC_COLUMN, metric_rca_val,failureType,rca_column):

    # st.write('*****************************************************************************')

    rcaDate = str(max_date).replace(' 00:00:00', '')
    bold_text = f"**RCA analysis for Anomaly {rcaDate}**"
    # st.write(bold_text)

    start_date_8_weeks_ago = max_date - timedelta(weeks=8)
    latest_8_weeks_data = failed_records[(failed_records['ds'] > start_date_8_weeks_ago) & (failed_records['ds'] < max_date)].copy()
    textDict = {'d1': 'dval'}
    RCAdf = pd.DataFrame()

    RCAdf = pd.DataFrame(columns=['List_Value', 'pct_recent', 'pct_history', 'abs', 'Column Name', 'Metric Val'])
    #columns_for_RCA_analysis = ['SP_SOURCE','INDICATION']

    RCADF = perform_root_cause(latest_8_weeks_data, METRIC_COLUMN, metric_rca_val,failureType,rca_column)

    return RCADF


def change_text(row):
    change_amount = abs(round(row['abs'], 1))
    if change_amount > 2:
        if row['pct_recent'] > row['pct_history']:
            return f"{row['Column Name']} {row['List_Value']} saw a {change_amount}% increase in its contribution to volume compared to the previous week (R8W)."
        else:
            return f"{row['Column Name']} {row['List_Value']} saw a {change_amount}% decrease in its contribution compared to the previous week (R8W)."
    else:
        return "Could not identify the potential reason for anomaly"