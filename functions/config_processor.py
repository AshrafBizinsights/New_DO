

def process_config_row(row):
    TABLE_NAME = row['Table Name']
    FILTER_TO_BE_APPLIED = row['Filter']
    DATE_COLUMN_IN_DATASET = row['Date_Field']
    
    AGGREGATE_COLUMNS = row['Aggregate Columns']
    PIVOT_FLAG = row['Pivot Down']
    columns_for_freshness = row['Data Freshness'].split(';')
    columns_for_volume = row['Data Volume'].split(';')
    columns_for_not_nulls = row['Not Null'].split(';')
    columns_to_check_uniqueness = row['Uniqueness'].split(';')
    columns_for_RCA_analysis = row['RCA Analysis Columns'].split(';')
    AGGREGATE_COLUMNS_RAW = row['Dimension Patterns'].split(';')
    compund_key_columns_list = row['Table Level Uniqueness'].split(';')
    rca_column = row["RCA Analysis Columns"].split(';')

    if PIVOT_FLAG == 'No':
        METRIC_COLUMNS = row['Dimension Patterns Metric'].split(';')
    else:
        METRIC_COLUMNS = row['Aggregate Metric'].split(';')

    metric_column_name = row['Metric Column Name']
    LIST_OF_AGGREGATE_COLUMNS = AGGREGATE_COLUMNS.split(';')
    
    # Returning the values to use them elsewhere
    return {
        "TABLE_NAME": TABLE_NAME,
        "FILTER_TO_BE_APPLIED": FILTER_TO_BE_APPLIED,
        "DATE_COLUMN_IN_DATASET": DATE_COLUMN_IN_DATASET,
        "AGGREGATE_COLUMNS": AGGREGATE_COLUMNS,
        "PIVOT_FLAG": PIVOT_FLAG,
        "columns_for_freshness": columns_for_freshness,
        "columns_for_volume": columns_for_volume,
        "columns_for_not_nulls": columns_for_not_nulls,
        "columns_to_check_uniqueness": columns_to_check_uniqueness,
        "columns_for_RCA_analysis": columns_for_RCA_analysis,
        "AGGREGATE_COLUMNS_RAW": AGGREGATE_COLUMNS_RAW,
        "compund_key_columns_list": compund_key_columns_list,
        "rca_column": rca_column,
        "METRIC_COLUMNS": METRIC_COLUMNS,
        "metric_column_name": metric_column_name,
        "LIST_OF_AGGREGATE_COLUMNS": LIST_OF_AGGREGATE_COLUMNS
    }
