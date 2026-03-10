import pandas as pd

def categorize_columns(df):
    ID = []
    cat = []
    num = []
    Date = []

    # Helper functions
    def is_useful_categorical(df, col):
        unique_values = df[col].nunique()
        total_values = len(df)
        return df[col].dtype in ['object', 'category'] and 1 < unique_values < 0.5 * total_values

    def is_useful_numerical(df, col):
        unique_values = df[col].nunique()
        return df[col].dtype in ['float64'] and 1 < unique_values < 0.9 * len(df)

    def is_id_column(df, col):
        unique_values = df[col].nunique()
        total_values = len(df)
        return unique_values == total_values or (unique_values >= 0.8 * total_values and df[col].dtype != 'float64')

    def is_date_column(df, col):
        return pd.api.types.is_datetime64_any_dtype(df[col])

    # Categorize columns
    for col in df.columns:
        if is_date_column(df, col):
            Date.append(col)
        elif is_id_column(df, col):
            ID.append(col)
        elif is_useful_categorical(df, col):
            cat.append((col, df[col].nunique()))  # Append as a tuple with unique count
        elif is_useful_numerical(df, col):
            num.append(col)

    # Sort categorical columns by number of unique values
    cat.sort(key=lambda x: x[1] , reverse=True)  # Sort by the unique value count
    cat = [col[0] for col in cat]  # Extract only the column names

    return {
        "ID": ID,
        "categorical": cat,
        "numerical": num,
        "date": Date
    }


custom_order = ["markdown", "dataframe", "altair_chart"]
def render_element(element_type, continer):
    global styled_df,final_chart
    if element_type == "dataframe":
        continer.dataframe(styled_df, width=1700, hide_index=True)
    elif element_type == "altair_chart":
        continer.altair_chart(final_chart, use_container_width=True)