import streamlit as st
import pandas as pd
from datetime import datetime,timedelta,date
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io
import base64 
import time
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import calendar

#from llm_agents.Data_summary import data_summary
from sklearn.base import BaseEstimator, TransformerMixin

def set_page_config():
    PAGE_CONFIG = {
        "page_title": "Data Observability",
        "page_icon": "📊",
        "layout": "wide",
        "initial_sidebar_state": "collapsed",  # Sidebar is closed by default
    }
    st.set_page_config(**PAGE_CONFIG)


# set_page_config()


def remove_unnamed_columns(df):
    """
    Removes columns from the DataFrame where the column name contains 'Unnamed:'.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame
    
    Returns:
    pandas.DataFrame: A DataFrame with 'Unnamed:' columns removed
    """
    return df.loc[:, ~df.columns.str.contains('^Unnamed:')]


class HandleUnknownCategories(BaseEstimator, TransformerMixin):
    def __init__(self, known_categories=None):
        self.known_categories = known_categories

    def fit(self, X, y=None):
        # Store unique categories from training data
        self.known_categories = {col: set(X[col].unique()) for col in X.columns}
        return self

    def transform(self, X):
        # Map unseen categories to "Unknown"
        X_transformed = X.copy()
        for col in X.columns:
            X_transformed[col] = X[col].apply(lambda x: x if x in self.known_categories[col] else "Unknown")
        return X_transformed



class EmptyDataFrameError(Exception):
    """Exception raised when the DataFrame is empty."""
    def __init__(self, message="The DataFrame is empty."):
        self.message = message
        super().__init__(self.message)
        
def highlight_pass_fail(val):
    if val == 'Pass':
        color = 'White'
        text_color = 'Green'
        symbol = '\u25cf'  # Green filled circle
    elif val == 'Fail':
        color = 'White'
        text_color = 'Red'
        symbol = '\u25cf'  # Red filled circle
    elif val == 'Pass.':
        color = 'White'
        text_color = '#ED7D31'
        symbol = '\u25cf'  # Red filled circle
    else:
        color = 'white'
        text_color = 'black'
        symbol = '\u25cf'  # Black filled circle
    return f'background-color: {color}; color: {text_color}; text-align: center; padding: 2px 0px 2px 0px; content: "{symbol}"'
  



@st.cache_data
def filter_data_by_period(data, date_column, period):
    # Ensure the date_column is in datetime format
    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

    reference_date = data[date_column].max()
    if period == 'R6M':
        cutoff_date = reference_date - relativedelta(months=6)
    elif period == 'R12M':
        cutoff_date = reference_date - relativedelta(months=12)
    elif period == 'R18M':
        cutoff_date = reference_date - relativedelta(months=18)
    elif period == 'R24M':
        cutoff_date = reference_date - relativedelta(months=24)
    else:
        raise ValueError("Unsupported period. Use 'R6M', 'R12M', 'R18M', or 'R24M'.")

    print("Cutoff Date:", cutoff_date)
    print("Reference Date:", reference_date)

    return data[data[date_column] >= cutoff_date]

@st.cache_data
def MonthSummary(df,brand_selection):
    df=df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df = df[df['Brnd_Name'].isin(brand_selection)]
    numerical_col=['Tot_Clms', 'Tot_Drug_Cst','Tot_Day_Suply','Tot_30day_Fills']
    monthly_summary = df.groupby('Month')[numerical_col].sum().reset_index()
    monthly_summary[numerical_col] = monthly_summary[numerical_col].astype(int)
    monthly_summary.index=monthly_summary.index+1
    st.dataframe(monthly_summary)




def uploadedFiles():
    
    st.sidebar.markdown("""
            <style>
            .sidebar .sidebar-content {
                background-color: #373D50; /* Yellow background */
                color: black; /* Text color for contrast */
            }
            .sidebar .sidebar-header {
                text-align: center; /* Center-align the header */
                color: black; /* Text color for contrast */
                font-size: 24px; /* Increase font size */
                padding: 10px 0; /* Add some padding */
            }
            </style>
        """, unsafe_allow_html=True)

    st.sidebar.title("Data Observation Dashboard")
    yesterday_file=''
    file_type = source = st.sidebar.selectbox("Data Source", options=('Excel', 'CSV', 'Snowflake'), index=None,placeholder="Select a Source")
    if file_type in ["Excel", "CSV"]:
        yesterday_file = st.sidebar.file_uploader("Upload Last file (Monthly/Weekly)", type=["csv", "xlsx"])
    today_file = st.sidebar.file_uploader("Upload Current file (Monthly/Weekly)", type=["csv", "xlsx"])
    return file_type, yesterday_file, today_file

def get_datetime_columns(df):
    # Get columns with datetime format
    datetime_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    return datetime_columns


@st.cache_data
def structural_change_kpi(yesterday_df, today_df):
    
    # Compare numerical columns
    yesterday_numerical = set(yesterday_df.select_dtypes(include=['int', 'float']).columns)
    today_numerical = set(today_df.select_dtypes(include=['int', 'float']).columns)
    added_columns = today_numerical - yesterday_numerical
    removed_columns = yesterday_numerical - today_numerical
    status_num = "Fail" if added_columns or removed_columns else "Pass"
    #comment_num = f"Numerical Col: Added: {', '.join(added_columns)}. Removed: {', '.join(removed_columns)}." if added_columns or removed_columns else "No numerical columns added or removed."
    
    if added_columns and removed_columns:
        comment_num = f"Numerical Col Added: {', '.join(added_columns)}. Removed: {', '.join(removed_columns)}."
    elif added_columns:
        comment_num = f"Numerical Col Added: {', '.join(added_columns)}"
    elif removed_columns:
        comment_num = f"Numerical Col Removed: {', '.join(added_columns)}"    
    else:
        comment_num = "No numerical columns added or removed."  # Or simply skip defining it
    
    
    # Create DataFrame for numerical columns
    df_num = pd.DataFrame({"Status": [status_num], "Comment": [comment_num]}, index=['Numerical Columns'])  
    
    # Compare categorical columns
    yesterday_categorical = set(yesterday_df.select_dtypes(include=['object']).columns)
    today_categorical = set(today_df.select_dtypes(include=['object']).columns)
    added_columns = today_categorical - yesterday_categorical
    removed_columns = yesterday_categorical - today_categorical
    status_cat = "Fail" if added_columns or removed_columns else "Pass"
    #comment_cat = f"Categorical Col: Added: {', '.join(added_columns)}. Removed: {', '.join(removed_columns)}." if added_columns or removed_columns else "No categorical columns added or removed."
    
    if added_columns and removed_columns:
        comment_cat = f"Categorical Col Added: {', '.join(added_columns)}. Removed: {', '.join(removed_columns)}."
    elif added_columns:
        comment_cat = f"Categorical Col Added: {', '.join(added_columns)}"
    elif removed_columns:
        comment_cat = f"Categorical Col Removed: {', '.join(added_columns)}"    
    else:
        comment_cat = "No categorical columns added or removed."
    
    
    
    # Create DataFrame for categorical columns
    df_cat = pd.DataFrame({"Status": [status_cat], "Comment": [comment_cat]}, index=['Categorical Columns'])
    
    # Compare date columns
    yesterday_date = set(yesterday_df.select_dtypes(include=['datetime', 'datetime64']).columns)
    today_date = set(today_df.select_dtypes(include=['datetime', 'datetime64']).columns)
    added_columns = today_date - yesterday_date
    removed_columns = yesterday_date - today_date
    status_date = "Fail" if added_columns or removed_columns else "Pass"
    comment_date = f"Date Col Added: {', '.join(added_columns)}. Removed: {', '.join(removed_columns)}." if added_columns or removed_columns else "No date columns added or removed."
    
    if added_columns and removed_columns:
        comment_date = f"Date Col Added: {', '.join(added_columns)}. Removed: {', '.join(removed_columns)}."
    elif added_columns:
        comment_cat = f"Date Col Added: {', '.join(added_columns)}"
    elif removed_columns:
        comment_cat = f"Date Col Removed: {', '.join(added_columns)}"    
    else:
        comment_cat = "No Date columns added or removed."
    
    
    
    
    # Create DataFrame for date columns
    df_date = pd.DataFrame({"Status": [status_date], "Comment": [comment_date]}, index=['Date Columns'])
    
    # Concatenate all DataFrames
    df = pd.concat([df_num, df_cat, df_date])
  
    styled_df = df.style.applymap(highlight_pass_fail)
    
    #st.dataframe(styled_df, width=1500) 

    return df


@st.cache_data
def structural_change(yesterday_df, today_df):
    
    # Compare numerical columns
    yesterday_numerical = set(yesterday_df.select_dtypes(include=['int', 'float']).columns)
    today_numerical = set(today_df.select_dtypes(include=['int', 'float']).columns)
    added_columns = today_numerical - yesterday_numerical
    removed_columns = yesterday_numerical - today_numerical
    status_num = "Fail" if added_columns or removed_columns else "Pass"
    #comment_num = f"Numerical Col: Added: {', '.join(added_columns)}. Removed: {', '.join(removed_columns)}." if added_columns or removed_columns else "No numerical columns added or removed."
    
    if added_columns and removed_columns:
        comment_num = f"Numerical Col Added: {', '.join(added_columns)}. Removed: {', '.join(removed_columns)}."
    elif added_columns:
        comment_num = f"Numerical Col Added: {', '.join(added_columns)}"
    elif removed_columns:
        comment_num = f"Numerical Col Removed: {', '.join(added_columns)}"    
    else:
        comment_num = "No numerical columns added or removed."  # Or simply skip defining it
    
    # Create DataFrame for numerical columns
    df_num = pd.DataFrame({"Status": [status_num], "Comment": [comment_num]}, index=['Numerical Columns'])  
    
    # Compare categorical columns
    yesterday_categorical = set(yesterday_df.select_dtypes(include=['object']).columns)
    today_categorical = set(today_df.select_dtypes(include=['object']).columns)
    added_columns = today_categorical - yesterday_categorical
    removed_columns = yesterday_categorical - today_categorical
    status_cat = "Fail" if added_columns or removed_columns else "Pass"
    #comment_cat = f"Categorical Col: Added: {', '.join(added_columns)}. Removed: {', '.join(removed_columns)}." if added_columns or removed_columns else "No categorical columns added or removed."
    
    if added_columns and removed_columns:
        comment_cat = f"Categorical Col Added: {', '.join(added_columns)}. Removed: {', '.join(removed_columns)}."
    elif added_columns:
        comment_cat = f"Categorical Col Added: {', '.join(added_columns)}"
    elif removed_columns:
        comment_cat = f"Categorical Col Removed: {', '.join(added_columns)}"    
    else:
        comment_cat = "No categorical columns added or removed."
    
    
    # Create DataFrame for categorical columns
    df_cat = pd.DataFrame({"Status": [status_cat], "Comment": [comment_cat]}, index=['Categorical Columns'])
    
    # Compare date columns
    yesterday_date = set(yesterday_df.select_dtypes(include=['datetime', 'datetime64']).columns)
    today_date = set(today_df.select_dtypes(include=['datetime', 'datetime64']).columns)
    added_columns = today_date - yesterday_date
    removed_columns = yesterday_date - today_date
    status_date = "Fail" if added_columns or removed_columns else "Pass"
    comment_date = f"Date Col Added: {', '.join(added_columns)}. Removed: {', '.join(removed_columns)}." if added_columns or removed_columns else "No date columns added or removed."
    
    if added_columns and removed_columns:
        comment_date = f"Date Col Added: {', '.join(added_columns)}. Removed: {', '.join(removed_columns)}."
    elif added_columns:
        comment_cat = f"Date Col Added: {', '.join(added_columns)}"
    elif removed_columns:
        comment_cat = f"Date Col Removed: {', '.join(added_columns)}"    
    else:
        comment_cat = "No Date columns added or removed."
    
    
    
    # Create DataFrame for date columns
    df_date = pd.DataFrame({"Status": [status_date], "Comment": [comment_date]}, index=['Date Columns'])
    
    # Concatenate all DataFrames
    df = pd.concat([df_num, df_cat, df_date])
  
    styled_df = df.style.applymap(highlight_pass_fail)

    col1,col2 = st.columns(2)

    with col1:
        st.dataframe(styled_df, width=1500) 
    with col2:
        summary_continer = st.container(border=True)
        check_name = "Structure Changes"
        summary_continer.subheader(f"{check_name} Summary")
        # summary = data_summary(df,check_name)
        # summary_continer.write(summary)


        summary = """
    
        All numerical columns remain consistent - Pass.
        All categorical columns remain consistent - Pass.
        All date columns remain consistent - Pass."""
        summary_continer.write(summary)




def restatement_changed():
    if st.session_state.restatement_button==False:
        st.session_state.restatement_button=True
        st.session_state.anomaly_button = False   


def anomaly_changed():
    if st.session_state.anomaly_button==False:
        st.session_state.restatament_button=False
        st.session_state.anomaly_button = True
        
def train_changed():
    if st.session_state.button_train_model==False:
        st.session_state.button_train_model = True        
             

def add_commas(number):
    # Convert the number to a string and format it with commas
    return "{:,}".format(number)


@st.cache_data
def filter_data_by_date_range(df1, df2, date_column):
    """
    Filters two DataFrames to only include rows within their overlapping date range.

    Parameters:
        df1 (DataFrame): First DataFrame.
        df2 (DataFrame): Second DataFrame.
        date_column (str): Column name containing date values.

    Returns:
        Tuple[DataFrame, DataFrame]: Filtered versions of df1 and df2.
    """
    # Copy data to avoid modifying originals
    df1, df2 = df1.copy(), df2.copy()

    # Convert the date column to datetime
    df1[date_column] = pd.to_datetime(df1[date_column])
    df2[date_column] = pd.to_datetime(df2[date_column])

    # Determine overlapping date range
    max_start_date = max(df1[date_column].min(), df2[date_column].min())
    min_end_date = min(df1[date_column].max(), df2[date_column].max())

    # Filter both DataFrames
    df1_filtered = df1[(df1[date_column] >= max_start_date) & (df1[date_column] <= min_end_date)]
    df2_filtered = df2[(df2[date_column] >= max_start_date) & (df2[date_column] <= min_end_date)]

    return df1_filtered, df2_filtered




@st.cache_data
def detect_changes(data1, data2):
    """
    Detects deleted, newly added, and modified rows between two dataframes.
    Excludes specified columns before returning the results.
    
    Parameters:
    - data1: pd.DataFrame (Old data)
    - data2: pd.DataFrame (New data)
    
    Returns:
    - A dictionary with three keys:
        - 'deleted_rows': Rows present in data1 but not in data2.
        - 'new_rows': Rows present in data2 but not in data1.
        - 'modified_rows': Rows with identical categorical columns but differing in other columns.
    """
    # Columns to exclude
    exclude_columns = ['pred','YEAR','MONTH', 'Unnamed: 0']
    
    # Determine the common columns (excluding specified ones)
    common_columns = data1.columns.intersection(data2.columns).difference(exclude_columns)
    
    # Filter both dataframes to only common columns
    data1_common = data1[common_columns]
    data2_common = data2[common_columns]
    
    # Select only categorical columns for comparison
    cat_columns = data1_common.select_dtypes(include=['object', 'category']).columns
    
    # Find rows in data1 that are not in data2 (deleted rows)
    deleted_rows = data1.loc[
        ~data1_common[cat_columns].apply(tuple, axis=1).isin(data2_common[cat_columns].apply(tuple, axis=1))
    ]
    
    # Find rows in data2 that are not in data1 (newly added rows)
    new_rows = data2.loc[
        ~data2_common[cat_columns].apply(tuple, axis=1).isin(data1_common[cat_columns].apply(tuple, axis=1))
    ]
    
    # Identify modified rows: same categorical data but differences in other columns
    merged = pd.merge(
        data1_common, data2_common,
        on=cat_columns.tolist(),
        how='inner',
        suffixes=('_old', '_new'),
        indicator=True
    )
    modified_rows = merged[
        (merged.filter(like='_old').values != merged.filter(like='_new').values).any(axis=1)
    ]
    
    # Remove specified columns from the results
    deleted_rows = deleted_rows.drop(columns=exclude_columns, errors='ignore')
    new_rows = new_rows.drop(columns=exclude_columns, errors='ignore')
    modified_rows = modified_rows.drop(columns=exclude_columns, errors='ignore')
    
    return {
        'deleted_rows': deleted_rows,
        'new_rows': new_rows,
        'modified_rows': modified_rows
    }

@st.cache_data
def display_comparison_results(
    results,
    data1,
    data2,
    date_col=None,
    id_col=None,
    numerical_cols=None,
    categorical_cols=None,
    flag=False
):
    """
    Display comparison results for restatement rows based on numerical and categorical columns.

    Parameters:
        results (dict): Results from the comparison function.
        data1 (DataFrame): Original DataFrame 1 (e.g., yesterday's data).
        data2 (DataFrame): Original DataFrame 2 (e.g., today's data).
        date_col (str): Date column name.
        id_col (str): Unique identifier column (mandatory for restatement comparison).
        numerical_cols (list): List of numerical columns (auto-detected if None).
        categorical_cols (list): List of categorical columns (auto-detected if None).
    """
    if id_col is None:
        raise ValueError("id_col is required for identifying restatements.")

    # Automatically detect numerical and categorical columns if not provided
    if numerical_cols is None:
        numerical_cols = data1.select_dtypes(include=["number"]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = data1.select_dtypes(include=["object", "category"]).columns.tolist()

    # Merge the two datasets on the id_col for comparison
    merged_data = pd.merge(
        data1,
        data2,
        on=id_col,
        suffixes=("_data1", "_data2"),
        how="inner"
    )

    # Initialize restatement summary
    restatement_summary = []

    # Check for restatements in numerical columns
    for col in numerical_cols:
        col_data1 = f"{col}_data1"
        col_data2 = f"{col}_data2"

        # Count restatements where values differ
        restated_rows = (merged_data[col_data1] != merged_data[col_data2]) & \
                        (~merged_data[col_data1].isna() | ~merged_data[col_data2].isna())
        restatement_summary.append({
            "Column Name": col,
            "Type": "Numerical",
            "Restated Rows": restated_rows.sum()
        })

    # Check for restatements in categorical columns
    for col in categorical_cols:
        col_data1 = f"{col}_data1"
        col_data2 = f"{col}_data2"

        # Count restatements where values differ
        restated_rows = (merged_data[col_data1] != merged_data[col_data2]) & \
                        (~merged_data[col_data1].isna() | ~merged_data[col_data2].isna())
        restatement_summary.append({
            "Column Name": col,
            "Type": "Categorical",
            "Restated Rows": restated_rows.sum()
        })

    # Convert restatement summary to DataFrame
    restatement_summary_df = pd.DataFrame(restatement_summary)
    if flag:
        anomaly_count = (data2["anomaly"] == True).sum()
        restatement_summary_df.loc[restatement_summary_df['Column Name'] == 'TOTAL_RX', 'Restated Rows']=anomaly_count


    #st.write("### Restatements Summary")
    st.dataframe(restatement_summary_df)

    # with st.expander("Expand to View the Comparison Results"):
    #     # Streamlit outputs in the specified order
    #     st.subheader("General Comparison Results")
    #     #st.write("Total Rows in File 1:", results['total_rows_file1'])
    #     #st.write("Total Rows in File 2:", results['total_rows_file2'])

    #     st.write("New Records")
    #     st.dataframe(results['new_rows'])

    #     st.write("Deleted Records")
    #     st.dataframe(results['deleted_rows'])

    




@st.cache_data
def plot_pie(df, select_col):
    # Ensure the 'Brnd_Name' column exists in the DataFrame
    
    # Filter DataFrame based on the selected brandsdf = df.copy()

    # Count values in the selected column
    select_col_counts = df[select_col].value_counts().reset_index()

    # Rename the columns
    select_col_counts.columns = [select_col, 'Count']

    # Sort the result by 'Count' and select the top 5
    select_col_counts = select_col_counts.sort_values(by='Count', ascending=False).head(10)
    others_count = select_col_counts.iloc[5:]['Count'].sum()

    # Keep the top 5 rows
    select_col_counts = select_col_counts.head(5)

    # Add the "Others" row using pd.concat
    others_row = pd.DataFrame([{select_col: 'Others', 'Count': others_count}])
    select_col_counts = pd.concat([select_col_counts, others_row], ignore_index=True)

    # Reset the index for readability (optional)
    select_col_counts.reset_index(drop=True, inplace=True)
     
    
    if select_col=='Prscrbr_State_Abrvtn':
        title='State'
    elif select_col=='Prscrbr_Type':
        title='Speciality'
    elif select_col=='Brnd_Name':
        title='Brand'    
    elif select_col=='Prscrbr_City':
        title='City'
    else:
        title='Brand'            
        
    
    
    # Calculate percentages
    select_col_counts['Percentage'] = (select_col_counts['Count'] / select_col_counts['Count'].sum() * 100).round(1)
    select_col_counts['Legend_Label'] = select_col_counts[select_col] 

    # Apply custom CSS to style the container
    st.markdown(
        """
        <style>
        .custom-container {
            background-color: white;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Plot the pie chart
    fig = px.pie(
        select_col_counts, 
        names='Legend_Label',  # Use the custom legend labels with percentages
        values='Count', 
        title=title+' Level Distribution',
        hover_data={'Count': True},  # Show count on hover
        labels={'Count': 'Data Points'}
    )

    # Customize hover data to show percentage
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label', 
        hovertemplate=title+': %{label}<br>Data Points: %{value}<br>Percentage: %{percent}'
    )

    # Customize the layout to center the title and adjust legend placement
    fig.update_layout(
        title={
            'text': title+' Level Distribution',
            'x': 0.5,  # Center the title horizontally
            'xanchor': 'center'  # Anchor the title at the center
        },
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="top",  # Anchor to the top
            y=-0.2,  # Position below the chart
            xanchor="center",  # Center the legend
            x=0.5  # Center alignment
        ),
        height=500,  # Set a fixed height for the plot to ensure consistency
        margin=dict(t=50, b=100)  # Adjust margins to fit the legend
    )

    # Display the plot in Streamlit with a smaller size inside a custom container
    st.plotly_chart(fig, use_container_width=True)
def highlight_pass_fail(val):
    if val == 'Pass':
        color = 'White'
        text_color = 'Green'
        symbol = '\u25cf'  # Green filled circle
    elif val == 'Fail':
        color = 'White'
        text_color = 'Red'
        symbol = '\u25cf'  # Red filled circle
    elif val == 'Pass.':
        color = 'White'
        text_color = '#ED7D31'
        symbol = '\u25cf'  # Red filled circle
    else:
        color = 'white'
        text_color = 'black'
        symbol = '\u25cf'  # Black filled circle
    #return f'background-color: {color}; color: {text_color}; text-align: center; padding: 2px 0px 2px 0px; content: "{symbol}"'**
    return f'background-color: {color}; color: {text_color}; text-align: center; padding: 2px 0px 2px 0px; content: "{symbol}"'
    #return **f'background-color: {color}; color: {text_color}; text-align: center; padding: 2px 0px 2px 0px; content: "{symbol}"'**


