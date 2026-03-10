import logging
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO


# Define a function called check_missing_values_new which takes several parameters
def check_missing_values_new(
    Missing_check_df, columns_for_not_nulls, baseline, baseline_obs, notnullDf
):
    # Log an INFO message indicating that the function has been called
    logging.info("check_missing_values_new function called.")
    try:
        # Call a function to check for null changes and get relevant dataframes
        null_count_df, col_dict, notnullDf = check_null_changes_new(
            Missing_check_df, baseline, baseline_obs, columns_for_not_nulls, notnullDf
        )

        return notnullDf
    except Exception as e:
        # Log a CRITICAL message if an error occurs in the try block
        logging.critical(f"Error occurred in check_missing_values_new function: {e}")
        # Return None if an error occurs
        return None
    

# Define a function called check_null_changes_new which takes several parameters
def check_null_changes_new(
    Null_check_df, baseline, baseline_obs, columns_for_not_nulls, notnullDf
):
    # Log an INFO message indicating that the function has been called
    logging.info("check_null_changes_new function called.")
    try:
        # Calculate the null count for each column in the DataFrame
        null_count_df = Null_check_df.isnull().sum().reset_index()
        null_count_df.columns = ["Column", "Null Count"]

        # Get the total number of observations in the DataFrame
        df_obs = Null_check_df.shape[0]



        # Initialize a dictionary to store column-wise null change information
        col_dict = {}

        # Log an INFO message indicating the initiation of null checks for specified columns
        bold_text = f"**Initiating Null checks for columns: {columns_for_not_nulls}**"
        logging.info(bold_text)

        # Iterate over each column in the list of columns for null checks
        for columns in columns_for_not_nulls:
            # Check if the column had no nulls in the baseline
            if baseline[baseline["Column"] == columns]["Null Count"].values[0] == 0:
                # Check if nulls are detected in the current DataFrame where there were none before
                if (
                    null_count_df[null_count_df["Column"] == columns][
                        "Null Count"
                    ].values[0]
                    != 0
                ):
                    # Log an INFO message indicating the detection of new null values in the column
                    logging.info(f"{columns} column has nulls where it didn't before.")

                    # Store information about the column's null status in the dictionary
                    col_dict[columns] = True

                    # Create a new row in the DataFrame to track not-null status
                    new_row = [columns, "Fail", "New NULL Values detected"]
                    new_row_df = pd.DataFrame([new_row], columns=notnullDf.columns)
                    notnullDf = pd.concat([notnullDf, new_row_df])
                else:
                    # Log an INFO message indicating that the column doesn't have nulls
                    logging.info(f"{columns} column doesn't have nulls.")

                    # Store information about the column's null status in the dictionary
                    col_dict[columns] = False

                    # Create a new row in the DataFrame to track not-null status
                    new_row = [columns, "Pass", ""]
                    new_row_df = pd.DataFrame([new_row], columns=notnullDf.columns)
                    notnullDf = pd.concat([notnullDf, new_row_df])
            else:
                # Calculate null count and baseline count for the current column
                null_count = null_count_df[null_count_df["Column"] == columns][
                    "Null Count"
                ].values[0]
                baseline_count = baseline[baseline["Column"] == columns][
                    "Null Count"
                ].values[0]

                # Perform a hypothesis test to determine if null proportions have significantly changed
                count = [baseline_count, null_count]
            
                nobs = [baseline_obs, df_obs]
                
                pval = proportions_ztest(count, nobs)[1]
            
                # Check if the p-value is less than or equal to 0.05
                if pval <= 0.05:
                    # Log an INFO message indicating significant changes in null proportions
                    logging.info(
                        f"{columns} column has nulls present, and they are significantly different than historical null proportion."
                    )

                    # Store information about the column's null status in the dictionary
                    col_dict[columns] = True

                    # Create a new row in the DataFrame to track not-null status
                    new_row = [
                        columns,
                        "Fail",
                        "Significantly different from historical trend",
                    ]
                    new_row_df = pd.DataFrame([new_row], columns=notnullDf.columns)
                    notnullDf = pd.concat([notnullDf, new_row_df])
                else:
                    # Log an INFO message indicating insignificant changes in null proportions
                    logging.info(
                        f"{columns} column has nulls present, but they are not significantly different than historical null proportion."
                    )

                    # Store information about the column's null status in the dictionary
                    new_row = [
                        columns,
                        "Pass.",
                        "Null detected. Not significantly different from historical trend.",
                    ]
                    new_row_df = pd.DataFrame([new_row], columns=notnullDf.columns)
                    notnullDf = pd.concat([notnullDf, new_row_df])
                    col_dict[columns] = False
  
        # Return the DataFrame containing null counts, the dictionary containing column-wise null change information, and the DataFrame tracking not-null status
        return null_count_df, col_dict, notnullDf
    except Exception as e:
        # Log an ERROR message if an exception occurs
        logging.error(f"Error occurred in check_null_changes_new function: {e}")
        # Return None if an error occurs
        return None



def check_nulls_new(
    Null_check_df,
    date,
    column,
    columns_for_not_nulls,
    baseline,
    baseline_obs,
    notnullDf,
):
    fig = check_missing_values_new(
        Null_check_df.loc[Null_check_df.ds == date],
        column,
        columns_for_not_nulls,
        baseline,
        baseline_obs,
        notnullDf,
    )



