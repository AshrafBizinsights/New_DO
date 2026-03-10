import logging
import streamlit as st
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO


# Define a function called check_uniqueness_new which takes several parameters
def check_uniqueness_new(Unique_check_df, date, columns_to_check_uniqueness, uniqueDf):
    # Log an INFO message indicating that the function has been called
    logging.info(f"Checking Uniqueness for columns {columns_to_check_uniqueness}")
    try:
        # Store a baseline DataFrame for comparison
        baseline = Unique_check_df

        # Filter the DataFrame for the specified date
        date_df = Unique_check_df.loc[Unique_check_df.ds == date]


        # Iterate over each column in the list of columns to check for uniqueness
        for columns in columns_to_check_uniqueness:
            # Check if the column is unique in both the baseline and the filtered DataFrame
            baseline_is_unique = baseline[columns].is_unique
            date_df_is_unique = date_df[columns].is_unique

         

            # Check if the uniqueness status has remained the same
            if baseline_is_unique == date_df_is_unique == True:
                # Log an INFO message indicating that the column is still unique
                logging.info(f"{columns} column is still unique")

                # Create a new row in the DataFrame to track uniqueness status
                new_row = [columns, "Pass", ""]
                new_row_df = pd.DataFrame([new_row], columns=uniqueDf.columns)
                uniqueDf = pd.concat([uniqueDf, new_row_df])
            elif baseline_is_unique != date_df_is_unique:
                # Log a WARNING message if the uniqueness status has changed
                logging.warning(
                    f"{columns}'s unique value was {baseline_is_unique}, but is now {date_df_is_unique}"
                )

                # Create a new row in the DataFrame to track uniqueness status
                new_row = [columns, "Fail", "Not Proportionate with historical trend"]
                new_row_df = pd.DataFrame([new_row], columns=uniqueDf.columns)
                uniqueDf = pd.concat([uniqueDf, new_row_df])
            else:
                # Log an INFO message if the column's values are not unique but the proportion remains the same
                logging.info(
                    f"{columns}'s values are not unique and it is in proportionate with Historical not unique trend"
                )

                # Create a new row in the DataFrame to track uniqueness status
                new_row = [
                    columns,
                    "Pass.",
                    "Proportionate with historical Unique trend",
                ]
                new_row_df = pd.DataFrame([new_row], columns=uniqueDf.columns)
                uniqueDf = pd.concat([uniqueDf, new_row_df])
        # Return the DataFrame containing uniqueness status for each column
        return uniqueDf
    except Exception as e:
        # Log an ERROR message if an exception occurs
        logging.error(f"Error occurred in check_uniqueness_new function: {e}")
        # Return None if an error occurs
        return None
