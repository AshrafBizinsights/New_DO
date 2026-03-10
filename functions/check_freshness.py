import pandas as pd

def check_freshness(freshness_check_df, date, columns_for_freshness, freshnessDf):

    # Iterate over each column in the list of columns for freshness check
    for columns in columns_for_freshness:
        # Check if there are no values for the specified date in the current column
        if (
            freshness_check_df.loc[freshness_check_df["ds"] == date, columns].shape[
                0
            ]
            == 0
        ):
            # Create a new row indicating failure due to lack of values for the date in the column
            new_row = [
                columns,
                "",
                "Fail",
                "No Value in column is having Latest Data",
            ]
            new_row_df = pd.DataFrame([new_row], columns=freshnessDf.columns)
            freshnessDf = pd.concat([freshnessDf, new_row_df])
        else:
            # Get unique values in the current column
            uniqueValues = freshness_check_df[columns].unique()
            # Iterate over each unique value in the column
            for val in uniqueValues:
                # Check if there is no data for the specified date and value in the column
                if (
                    freshness_check_df.loc[
                        (freshness_check_df["ds"] == date)
                        & (freshness_check_df[columns] == val)
                    ].shape[0]
                    == 0
                ):
                    # Create a new row indicating failure due to lack of data for the date and value
                    new_row = [
                        columns,
                        val,
                        "Fail",
                        f"No data for {date} for {val} in column {columns}",
                    ]
                    new_row_df = pd.DataFrame(
                        [new_row], columns=freshnessDf.columns
                    )
                    freshnessDf = pd.concat([freshnessDf, new_row_df])
                else:
                    # Create a new row indicating successful freshness check
                    new_row = [columns, val, "Pass", ""]
                    new_row_df = pd.DataFrame(
                        [new_row], columns=freshnessDf.columns
                    )
                    freshnessDf = pd.concat([freshnessDf, new_row_df])

    return freshnessDf