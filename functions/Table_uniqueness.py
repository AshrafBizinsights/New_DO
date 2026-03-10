import pandas as pd
from functions.restatement import highlight_pass_fail

def is_compound_key(df, columns_list,date,container_data_uniqueness):
    # st.write(columns_list)
    # st.write(type(columns_list))
    df = df.loc[df.ds == date]
    df.to_csv('uniqueTable.csv')
    is_unique = df.duplicated(columns_list).sum() == 0
    if is_unique:
        #container_data_uniqueness.write(f"The combination of columns forms a unique key for the latest week.\n {columns_list}")
        container_data_uniqueness.write("The combination of columns does not form a unique key for the latest week.")
        duplicated_records = df[df.duplicated(columns_list, keep=False)]
        duplicated_records.to_csv('duplicated_records.csv')

        

    else:
        listColumns = ', '.join(columns_list)
        data = {'Dimension': listColumns,'Result': 'Pass','Comment': 'No Duplicates found'}
        df = pd.DataFrame(data, index=[0])
        styled_df = df.style.applymap(highlight_pass_fail)
        container_data_uniqueness.dataframe(styled_df, width=1500, hide_index=True)

        #container_data_uniqueness.write(duplicated_records.head(10))


