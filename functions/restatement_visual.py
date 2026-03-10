import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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




import pandas as pd
import plotly.graph_objects as go
import streamlit as st

#from llm_agents.Data_summary import data_summary



import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def plot_anomaly_category_distribution(today_df_final, dimensionCol):
    d = {}
    
    for col in dimensionCol:
        allRestatmentData = today_df_final[(today_df_final['anomaly'] == True)].reset_index(drop=True)
        filterallRestatmentData = allRestatmentData.groupby([col])[col].count()
        filterallRestatmentData = pd.DataFrame(filterallRestatmentData)
        filterallRestatmentData.rename(columns={col: 'count'}, inplace=True)
        filterallRestatmentData = filterallRestatmentData.reset_index()
        filterallRestatmentData.sort_values(by=['count'], ascending=False, inplace=True)
        filterallRestatmentData.reset_index(drop=True, inplace=True)
        filterallRestatmentData['percentage'] = round((filterallRestatmentData['count'] / filterallRestatmentData['count'].sum()) * 100, 2)
        
        if filterallRestatmentData[col][0] not in d:
            d[col + ' ' + '=' + ' ' + filterallRestatmentData[col][0]] = filterallRestatmentData['percentage'][0]
    
    # Convert dictionary to DataFrame
    keys = list(d.keys())
    values = list(d.values())
    df_keys_values = pd.DataFrame({'Key': keys, 'Value': values})
    # Count anomalies
    anomaly_count = len(today_df_final[today_df_final['anomaly'] == True])

    # Convert anomaly count to a DataFrame
    anomaly_df = pd.DataFrame({'Check': ['Anomaly Count'], 'Value': [anomaly_count]})

    # Combine both DataFrames
    combined_df = pd.concat([df_keys_values, anomaly_df], ignore_index=True)

 

    # Create a Plotly horizontal bar chart
    fig = go.Figure()

    # Add bars to the figure
    fig.add_trace(go.Bar(
        x=values,
        y=keys,
        orientation='h',
        marker=dict(color='skyblue')
    ))
    
    # Set the layout for the Plotly figure
    fig.update_layout(
        title={'text': "Percentage Distribution of Restatements by Category \n ", 'x': 0.5, 'xanchor': 'center',  'yanchor': 'top'},
        xaxis=dict(title="Percentage (%)", range=[0, 100]),  # Limit x-axis to 100%
        height=400,  # Adjust the figure height as needed
        width=400,  # Decrease the width of the graph
        margin=dict(t=100)  # Adjust top margin to 18px, bottom and left margins
    )

    # Add the percentage labels on each bar
    for i, value in enumerate(values):
        fig.add_annotation(
            x=value + 1,  # Position slightly right of the bar
            y=keys[i],
            text=f"{value}%",
            showarrow=False,
            font=dict(size=15),
            xanchor='left',
            yanchor='middle'
        )

    # Display the figure in Streamlit
    
    col1,col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        summary_continer = st.container(border=True)
        check_name = "Restatement Check"
        summary_continer.subheader(f"{check_name} Summary")
        # summary = data_summary(combined_df,check_name)
        summary = """
    
        Total Records: 15,227.
        Restatement Records: 1,840.
        Non-Restatement Records: 13,387.
        Trend Analysis: Time series comparison shows non-overlapping trends between the current and previous files, indicating potential restatement activity during the latest weeks.
        Root Causes Identified:Majority restatements are concentrated in records related to SPECIALTY = NP (69.13%)Followed by INDICATION = Indication6 (28.1%) and PRODUCT = Brand11 (7.12%).
        Comments: Restatements are primarily associated with specific specialties and indications, suggesting targeted investigation may be needed.
        The data reveals a moderate level of restatement, with clear categorical drivers. Ongoing monitoring and RCA are recommended for categories with high contribution to restatement.
                """
        summary_continer.write(summary)


@st.cache_data
def Brand_Trend_Graph(df1, df2, selected_brand, month_select, Num_Dim):
    # Create copies of DataFrames to avoid modifying the originals
    df1 = df1.copy()
    df2 = df2.copy()

    min=df2['WEEK'].min()
    max=df1['WEEK'].max()
    
    df1=df1[(df1['WEEK']>=min) & (df1['WEEK']<=max)]
    df2=df2[(df2['WEEK']>=min) & (df2['WEEK']<=max)]
    
    df1=df1[df1['WEEK'].dt.month>=4]
    df2=df2[df2['WEEK'].dt.month>=4]
    
    # Create a mapping of month names to numbers
    month_name_to_num = {month: index for index, month in enumerate(calendar.month_name) if month}

    # Check if month_select contains numbers or names
    if month_select:
        # Ensure month_select contains month names, not numbers
        if isinstance(month_select[0], str):
            # Convert the selected month names to their corresponding numbers
            month_numbers = [month_name_to_num[month] for month in month_select]
        else:
            # Assume month_select already contains month numbers
            month_numbers = month_select
        
        # Filter both DataFrames based on the selected month numbers
        df1 = df1[df1['WEEK'].dt.month.isin(month_numbers)]
        df2 = df2[df2['WEEK'].dt.month.isin(month_numbers)]
        
    # Ensure selected_brand is a list
    if isinstance(selected_brand, str):
        selected_brand = [selected_brand]
    
    # Initialize a Plotly figure
    fig = go.Figure()
    
    # Loop through each brand to plot its data
    for brand in selected_brand:
        # Filter df1 for the current brand
        brand_data1 = df1[df1["PRODUCT"] == brand]
        # Aggregate data by date for daily totals
        daily_aggregate1 = brand_data1.groupby('WEEK').agg({Num_Dim: 'sum'}).reset_index()

        # Add a trace for df1 (could be current data)
        fig.add_trace(go.Scatter(x=daily_aggregate1['WEEK'], y=daily_aggregate1[Num_Dim], 
                                 mode='lines', name=f'{brand} (Last File Month)'))

        # Filter df2 for the current brand
        brand_data2 = df2[df2["PRODUCT"] == brand]
        daily_aggregate2 = brand_data2.groupby('WEEK').agg({Num_Dim: 'sum'}).reset_index()

        # Add a trace for df2 (could be historical data)
        fig.add_trace(go.Scatter(x=daily_aggregate2['WEEK'], y=daily_aggregate2[Num_Dim], 
                                 mode='lines', name=f'{brand} (Current File Month)', line=dict(dash='dash')))

    # Update layout to improve readability
    fig.update_layout(title={'text': "Brand Trend Over Time", 'x': 0.5, 'xanchor': 'center'}, xaxis_title="Date", yaxis_title=Num_Dim,
                      legend_title="Brand", legend=dict(yanchor="top", y=-0.3, xanchor="center", x=0.5))
    
    
    st.plotly_chart(fig, use_container_width=True)
    return fig


@st.cache_data
def Anomalies_Brand(testingData, filter_Col):
    # Filter and count anomalous records
    anomalousData = testingData[testingData['pred'] == 1].groupby([filter_Col])['pred'].count().reset_index()
    anomalousData.rename(columns={'pred': 'Total_Anomalous'}, inplace=True)

    # Sort by the total number of anomalies and get the top 10
    anomalousData = anomalousData.sort_values(by='Total_Anomalous', ascending=False).head(5)
    anomalousData = anomalousData.sort_values(by='Total_Anomalous', ascending=True)

    # Convert columns to integers
    anomalousData['Total_Anomalous'] = anomalousData['Total_Anomalous'].astype(int)

    # Determine text position based on Total_Anomalous value
    text_position = ['inside' if value >= 20 else 'outside' for value in anomalousData['Total_Anomalous']]  # Adjust threshold as needed

    # Create a Plotly bar chart
    fig = go.Figure()

    # Add bars to the figure
    fig.add_trace(go.Bar(
        y=anomalousData[filter_Col],
        x=anomalousData['Total_Anomalous'],
        orientation='h',
        marker=dict(color='skyblue'),
        text=anomalousData['Total_Anomalous'],
        textposition=text_position,  # Set text position based on visibility
        textfont=dict(size=15)  # Increase data label size
    ))

    # Update layout for better appearance
    fig.update_layout(
        title='Top 5 Restatement Records by {}'.format(filter_Col),
        xaxis_title='Number of Records',
        yaxis_title=filter_Col,
        yaxis=dict(tickfont=dict(size=15)),  # Reduce y-axis label size
        xaxis=dict(tickfont=dict(size=15)),
        showlegend=False,
        margin=dict(l=150, r=150, t=50, b=50),
        height=350  # Set height to 350 pixels
    )
    st.markdown(
        """
        <style>
        .graph-container {
            margin-top: 9px;  /* Adjust the margin as needed to shift the graph down */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        st.markdown('<div class="graph-container"></div>', unsafe_allow_html=True)  # Apply the custom styling
        st.plotly_chart(fig, use_container_width=True)

@st.cache_data
# def BarChartNumberAnomaliesDetected(testingData):
#     fig, ax = plt.subplots(figsize=(6, 1))  # Increased height to 3

#     goodRecords = len(testingData[testingData['pred'] == 0])
#     anomalousRecords = len(testingData[testingData['pred'] == 1])

#     # Bar plots
#     ax.barh(np.arange(1), goodRecords, color='skyblue', label='Non-Anomalous Records')
#     ax.barh(np.arange(1), anomalousRecords, color='#0063C8', label='Anomalous Records', left=goodRecords)

#     ax.set_yticks(np.arange(1))
#     ax.set_yticklabels([''])

#     #ax.set_title('Non-Anomalous Data vs Anomalous Data', pad=17,fontsize=10.2)  # Added pad to increase distance from the plot
#     plt.xticks([])

#     # Annotate bars with values
#     ax.text(goodRecords / 2, 0, str(goodRecords), va='center', ha='center', color='black', fontsize=15)
#     ax.text(goodRecords + anomalousRecords / 2, 0, str(anomalousRecords), va='center', ha='center', color='black',
#             fontsize=15)

#     plt.tight_layout()

#     # Move legend
#     #plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), bbox_transform=ax.transAxes, fontsize=8)
#     return fig

def BarChartNumberAnomaliesDetected(testingData):
    fig, ax = plt.subplots(figsize=(6, 1))  # Increased height to 1

    goodRecords = len(testingData[testingData['pred'] == 0])
    anomalousRecords = len(testingData[testingData['pred'] == 1])

    # Bar plots
    ax.barh(0, goodRecords, color='skyblue', label='Non-Anomalous Records')
    ax.barh(0, anomalousRecords, color='#0063C8', label='Anomalous Records', left=goodRecords)

    # Set y-ticks (single bar, so one y-tick at index 0)
    ax.set_yticks([0])
    ax.set_yticklabels([''])

    # Remove x-ticks
    ax.set_xticks([])

    # Annotate bars with values
    ax.text(goodRecords / 2, 0, str(goodRecords), va='center', ha='center', color='black', fontsize=15)
    ax.text(goodRecords + anomalousRecords / 2, 0, str(anomalousRecords), va='center', ha='center', color='black',
            fontsize=15)

    # Tighten layout and remove extra space
    ax.margins(y=0)  # Remove vertical spacing
    ax.set_xlim(0, goodRecords + anomalousRecords)  # Remove extra horizontal space
    plt.tight_layout()
    
    return fig



def fig_to_base64(_fig):
    buf = io.BytesIO()
    _fig.savefig(buf, format="png", bbox_inches='tight')  # Ensure entire figure is saved
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return img_base64





