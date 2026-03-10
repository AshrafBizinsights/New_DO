import pandas as pd
import numpy as np
import streamlit as st
from datetime import timedelta


def date_selector(EXTRACTED_DATA_FROM_DB):
    # Ensure 'selected_day' exists in session state for tracking
    if "selected_day" not in st.session_state:
        st.session_state.selected_day = ""
    
    # Determine the maximum date in the data and convert it to date format
    max_today_df = pd.to_datetime(EXTRACTED_DATA_FROM_DB["ds"].max())
    
    # Display two columns for date input and week selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<p style='text-align: center; font-weight: bold;'>Choose a Date</p>", unsafe_allow_html=True)
        selected_date = st.date_input(
            label="",
            value=max_today_df,
            format="MM/DD/YYYY",
            label_visibility="collapsed"  # Hide the label
        )
    
    # Set up the fixed dates (1st, 8th, 15th, 22nd, and 29th) for the selected month
    month_start = selected_date.replace(day=1)
    selected_month_internal = [(month_start + timedelta(days=day - 1)).strftime("%m/%d/%Y") for day in [1, 8, 15, 22, 29]]
    
    with col2:
        st.markdown("<p style='text-align: center; font-weight: bold;'>Select a Date in the Month</p>", unsafe_allow_html=True)
        
        # Display five date buttons for selection
        cols = st.columns(5, gap="small")
        for i, internal_day in enumerate(selected_month_internal):
            with cols[i]:
                # Use internal_day as the label for each button
                if st.button(internal_day, key=f"day_button_{i}"):
                    st.session_state.selected_day = internal_day

    # Return the selected date from session state or the date picker
    data_freshness_date = st.session_state.selected_day or selected_date.strftime("%m/%d/%Y")
    return data_freshness_date

def render_metrics_dashboard(st, Data_freshness, Data_Validation, Brand_Volume, Unknown_Unknown, data_triangulation ,testingdata, anomalies_category, results, render_metric, add_commas, df):
    total_freshness = len(Data_freshness)
    #fail_freshness = len(Data_freshness[Data_freshness["Status"] == "Fail"])
    fail_freshness = 0
    freshness_status = "Good" if fail_freshness == 0 else "Issues Identified"
    
    total_validation = len(Data_Validation)
    fail_validation = len(Data_Validation[Data_Validation["Status"] == "Fail"])

    validation_status = (
        "Warnings" if fail_validation == 0 and not Data_Validation["Comments"].isna().all()
        else ("Issues Identified" if fail_validation > 0 else "Good")
    )

    total_Brand_volume = len(Brand_Volume)
    fail_Brand_volume = len(Brand_Volume[Brand_Volume["Status"] == "Fail"])
    brand_volume_status = "Good" if fail_Brand_volume == 0 else "Issues Identified"
    
    total_Unknown = len(Unknown_Unknown)
    fail_Unknown = len(Unknown_Unknown[Unknown_Unknown["Status"] == "Fail"])
    unknown_status = "Good" if fail_Unknown == 0 else "Issues Identified"

    total_Unknown = len(Unknown_Unknown)
    fail_Unknown = len(Unknown_Unknown[Unknown_Unknown["Status"] == "Fail"])
    unknown_status = "Good" if fail_Unknown == 0 else "Issues Identified"
    
    total_variance = len(data_triangulation)
    fail_variance = len(data_triangulation[data_triangulation["Status"] == "Fail"])
    variance_status = "Good" if fail_Unknown == 0 else "Issues Identified"

    restatement_status = "Good" if len(testingdata[(testingdata['pred'] == 0)]) == 0 else "Issues Identified"
    
    col1, col2 = st.columns(2)
    st.write(" ")
    with col1:
        failing_rows = df[df["Status"] != "Pass"]
        total_rows = len(df)
        not_pass = len(failing_rows)
        correctness_ratio = f"{not_pass}/{total_rows}"
        
        if not_pass > 0:
            issue_reason = "<br>".join(failing_rows["Comment"].dropna().tolist())
        else:
            issue_reason = "Shows the structural changes in data."
        
        render_metric(col1, "Structural Changes", correctness_ratio, "Issue" if not_pass > 0 else "Good", f"{issue_reason}")

    with col2:
        issue_reason = "Indicates the number of anomaly records related to brand volume."
        if brand_volume_status == "Issues Identified":
            if "Granularity" in Brand_Volume.columns:
                failed_records = Brand_Volume[Brand_Volume["Status"] == "Fail"]
                issue_reason = "<br>".join(failed_records["Granularity"].dropna().tolist())
            else:
                issue_reason = "Error: 'Granularity' column is missing from Brand_Volume data."
        
        correctness_ratio = f"{fail_Brand_volume}/{total_Brand_volume}"
        render_metric(col2, "Brand Volume", correctness_ratio, brand_volume_status, f"Anomaly found in: {issue_reason}")

    col3, col4 = st.columns(2)
    st.write(" ")
    with col3:
        if freshness_status == "Good":
            issue_reason = "Validates data freshness against expected columns."
        elif freshness_status == "Issues Identified":
            failed_records = Data_freshness[Data_freshness["Status"] == "Fail"]
            if "Dimension" in failed_records.columns and "Value" in failed_records.columns:
                issue_reason = "<br>".join(
                    f"{dim}: {val}" for dim, val in zip(failed_records["Dimension"], failed_records["Value"])
                )
            else:
                issue_reason = "Error: 'Dimension' or 'Value' column is missing from Data_freshness."
        
        correctness_ratio = f"{fail_freshness}/{total_freshness}"
        render_metric(col3, "Data Freshness", correctness_ratio, freshness_status, f"{issue_reason}")

    with col4:
        total_rows_file1 = add_commas(testingdata[(testingdata['pred'] == 1)].shape[0])
        render_metric(col4, "Restatement Detected", total_rows_file1, restatement_status, "Indicates restatements were deducted in the number of rows in the data.")

    col5, col6 = st.columns(2)
    st.write(" ")
    with col5:
        if validation_status == "Good":
            issue_reason = "Validates data against expected columns."
        elif validation_status == "Warnings":
            issue_reason = "All columns have passed the checks, but with warnings based on historical trend analysis."
        elif validation_status == "Issues Identified":
            failed_records = Data_Validation[Data_Validation["Status"] == "Fail"]
            if "Dimension" in failed_records.columns:
                issue_reason = "<br>".join(failed_records["Dimension"].dropna().tolist())
            else:
                issue_reason = "Error: 'Dimension' column is missing from Data_Validation."
        
        correctness_ratio = f"{fail_validation}/{total_validation}"
        render_metric(col5, "Data Validation", correctness_ratio, validation_status, f"{issue_reason}")

    with col6:
        issue_reason = "Indicates the number of anomaly contribution records."
        if unknown_status == "Issues Identified":
            if "Granularity" in Unknown_Unknown.columns:
                failed_records = Unknown_Unknown[Unknown_Unknown["Status"] == "Fail"]
                issue_reason = "<br>".join(failed_records["Granularity"].dropna().tolist())
            else:
                issue_reason = "Error: 'Granularity' column is missing from Unknown_Unknown data."
        
        correctness_ratio = f"{fail_Unknown}/{total_Unknown}"
        render_metric(col6, "Unknown Unknowns", correctness_ratio, unknown_status, f"Unknown unknowns found in: {issue_reason}")

    col7, col8 = st.columns(2)
    with col7:
        issue_reason = "TRx Metric on October 27, 2023."
        variance_status = "Issues Identified"
        correctness_ratio = f"{fail_variance}/{total_variance}"
        render_metric(col7, "Data Triangulation", correctness_ratio, variance_status, f"Data Triangulation found in: {issue_reason}")



def render_metric(column, title, correctness_ratio, status, tooltip_text):
    # Define the color and icon based on the status
    status_colors = {
        "Good": "#4CAF50",    # Green
        "Warnings": "#FFC107", # Yellow
        "Issues Identified": "#F44336"     # Red
    }
    color = status_colors.get(status, "#333")  # Default to a dark color if status is unknown

    # Set icons based on the status
    icons = {
        "Good": '<span style="color: white;">✔</span>',  # White checkmark for good status
        "Warnings": '<span style="color: white;">❕</span>',  # White exclamation mark for warnings
        "Issues Identified": '<span style="color: white;">❕</span>'  # White exclamation mark for issues
    }
    icon = icons.get(status, "❓")

    # Tooltip icon with hover text, added CSS directly to kpi_html for reliable styling in Streamlit
    kpi_html = f"""
    <style>
        .tooltip {{
            position: relative;
            display: inline-block;
            cursor: pointer;
        }}
        .tooltip .tooltiptext {{
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: left;
            border-radius: 10px;
            padding: 10px;
            position: absolute;
            z-index: 2;
            bottom: 120%;  /* Adjusted for better placement */
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.8em;
        }}
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}
    </style>
    
    <div style="
        border: 1px solid #d9d9d9; 
        border-radius: 10px; 
        padding: 15px; 
        background-color: #f9f9f9;
        display: flex;
        align-items: center;
        justify-content: space-between;
    ">
        <div style="display: flex; align-items: stretch;">
            <div style="font-size: 24px; background-color: {color}; color: white; width: 50px; padding: 20px 0; border-radius: 10px 0 0 10px; margin-right: 10px; height: 100%; display: flex; align-items: center; justify-content: center;">
                {icon}
            </div>
            <div>
                <div style="display: flex; align-items: center;">
                    <div style="font-weight: bold; font-size: 25px; color: #333;">{title}</div>
                    <div class="tooltip" style="margin-left: 25px;">
                        <span style="font-size: 22px; color: #777; font-weight: bold;">
                            <div style="width: 24px; height: 24px; background-color: lightblue; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 18px; color: white;">
                                i
                            </div>
                        </span>
                        <div class="tooltiptext">{tooltip_text}</div>
                    </div>
                </div>
                <div style="border-bottom: 1px solid #d9d9d9; margin-top: 5px; width: 100%;"></div>  <!-- Thin line below title -->
                <div style="display: flex; align-items: center; margin-top: 5px;">
                    <div style="font-weight: bold; font-size: 18px; color: {color}; margin-right: 18px;">{correctness_ratio}</div>
                    <div style="font-size: 18px; color: #777;">{status}</div>  <!-- Status next to correctness ratio -->
                </div>
            </div>
        </div>
    </div>
    """

    # Render the HTML content in Streamlit
    column.markdown(kpi_html, unsafe_allow_html=True)
