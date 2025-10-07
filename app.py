# app.py (Example using Streamlit)
# This is a basic structure, you would need to install streamlit (pip install streamlit)
# and save this code as app.py, then run 'streamlit run app.py' in your terminal

import streamlit as st
from data_processing import load_and_clean_data, select_features
from model_training import train_estimation_models, train_main_model
from prediction_logic import (
    calculate_inbound_jacks, calculate_nwpet_jacks, calculate_drp_jacks,
    calculate_manual_labor, calculate_nwpet_labor, predict_fob_battery_jacks
)
import pandas as pd
import os

# --- Load Data and Train Models (or Load Saved Models) ---
# In a real app, you might load pre-trained models instead of retraining every time.
# For demonstration, we'll load and train.
data_file_path = 'Base Data CSV-2.csv' # Adjust path as needed
if not os.path.exists(data_file_path):
    st.error(f"Data file not found at {data_file_path}. Please upload the data file.")
else:
    cleaned_df = load_and_clean_data(data_file_path)

    if cleaned_df is not None:
        X, y = select_features(cleaned_df)

        if X is not None and y is not None:
            model_total_cases_estimation_tree, model_pf_cases_estimation_tree = train_estimation_models(cleaned_df)
            lasso_model, X_test, y_test = train_main_model(X, y) # X_test, y_test for evaluation if needed
            percentile_75 = cleaned_df[['Total Outbound Prod.', 'Total PF Prod.', 'PF Line Items', 'PF Items/OBD']].quantile(0.75)

            models = {
                'model_total_cases_estimation_tree': model_total_cases_estimation_tree,
                'model_pf_cases_estimation_tree': model_pf_cases_estimation_tree,
                'lasso_model': lasso_model
            }

            # --- Streamlit App Layout ---
            st.title("DC SKP - Labor Demand Tool")

            st.header("Configure Inputs")

            # Running Plants Section
            st.subheader("Running Plants")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                plant_ple = st.checkbox("Dairy & Juice", value=False)
            with col2:
                plant_e1 = st.checkbox("Egron-1", value=False)
            with col3:
                plant_e2 = st.checkbox("Egron-2", value=False)
            with col4:
                plant_waters = st.checkbox("Waters", value=False)
            running_plants_checked = {'PLE': plant_ple, 'E1': plant_e1, 'E2': plant_e2, 'Waters': plant_waters}


            # F&B Outbound Inputs
            st.subheader("F&B Outbound Input")
            fob_vehicles_in_plan = st.number_input("Expected Arrival F&B", value=0, min_value=0)
            fob_orders = st.number_input("Total OBDs", value=0, min_value=0)
            fob_input_data = {'Vehicles in Plan': fob_vehicles_in_plan, 'Orders': fob_orders}

            # NW PET Outbound Inputs
            st.subheader("NW PET Outbound Input")
            nwpet_expected_arrival = st.number_input("Expected Arrival NW PET", value=0, min_value=0)

            # DRP Inputs
            st.subheader("DRP Input")
            drp_pending_loads = st.number_input("Pending Loads", value=0, min_value=0)

            # Manual Labor Inputs
            st.subheader("Manual Labor Input")
            manual_vehicles = st.number_input("Manual Offloading Vehicles", value=0, min_value=0)


            # --- Generate Labor Demand Button ---
            if st.button("Generate Labor Demand"):
                st.subheader("Optimized Labor Demand")

                # Perform calculations and predictions
                inbound_jacks = calculate_inbound_jacks(running_plants_checked)
                nwpet_jacks = calculate_nwpet_jacks(nwpet_expected_arrival)
                drp_jacks = calculate_drp_jacks(drp_pending_loads)
                manual_labor = calculate_manual_labor(manual_vehicles)
                nwpet_labor = calculate_nwpet_labor(nwpet_expected_arrival)

                fob_predicted_jacks = predict_fob_battery_jacks(fob_input_data, models, percentile_75)

                # Display results
                st.subheader("F&B")
                if fob_predicted_jacks is not None:
                     st.write(f"Outbound F&B Battery Jack: {fob_predicted_jacks}")
                st.write(f"DRP Battery Jack: {drp_jacks}")
                st.write(f"Inbound Battery Jack: {inbound_jacks}")
                st.write("Pallet Handling Battery Jack: 1") # Fixed output
                st.write(f"Manual Labor: {manual_labor}")

                st.subheader("Waters")
                st.write(f"NW PET Battery Jack: {nwpet_jacks}")
                st.write(f"NW PET Labor: {nwpet_labor}")

        else:
            st.error("Failed to select features or target from the data.")
    else:
        st.error("Failed to load or clean data.")