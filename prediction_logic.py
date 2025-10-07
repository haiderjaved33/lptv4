# prediction_logic.py

import pandas as pd
import math

# Assuming models and percentile_75 are loaded or available in the application environment

def calculate_inbound_jacks(running_plants_checked):
    """Calculates Inbound Battery Jacks based on running plants."""
    return sum(running_plants_checked.values())

def calculate_nwpet_jacks(expected_arrival_nwpet):
    """Calculates NW PET Battery Jacks (1 for every 3 vehicles, rounded down)."""
    return math.floor(expected_arrival_nwpet / 3)

def calculate_drp_jacks(pending_loads):
    """Calculates DRP Battery Jacks (1 for every 4 loads, rounded up)."""
    return math.ceil(pending_loads / 4)

def calculate_manual_labor(manual_vehicles):
    """Calculates Manual Labor (2 per vehicle, rounded up)."""
    manual_labor = manual_vehicles * 2
    return math.ceil(manual_labor)

def calculate_nwpet_labor(expected_arrival_nwpet):
    """Calculates NW PET Labor (2 for every 4 vehicles, rounded down)."""
    nwpet_labor = (expected_arrival_nwpet / 4) * 2
    return math.floor(nwpet_labor)

def predict_fob_battery_jacks(input_data, models, percentile_75):
    """
    Predicts F&B Outbound Battery Jacks using the trained models and percentile data.

    Args:
        input_data (dict): Dictionary containing user inputs ('Vehicles in Plan', 'Orders').
        models (dict): Dictionary containing trained models
                       ('model_total_cases_estimation_tree', 'model_pf_cases_estimation_tree', 'lasso_model').
        percentile_75 (pandas.Series): Series containing 75th percentile values.

    Returns:
        int: The predicted and rounded-up number of Outbound F&B Battery Jacks.
             Returns None if models are missing or input data is invalid.
    """
    vehicles_in_plan = input_data.get('Vehicles in Plan', 0.0)
    orders = input_data.get('Orders', 0.0)

    model_total_cases_estimation_tree = models.get('model_total_cases_estimation_tree')
    model_pf_cases_estimation_tree = models.get('model_pf_cases_estimation_tree')
    lasso_model = models.get('lasso_model')

    if not all([model_total_cases_estimation_tree, model_pf_cases_estimation_tree, lasso_model]):
        print("Error: Required models for prediction are not loaded.")
        return None

    try:
        estimation_input_df = pd.DataFrame([[orders, vehicles_in_plan]], columns=['Orders', 'Vehicles in Plan'])
        estimated_total_cases = model_total_cases_estimation_tree.predict(estimation_input_df)[0]
        estimated_pf_cases = model_pf_cases_estimation_tree.predict(estimation_input_df)[0]

        total_outbound_prod = percentile_75.get('Total Outbound Prod.', 0.0)
        total_pf_prod = percentile_75.get('Total PF Prod.', 0.0)
        pf_line_items = percentile_75.get('PF Line Items', 0.0)
        pf_items_obd = percentile_75.get('PF Items/OBD', 0.0)


        input_values = [vehicles_in_plan, estimated_total_cases, estimated_pf_cases,
                        total_outbound_prod, total_pf_prod, pf_line_items, orders, pf_items_obd]
        input_df = pd.DataFrame([input_values], columns=['Vehicles in Plan', 'Total Cases Dispatched', 'PF Cases Dispatched',
                                                         'Total Outbound Prod.', 'Total PF Prod.', 'PF Line Items', 'Orders', 'PF Items/OBD'])

        predicted_plan = lasso_model.predict(input_df)[0]
        predicted_plan_rounded = math.ceil(predicted_plan)
        return predicted_plan_rounded

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Example usage (for testing purposes)
if __name__ == '__main__':
    # This would typically be called within the application after getting user inputs
    # and loading models/percentiles
    # from your_model_module import model_total_cases_estimation_tree, model_pf_cases_estimation_tree, lasso_model
    # from your_data_processing_module import percentile_75
    #
    # dummy_models = {
    #     'model_total_cases_estimation_tree': ...,
    #     'model_pf_cases_estimation_tree': ...,
    #     'lasso_model': ...
    # }
    # dummy_percentile_75 = pd.Series({...})
    # dummy_input_data = {'Vehicles in Plan': 30, 'Orders': 50}
    #
    # predicted_jacks = predict_fob_battery_jacks(dummy_input_data, dummy_models, dummy_percentile_75)
    # if predicted_jacks is not None:
    #     print(f"Predicted F&B Battery Jacks: {predicted_jacks}")

    # Example of independent calculations
    print("Inbound Jacks (PLE, E1 checked):", calculate_inbound_jacks({'PLE': True, 'E1': True, 'E2': False, 'Waters': False}))
    print("NW PET Jacks (5 vehicles):", calculate_nwpet_jacks(5))
    print("DRP Jacks (10 loads):", calculate_drp_jacks(10))
    print("Manual Labor (3 vehicles):", calculate_manual_labor(3))
    print("NW PET Labor (8 vehicles):", calculate_nwpet_labor(8))