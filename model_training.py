# model_training.py

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
import pandas as pd

def train_estimation_models(df):
    """
    Trains Decision Tree Regressor models for estimating Total Cases Dispatched
    and PF Cases Dispatched.

    Args:
        df (pandas.DataFrame): The cleaned DataFrame.

    Returns:
        tuple: Trained model for Total Cases (model_total_cases_estimation_tree)
               and trained model for PF Cases (model_pf_cases_estimation_tree).
    """
    estimation_features = ['Orders', 'Vehicles in Plan']
    target_total_cases = 'Total Cases Dispatched'
    target_pf_cases = 'PF Cases Dispatched'

    # Ensure required columns exist
    if not all(col in df.columns for col in estimation_features + [target_total_cases, target_pf_cases]):
        print("Error: Required columns for estimation model training are missing.")
        return None, None

    X_est = df[estimation_features]
    y_est_total_cases = df[target_total_cases]
    y_est_pf_cases = df[target_pf_cases]

    # Train Total Cases estimation model
    model_total_cases_estimation_tree = DecisionTreeRegressor(random_state=42)
    model_total_cases_estimation_tree.fit(X_est, y_est_total_cases)

    # Train PF Cases estimation model
    model_pf_cases_estimation_tree = DecisionTreeRegressor(random_state=42)
    model_pf_cases_estimation_tree.fit(X_est, y_est_pf_cases)

    print("Estimation models trained successfully.")
    return model_total_cases_estimation_tree, model_pf_cases_estimation_tree

def train_main_model(X, y):
    """
    Trains the main Lasso regression model.

    Args:
        X (pandas.DataFrame): Features DataFrame.
        y (pandas.Series): Target Series.

    Returns:
        sklearn.linear_model.Lasso: The trained Lasso model.
    """
    # Split data for training the main model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train Lasso model
    lasso_model = Lasso(alpha=1.0) # Using alpha=1.0 as in the notebook
    lasso_model.fit(X_train, y_train)

    print("Main Lasso model trained successfully.")
    return lasso_model, X_test, y_test

# Example usage (for testing purposes)
if __name__ == '__main__':
    # This would typically be called after data_processing
    # from data_processing import load_and_clean_data, select_features
    # cleaned_df = load_and_clean_data()
    # if cleaned_df is not None:
    #     X, y = select_features(cleaned_df)
    #     if X is not None and y is not None:
    #         total_cases_model, pf_cases_model = train_estimation_models(cleaned_df)
    #         main_model, X_test, y_test = train_main_model(X, y)
    #         print("\nModels trained.")
    pass # Placeholder as we don't have the data loading here directly