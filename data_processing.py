# data_processing.py

import pandas as pd
import numpy as np

def load_and_clean_data(file_path='/content/Base Data CSV-2.csv'):
    """
    Loads the data, converts relevant columns to numeric, and handles outliers.

    Args:
        file_path (str): The path to the CSV data file.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please make sure the file is in the correct location.")
        return None

    numeric_cols = ['Expected Arrival', 'Actual Arrival', 'Carry Forward ',
                    'Vehicles in Plan', 'Loading Labor', 'BJ Operators',
                    'True BJ Plan', 'Pallet Sorters', 'Truck Inspectors',
                    'Total Cases Dispatched', 'PF Cases Dispatched',
                    'Total Outbound Prod.', 'Total PF Prod.', 'PF Line Items',
                    'Orders', 'PF Items/OBD']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

    # Calculate PF% and add it
    df['PF%'] = (df['PF Cases Dispatched'] / df['Total Cases Dispatched']) * 100
    df['PF%'] = df['PF%'].replace([np.inf, -np.inf], np.nan).fillna(0) # Handle potential division by zero

    # Handle outliers using the IQR method
    outlier_cols = ['Total Cases Dispatched', 'PF Cases Dispatched', 'Total Outbound Prod.',
                    'Total PF Prod.', 'PF Line Items', 'Orders']

    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df

def select_features(df):
    """
    Selects the relevant features and target variable.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing the features DataFrame (X) and target Series (y).
    """
    selected_features = ['Vehicles in Plan', 'Total Cases Dispatched', 'PF Cases Dispatched',
                         'Total Outbound Prod.', 'Total PF Prod.', 'PF Line Items', 'Orders', 'PF Items/OBD'] # Using PF Items/OBD as per the selected columns earlier
    target = 'True BJ Plan'

    # Ensure all selected features exist in the DataFrame
    available_features = [f for f in selected_features if f in df.columns]
    if len(available_features) != len(selected_features):
        missing = list(set(selected_features) - set(available_features))
        print(f"Warning: Missing selected features in DataFrame: {missing}")
        # Use only available features
        selected_features = available_features

    if target not in df.columns:
         print(f"Error: Target variable '{target}' not found in DataFrame.")
         return None, None

    X = df[selected_features]
    y = df[target]

    return X, y

# Example usage (for testing purposes)
if __name__ == '__main__':
    cleaned_df = load_and_clean_data()
    if cleaned_df is not None:
        print("Data loaded and cleaned successfully.")
        print(cleaned_df.head())
        X, y = select_features(cleaned_df)
        if X is not None and y is not None:
            print("\nFeatures and target selected.")
            print("Features (X) shape:", X.shape)
            print("Target (y) shape:", y.shape)