import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # --- ADDED ---
import os      # --- ADDED ---

# Define the path to your data
DATA_PATH = "data/creditcard.csv"

def load_and_prep_data(data_path=DATA_PATH):
    """
    Loads, preprocesses, and splits the credit card fraud data.
    
    Returns:
        X_train_normal (pd.DataFrame): Training features (normal transactions only).
        X_test (pd.DataFrame): Test features (all transactions).
        y_test (pd.Series): Test labels (all transactions).
    """
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please download the dataset from Kaggle and place it in the 'data/' directory.")
        return None, None, None

    print("Data loaded successfully.")

    # 1. Preprocessing
    
    # Scale 'Amount'
    scaler = StandardScaler()
    data['Scaled_Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    
    # --- ADDED: Save the scaler ---
    # Ensure the 'models' directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save the scaler so the app.py can use it
    scaler_path = 'models/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    # --- END ADDED ---
    
    # --- EDIT: Feature Engineer 'Time' ---
    # Convert 'Time' (seconds) into a 24-hour cycle (in seconds)
    seconds_in_day = 24 * 60 * 60
    data['Time_of_Day'] = data['Time'] % seconds_in_day
    
    # Use sin/cos to make it cyclical (so 23:59 is close to 00:00)
    data['Time_sin'] = np.sin(2 * np.pi * data['Time_of_Day'] / seconds_in_day)
    data['Time_cos'] = np.cos(2 * np.pi * data['Time_of_Day'] / seconds_in_day)
    
    # Now drop the original 'Time', 'Amount', and the intermediate 'Time_of_Day'
    data = data.drop(['Time', 'Amount', 'Time_of_Day'], axis=1)
    # --- END EDIT ---

    # 2. Split features and labels
    X = data.drop('Class', axis=1)
    y = data['Class']

    # 3. Create Training and Test Sets
    # We stratify by 'y' to ensure both train and test sets have a
    # proportional representation of fraud cases.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Create the Autoencoder Training Set (NORMAL TRANSACTIONS ONLY)
    # We will train the autoencoder only on non-fraudulent data.
    X_train_normal = X_train[y_train == 0].copy()
    
    print(f"Total transactions in training set: {len(X_train)}")
    print(f"Normal transactions for autoencoder training: {len(X_train_normal)}")
    print(f"Total transactions in test set: {len(X_test)}")
    print(f"Fraud transactions in test set: {sum(y_test)}")

    # We only need to return the normal training data, and the full test set
    return X_train_normal, X_test, y_test

if __name__ == "__main__":
    # This allows you to run this file directly to test it
    X_train_norm, X_t, y_t = load_and_prep_data()
    if X_train_norm is not None:
        print("\nData preprocessing test successful:")
        # Note: Shape is now 31 features (V1-V28, Scaled_Amount, Time_sin, Time_cos)
        print(f"X_train_normal shape: {X_train_norm.shape}")
        print(f"X_test shape: {X_t.shape}")
        print(f"y_test shape: {y_t.shape}")