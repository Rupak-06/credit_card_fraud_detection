# (This is a NEW FILE: src/evaluate_isolation_forest.py)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest

# Import our custom data loader
from src.data_preprocessing import load_and_prep_data

def run_isolation_forest_evaluation():
    """
    Trains and evaluates an Isolation Forest model.
    """
    print("Starting Isolation Forest evaluation...")

    # 1. Load Data
    # Use the *same* data as the autoencoder
    X_train_normal, X_test, y_test = load_and_prep_data()
    
    if X_test is None or y_test is None:
        return # Stop if data loading failed

    # Calculate the contamination rate (expected % of fraud)
    # This is a key parameter for the Isolation Forest
    fraud_rate = sum(y_test) / len(y_test)
    print(f"Calculated contamination rate (fraud %): {fraud_rate:.6f}")

    # 2. Build and Train Model
    # We train ONLY on normal data, just like our autoencoder
    print("Training Isolation Forest...")
    model = IsolationForest(contamination=fraud_rate, random_state=42)
    model.fit(X_train_normal)

    # 3. Get Predictions
    # model.predict gives -1 for anomalies (fraud) and 1 for inliers (normal)
    print("Making predictions on test set...")
    scores = model.decision_function(X_test)
    predictions = model.predict(X_test)

    # 4. Convert predictions to our format: 0=normal, 1=fraud
    # This is a crucial step for comparison!
    y_pred = [1 if p == -1 else 0 for p in predictions]

    # 5. Evaluate
    print("\n--- Isolation Forest Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fraud (1)']))

    # 6. Plot and Save Confusion Matrix
    print("\n--- Isolation Forest Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Predicted: Normal', 'Predicted: Fraud'],
                yticklabels=['Actual: Normal', 'Actual: Fraud'])
    plt.title('Isolation Forest Confusion Matrix')
    
    # Save the plot
    cm_plot_path = os.path.join("models", "iforest_confusion_matrix.png")
    plt.savefig(cm_plot_path)
    print(f"Confusion matrix plot saved to {cm_plot_path}")

if __name__ == "__main__":
    run_isolation_forest_evaluation()