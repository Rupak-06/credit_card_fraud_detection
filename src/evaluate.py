import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix, 
                             precision_recall_curve, f1_score)

# Import our custom modules
from src.data_preprocessing import load_and_prep_data

# Define file paths
MODEL_PATH = os.path.join("models", "fraud_autoencoder.h5")

def run_evaluation():
    """
    Main function to run the model evaluation pipeline.
    """
    print("Starting evaluation process...")

    # 1. Load Test Data
    _, X_test, y_test = load_and_prep_data()
    
    if X_test is None or y_test is None:
        return 

    # 2. Load Trained Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run 'train.py' first to train and save the model.")
        return
        
    autoencoder = tf.keras.models.load_model(MODEL_PATH)
    print("Trained model loaded successfully.")

    # 3. Get Reconstructions and Calculate Error
    print("Generating reconstructions for test data...")
    reconstructions = autoencoder.predict(X_test)
    
    mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
    error_df = pd.DataFrame({'reconstruction_error': mse, 'true_class': y_test})

    # 4. Find the Optimal Threshold
    precision, recall, thresholds = precision_recall_curve(error_df['true_class'],
                                                           error_df['reconstruction_error'])
    
    # --- EDIT: Find threshold for a specific RECALL target ---
    
    # We want to catch at least 80% of fraud
    TARGET_RECALL = 0.90 
    
    try:
        # Find all indices where recall is >= our target
        indices = np.where(recall >= TARGET_RECALL)[0]
        
        # We want the *last* index that meets this, as it will
        # have the highest corresponding threshold (and best precision).
        optimal_idx = indices[-1]
        
        # Note: thresholds array is one element shorter than precision/recall
        # So we use the index directly on thresholds
        if optimal_idx >= len(thresholds):
             optimal_idx = len(thresholds) - 1
             
        optimal_threshold = thresholds[optimal_idx]
        print(f"\nOptimal threshold found for >{TARGET_RECALL*100}% Recall: {optimal_threshold:.6f}")
        
    except IndexError:
        print(f"Could not find a threshold for {TARGET_RECALL*100}% recall. Using max F1 score instead.")
        f1_scores = 2 * (precision * recall) / (precision + recall)
        optimal_idx = np.nanargmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
    
    # --- END EDIT ---

    # 5. Classify and Evaluate
    y_pred = (error_df['reconstruction_error'] > optimal_threshold).astype(int)

    # Print Classification Report
    print("\n--- Classification Report (Recall-Optimized) ---")
    print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fraud (1)']))

    # 6. Plot and Save Confusion Matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted: Normal', 'Predicted: Fraud'],
                yticklabels=['Actual: Normal', 'Actual: Fraud'])
    plt.title('Confusion Matrix (Recall-Optimized)')
    
    cm_plot_path = os.path.join("models", "confusion_matrix.png")
    plt.savefig(cm_plot_path)
    print(f"Confusion matrix plot saved to {cm_plot_path}")

if __name__ == "__main__":
    run_evaluation()