import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Import our custom modules
from src.data_preprocessing import load_and_prep_data
from src.model import build_autoencoder

# Define file paths
MODEL_SAVE_PATH = os.path.join("models", "fraud_autoencoder.h5")

def run_training():
    """
    Main function to run the model training pipeline.
    """
    print("Starting training process...")
    
    # 1. Load Data
    X_train_normal, _, _ = load_and_prep_data()
    
    if X_train_normal is None:
        return # Stop if data loading failed

    # 2. Build Model
    input_dim = X_train_normal.shape[1]
    autoencoder = build_autoencoder(input_dim)

    # 3. Train Model
    print("\nTraining autoencoder...")
    
    # Set up EarlyStopping
    # This stops training if the validation loss doesn't improve for 10 epochs
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10, 
                             verbose=1, restore_best_weights=True)

    # Train the autoencoder
    # NOTE: The input (x) and target (y) are BOTH X_train_normal.
    # The model is learning to reconstruct its own input.
    history = autoencoder.fit(
        X_train_normal, X_train_normal,
        epochs=100,
        batch_size=256,
        shuffle=True,
        validation_split=0.1, # Use 10% of normal data for validation
        callbacks=[early_stop],
        verbose=1
    )

    # 4. Save Model
    # Ensure the 'models' directory exists
    os.makedirs("models", exist_ok=True)
    autoencoder.save(MODEL_SAVE_PATH)
    print(f"\nModel saved successfully to {MODEL_SAVE_PATH}")

    # 5. Plot and save training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    # Save the plot
    plot_path = os.path.join("models", "training_loss_plot.png")
    plt.savefig(plot_path)
    print(f"Training plot saved to {plot_path}")
    # plt.show() # Uncomment this if running interactively

if __name__ == "__main__":
    # This allows you to run: python src/train.py
    run_training()