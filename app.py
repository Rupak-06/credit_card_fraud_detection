from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# --- Your Threshold ---
FRAUD_THRESHOLD = 1.304159

# --- File Paths ---
MODEL_PATH = os.path.join("models", "fraud_autoencoder.h5")
SCALER_PATH = os.path.join("models", "scaler.pkl")

# --- Load Model and Scaler ---
@st.cache_resource
def load_autoencoder_model():
    """Loads the compiled Keras autoencoder model."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model '{MODEL_PATH}': {e}")
        st.error("Please run 'train.py' to generate the model file.")
        return None

@st.cache_resource
def load_scaler():
    """Loads the joblib-dumped scaler."""
    try:
        scaler = joblib.load(SCALER_PATH)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler '{SCALER_PATH}': {e}")
        st.error("Please run 'src/data_preprocessing.py' to generate the scaler file.")
        return None

# --- NEW FUNCTION: Converts error to a 0-100% confidence score ---
def get_fraud_confidence(error, threshold, steepness=1.0):
    """
    Maps the reconstruction error to a 0-100% confidence score
    using a sigmoid function.
    
    - steepness: Controls how "sharp" the percentage change is.
                 A higher value makes it jump from 0% to 100% faster.
                 1.0 is a good, smooth default.
    """
    confidence = 1 / (1 + np.exp(-steepness * (error - threshold)))
    return confidence

# --- Helper Function for Preprocessing ---
def preprocess_input(time_val, amount_val, v_features_list, scaler_instance):
    """
    Takes raw user input and preprocesses it to match the
    31-feature input required by the model.
    """
    
    # 1. Process Time (sin/cos transform)
    seconds_in_day = 24 * 60 * 60
    time_of_day = time_val % seconds_in_day
    time_sin = np.sin(2 * np.pi * time_of_day / seconds_in_day)
    time_cos = np.cos(2 * np.pi * time_of_day / seconds_in_day)
    
    # 2. Process Amount (scaling)
    amount_array = np.array([amount_val]).reshape(-1, 1)
    scaled_amount = scaler_instance.transform(amount_array).flatten()
    
    # 3. Combine all 31 features
    v_features = np.array(v_features_list)
    final_features = np.hstack([v_features, scaled_amount, time_sin, time_cos])
    
    return final_features.reshape(1, -1)

# --- Streamlit Web Page UI ---
st.set_page_config(page_title="Fraud Detector", layout="wide")
st.title("💳 Credit Card Fraud Detector (Autoencoder)")

st.info("""
Enter the transaction details. This app replicates the full preprocessing pipeline 
(Time → sin/cos, Amount → scaled) and feeds the final 31 features 
to the trained autoencoder model.
""")

model = load_autoencoder_model()
scaler = load_scaler()

if model is None or scaler is None:
    st.error("Application cannot start. Please see error messages above.")
else:
    with st.form("transaction_form"):
        st.subheader("Transaction Details")
        
        col1, col2 = st.columns(2)
        with col1:
            time_val = st.number_input(
                "Transaction Time (Raw 'Time' feature from dataset)", 
                min_value=0.0, 
                format="%.2f",
                help="The number of seconds elapsed since the first transaction."
            )
        with col2:
            amount_val = st.number_input(
                "Transaction Amount (Raw 'Amount' feature)", 
                min_value=0.0, 
                format="%.2f",
                help="The transaction amount (e.g., 99.99)."
            )
            
        # --- EDITED LINE ---
        st.subheader("Anonymized Features (Feature 1 - Feature 28)")
        v_features_str = st.text_area(
            # --- EDITED LINE ---
            "Paste Feature 1 through Feature 28 as a comma-separated list",
            height=150,
            help="Example: -1.35, -0.07, 2.53, ..., 1.25, -0.21"
        )
        
        submitted = st.form_submit_button("Check Transaction")

    # --- Processing Logic (after form is submitted) ---
    if submitted:
        try:
            v_features_list = [float(x.strip()) for x in v_features_str.split(',')]
            
            if len(v_features_list) != 28:
                # --- EDITED LINE ---
                st.error(f"Error: Expected 28 features, but got {len(v_features_list)}.")
            else:
                # 1. Preprocess
                final_features = preprocess_input(time_val, amount_val, v_features_list, scaler)
                
                # 2. Predict
                reconstructed_features = model.predict(final_features)
                
                # 3. Calculate Error
                mse = np.mean(np.power(final_features - reconstructed_features, 2), axis=1)
                reconstruction_error = mse[0]
                
                # --- UPDATED DISPLAY LOGIC ---
                
                # 4. Convert error to confidence percentage
                fraud_confidence = get_fraud_confidence(reconstruction_error, FRAUD_THRESHOLD, steepness=1.0)
                fraud_percentage = fraud_confidence * 100
                normal_percentage = (1 - fraud_confidence) * 100

                # 5. Display results
                st.subheader("Prediction Result")
                
                # Show the percentages first
                if fraud_percentage > 50:
                    st.error(f"Prediction: **{fraud_percentage:.2f}% FRAUDULENT**")
                    st.write(f"(This transaction is **{normal_percentage:.2f}%** likely to be legitimate)")
                else:
                    st.success(f"Prediction: **{normal_percentage:.2f}% LEGITIMATE**")
                    st.write(f"(This transaction is **{fraud_percentage:.2f}%** likely to be fraudulent)")
                
                # Show the raw scores for reference
                st.write("---")
                with st.expander("Show detailed scores"):
                    st.write(f"**Reconstruction Error:** `{reconstruction_error:.6f}`")
                    st.write(f"**Fraud Threshold (50% point):** `{FRAUD_THRESHOLD:.6f}`")

        except ValueError:
            # --- EDITED LINE ---
            st.error("Invalid input for features. Please enter only numbers, separated by commas.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")