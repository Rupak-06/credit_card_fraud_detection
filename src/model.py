from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

def build_autoencoder(input_dim):
    """
    Builds the Denoising Autoencoder model architecture.
    
    Args:
        input_dim (int): The number of features in the input data.
        
    Returns:
        A compiled Keras Autoencoder model.
    """
    # Define encoding and latent space dimensions
    encoding_dim = 14
    latent_dim = 7

    # --- Encoder ---
    input_layer = Input(shape=(input_dim, ), name="Input")
    
    # --- EDIT: Add Dropout layer to create a Denoising Autoencoder ---
    # This adds "noise" to the input, forcing the model
    # to learn more robust features.
    corrupted_input = Dropout(0.1)(input_layer) 
    # --- END EDIT ---

    
    # "encoded" is the encoded representation of the input
    # --- EDIT: Feed the *corrupted* input to the first layer ---
    encoder_layer_1 = Dense(encoding_dim, activation='relu', name="Encoder_1")(corrupted_input)
    encoder_layer_2 = Dense(latent_dim, activation='relu', name="Encoder_2_Latent")(encoder_layer_1)

    # --- Decoder ---
    # "decoded" is the lossy reconstruction of the input
    decoder_layer_1 = Dense(encoding_dim, activation='relu', name="Decoder_1")(encoder_layer_2)
    decoder_layer_2 = Dense(input_dim, activation='sigmoid', name="Decoder_2_Output")(decoder_layer_1)

    # --- Autoencoder Model ---
    # This model maps an input to its reconstruction
    autoencoder = Model(inputs=input_layer, outputs=decoder_layer_2, name="Autoencoder")

    # Compile the model
    # We use Mean Squared Error (MSE) to measure the reconstruction error
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Denoising Autoencoder model built and compiled successfully.")
    autoencoder.summary()
    
    return autoencoder

if __name__ == "__main__":
    # This allows you to run this file directly to test it
    print("\nTesting model build:")
    # --- EDIT: Updated dim to match new data preprocessing ---
    dummy_input_dim = 31 
    model = build_autoencoder(dummy_input_dim)