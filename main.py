import argparse
import os

# Set TensorFlow log level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Import the main functions from our source files
# We use 'try-except' for cleaner error handling if modules are not found
try:
    from src import train
    from src import evaluate
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure your project structure is correct and __init__.py exists in 'src/'.")
    exit(1)

def main():
    """
    Main entry point for the credit fraud detection project.
    
    Parses command-line arguments to run either the training
    or evaluation pipeline.
    """
    
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection Project"
    )
    
    # Add an argument
    parser.add_argument(
        '--action',
        type=str,
        choices=['train', 'evaluate'],
        required=True,
        help="Specify the action to run: 'train' or 'evaluate'"
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Execute the chosen action
    if args.action == 'train':
        print("--- Running Model Training ---")
        train.run_training()
        print("--- Training Finished ---")
        
    elif args.action == 'evaluate':
        print("--- Running Model Evaluation ---")
        evaluate.run_evaluation()
        print("--- Evaluation Finished ---")

if __name__ == "__main__":
    main()      