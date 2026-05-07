"""
Main script for Autism Prediction Model
Project-1: Beginner Level
"""

from autism_prediction_model import AutismPredictionModel
import sys


def main():
    """
    Main function to run the autism prediction model.
    """
    print("="*60)
    print("AUTISM PREDICTION USING MACHINE LEARNING")
    print("Beginner Project - 1")
    print("="*60 + "\n")
    
    # Initialize model
    model = AutismPredictionModel(random_state=42)
    
    # Load data
    print("Step 1: Loading data...")
    # Update this path to your dataset
    data = model.load_data('path/to/your/dataset.csv')
    
    if data is None:
        print("Failed to load data. Exiting...")
        sys.exit(1)
    
    # Preprocess data
    print("\nStep 2: Preprocessing data...")
    model.preprocess_data(data, target_column='autism', test_size=0.2)
    
    # Train model
    print("\nStep 3: Training model...")
    model.train()
    
    # Evaluate model
    print("\nStep 4: Evaluating model...")
    metrics = model.evaluate()
    
    # Visualizations
    print("\nStep 5: Generating visualizations...")
    model.plot_confusion_matrix(metrics)
    model.feature_importance()
    
    print("\n" + "="*60)
    print("PROCESS COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()