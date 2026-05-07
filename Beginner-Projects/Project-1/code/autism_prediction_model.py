"""
Autism Prediction using Machine Learning
Project-1: Beginner Level

This module contains the main machine learning model for autism prediction.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class AutismPredictionModel:
    """
    A machine learning model for predicting autism spectrum disorder.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the model.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        
    def load_data(self, filepath):
        """
        Load dataset from CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            data = pd.read_csv(filepath)
            print(f"Dataset loaded successfully. Shape: {data.shape}")
            return data
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
            return None
    
    def preprocess_data(self, data, target_column='autism', test_size=0.2):
        """
        Preprocess and split the dataset.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_column (str): Name of target column
            test_size (float): Proportion of test set
        """
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
    
    def train(self):
        """
        Train the machine learning model.
        """
        print("Training model...")
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed!")
    
    def evaluate(self):
        """
        Evaluate model performance on test set.
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        self.predictions = self.model.predict(self.X_test)
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.predictions),
            'precision': precision_score(self.y_test, self.predictions),
            'recall': recall_score(self.y_test, self.predictions),
            'f1_score': f1_score(self.y_test, self.predictions),
            'confusion_matrix': confusion_matrix(self.y_test, self.predictions)
        }
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print("="*50 + "\n")
        
        return metrics
    
    def plot_confusion_matrix(self, metrics):
        """
        Plot confusion matrix.
        
        Args:
            metrics (dict): Dictionary containing evaluation metrics
        """
        plt.figure(figsize=(8, 6))
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved as 'confusion_matrix.png'")
        plt.show()
    
    def feature_importance(self):
        """
        Get and plot feature importance.
        """
        feature_importance = self.model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_importance)), feature_importance)
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Feature importance plot saved as 'feature_importance.png'")
        plt.show()


if __name__ == "__main__":
    print("Autism Prediction Model - Beginner Project")
    print("This module is designed to be imported and used in Jupyter notebooks or scripts.")