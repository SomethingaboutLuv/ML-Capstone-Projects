"""
Data Exploration and Analysis
Autism Prediction Project - Exploratory Data Analysis (EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataExplorer:
    """
    A class for exploring and analyzing the autism prediction dataset.
    """
    
    def __init__(self, filepath):
        """
        Initialize data explorer with dataset.
        
        Args:
            filepath (str): Path to the CSV file
        """
        self.data = pd.read_csv(filepath)
        self.filepath = filepath
        
    def basic_info(self):
        """
        Display basic information about the dataset.
        """
        print("="*60)
        print("DATASET BASIC INFORMATION")
        print("="*60)
        print(f"\nDataset Shape: {self.data.shape}")
        print(f"Number of Rows: {self.data.shape[0]}")
        print(f"Number of Columns: {self.data.shape[1]}")
        print("\nFirst few rows:")
        print(self.data.head())
        print("\nData Types:")
        print(self.data.dtypes)
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        print("="*60 + "\n")
    
    def statistical_summary(self):
        """
        Display statistical summary of the dataset.
        """
        print("="*60)
        print("STATISTICAL SUMMARY")
        print("="*60)
        print(self.data.describe())
        print("="*60 + "\n")
    
    def target_distribution(self, target_column='autism'):
        """
        Analyze target variable distribution.
        
        Args:
            target_column (str): Name of target column
        """
        print(f"\nTarget Variable Distribution ({target_column}):")
        print(self.data[target_column].value_counts())
        print(f"\nTarget Distribution Percentage:")
        print(self.data[target_column].value_counts(normalize=True) * 100)
        
        # Plot
        plt.figure(figsize=(8, 5))
        self.data[target_column].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title(f'Distribution of {target_column}')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('target_distribution.png')
        print("\nTarget distribution plot saved as 'target_distribution.png'")
        plt.show()
    
    def correlation_analysis(self, target_column='autism'):
        """
        Analyze feature correlations.
        
        Args:
            target_column (str): Name of target column
        """
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        correlation_matrix = self.data.corr()
        
        # Get top correlations with target
        if target_column in correlation_matrix.columns:
            target_corr = correlation_matrix[target_column].sort_values(ascending=False)
            print(f"\nTop 10 Features Correlated with {target_column}:")
            print(target_corr.head(10))
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        print("\nCorrelation heatmap saved as 'correlation_heatmap.png'")
        plt.show()
    
    def missing_values_analysis(self):
        """
        Analyze and visualize missing values.
        """
        print("\n" + "="*60)
        print("MISSING VALUES ANALYSIS")
        print("="*60)
        
        missing = self.data.isnull().sum()
        missing_percent = (missing / len(self.data)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_percent
        })
        
        print(missing_df[missing_df['Missing Count'] > 0])
        
        if missing.sum() > 0:
            plt.figure(figsize=(10, 6))
            missing[missing > 0].plot(kind='bar')
            plt.title('Missing Values Count')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig('missing_values.png')
            print("\nMissing values plot saved as 'missing_values.png'")
            plt.show()
        else:
            print("\nNo missing values found!")
    
    def full_eda(self, target_column='autism'):
        """
        Run complete exploratory data analysis.
        
        Args:
            target_column (str): Name of target column
        """
        print("\n" + "#"*60)
        print("# STARTING EXPLORATORY DATA ANALYSIS")
        print("#"*60 + "\n")
        
        self.basic_info()
        self.statistical_summary()
        self.target_distribution(target_column)
        self.missing_values_analysis()
        self.correlation_analysis(target_column)
        
        print("\n" + "#"*60)
        print("# EDA COMPLETED SUCCESSFULLY")
        print("#"*60 + "\n")


if __name__ == "__main__":
    print("Data Exploration Module")
    print("This module is designed to be imported and used in Jupyter notebooks or scripts.")