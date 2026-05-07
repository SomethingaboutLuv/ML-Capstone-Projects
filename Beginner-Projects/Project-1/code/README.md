# Autism Prediction using Machine Learning
## Project-1: Beginner Level

### Overview
This project implements a machine learning model to predict autism spectrum disorder using patient features and behavioral data.

### Project Structure
```
code/
├── autism_prediction_model.py    # Main model class
├── main.py                        # Pipeline execution script
├── data_exploration.py            # EDA utilities
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

### Key Features
- **Random Forest Classifier** for autism prediction
- **Data Preprocessing** with StandardScaler normalization
- **Model Evaluation** with multiple metrics (Accuracy, Precision, Recall, F1)
- **Visualization Tools** for confusion matrix and feature importance
- **Exploratory Data Analysis** utilities

### Dependencies
```
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
```

### Installation
```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Run Complete Pipeline
```bash
python main.py
```

#### Option 2: Use in Jupyter Notebook
```python
from autism_prediction_model import AutismPredictionModel
from data_exploration import DataExplorer

# Initialize model
model = AutismPredictionModel()

# Load and preprocess data
data = model.load_data('your_dataset.csv')
model.preprocess_data(data)

# Train and evaluate
model.train()
metrics = model.evaluate()
```

### Dataset Format
Your CSV file should contain:
- Feature columns (e.g., behavioral indicators, scores)
- Target column named 'autism' (binary: 0 or 1)

### Output
The model generates:
- Confusion matrix visualization
- Feature importance plot
- Performance metrics report

### Model Performance
- **Algorithm**: Random Forest Classifier
- **Number of Trees**: 100
- **Train-Test Split**: 80-20
- **Scaling**: StandardScaler

### Future Improvements
- Hyperparameter tuning
- Cross-validation
- Additional algorithms (SVM, Gradient Boosting)
- Feature selection
- Class imbalance handling (SMOTE)

---

**Author**: ML Capstone Projects  
**Level**: Beginner  
**Last Updated**: 2026-05-07