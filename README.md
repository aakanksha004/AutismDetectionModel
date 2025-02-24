
# Autism Prediction Model

## Overview
This project implements a machine learning solution to predict autism spectrum disorder using behavioral and demographic features. The model uses various machine learning algorithms and achieves high accuracy in predictions.

## Workflow

### 1. Importing Dependencies
The project utilizes various Python libraries for:
- Data manipulation and analysis (Pandas, NumPy)
- Data visualization (Matplotlib, Seaborn)
- Machine learning (Scikit-learn, XGBoost)
- Handling imbalanced data (SMOTE)
- Model persistence (Pickle)

### 2. Data Loading and Understanding
- Dataset loading and initial inspection
- Understanding data structure and features
- Checking for missing values and data types
- Initial statistical summary of the dataset

### 3. Exploratory Data Analysis (EDA)
- **Univariate Analysis**
  - Distribution analysis of numerical features using histograms
  - Outlier detection using box plots
  - Statistical summary of individual features

- **Multivariate Analysis**
  - Correlation analysis between features using heatmaps
  - Feature relationship examination
  - Target variable distribution analysis

### 4. Data Preprocessing
- **Outlier Treatment**
  - Identification of outliers using IQR method
  - Replacement of outliers with median values
  - Validation of outlier treatment

- **Feature Engineering**
  - Label encoding of categorical variables:
    - Gender
    - Ethnicity
    - Jaundice status
    - Country of residence
  - Feature scaling where necessary
  - Handling missing values if any

- **Data Balancing**
  - Implementation of SMOTE technique
  - Creation of synthetic samples
  - Validation of balanced dataset

- **Train-Test Split**
  - Data division into training and testing sets
  - Maintaining stratified sampling

### 5. Model Training
Three different models were evaluated using 5-fold cross-validation:
- Decision Tree: 86% accuracy
- Random Forest: 92% accuracy
- XGBoost: 90% accuracy
  
### 6. Model Selection and Hyperparameter Tuning
- Randomized search cross-validation
- Hyperparameter optimization
- Model performance comparison
- Selection of best performing model

### 7. Model Evaluation
-The final model achieved:
- Overall Accuracy: 81.87%
- Weighted Average F1-Score: 0.82
  
### 8. Model Persistence
- Saving the trained model
- Storing necessary encoders
- Maintaining feature information

### 9. Predictive System
- Model loading functionality
- Input data preprocessing pipeline
- Prediction generation system
- Probability score calculation

## Features
The model uses the following features:
- A1-A10 Scores: Behavioral markers (binary scores)
- Age: Numerical value
- Gender: Binary encoded
- Ethnicity: Categorical (encoded)
- Jaundice: Binary encoded
- Autism: Target variable
- Country of residence: Categorical (encoded)

## Performance Metrics
- Overall Accuracy: 81.87%
- Weighted Average F1-Score: 0.82
- Detailed class-wise performance:
  - Class 0 (Non-Autistic): 88% F1-Score
  - Class 1 (Autistic): 61% F1-Score


## Detailed Performance Metrics

### Classification Report
```
              precision    recall  f1-score   support

           0       0.89      0.87      0.88       124
           1       0.59      0.64      0.61        36

    accuracy                           0.82       160
   macro avg       0.74      0.75      0.75       160
weighted avg       0.82      0.82      0.82       160
```

### Confusion Matrix
```
[[108  16]
 [ 13  23]]
```

## Usage

### 1. Data Preparation
```python
# Load and preprocess data
# Apply the same preprocessing steps used during training
for column, encoder in encoders.items():
    input_data_df[column] = encoder.transform(input_data_df[column])
```

### 2. Load Model
```python
with open("best_model.pkl", "rb") as f:
    model_data = pickle.load(f)
```

### 3. Make Predictions
```python
prediction = loaded_model.predict(input_data_df)
pred_probs = loaded_model.predict_proba(input_data_df)
```

## Requirements
- Python 3.x
- scikit-learn
- XGBoost
- pandas
- numpy
- pickle
- imbalanced-learn (for SMOTE)

## Future Improvements
1. Feature importance analysis
2. Hyperparameter optimization
3. Model interpretability enhancements
4. Additional validation on diverse datasets
5. Web interface for easy prediction access
6. Improved data preprocessing pipeline
7. Enhanced cross-validation techniques
8. More sophisticated feature engineering

