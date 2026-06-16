# Disease Prediction from Medical Data

## Overview
A machine learning project to predict whether a patient has heart disease based on clinical data. Built as part of the CodeAlpha Machine Learning Internship.

## Dataset
UCI Heart Disease Dataset — 303 patients, 13 features including age, blood pressure, cholesterol, and maximum heart rate.

## Models Used
- Logistic Regression
- Random Forest
- SVM
- XGBoost

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Best Model
Random Forest — ROC-AUC: 0.893

## How to Run
1. Install dependencies: pip install numpy pandas matplotlib seaborn scikit-learn xgboost
2. Place heart.csv in the same folder as the script
3. Run: python task4_disease_prediction.py

## Results
The results chart (task4_results.png) includes confusion matrix, ROC curves, and feature importance plot.

## Tools & Libraries
Python, Scikit-learn, XGBoost, Pandas, Matplotlib, Seaborn
