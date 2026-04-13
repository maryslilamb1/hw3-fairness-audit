# HW3: Fairness & Bias Audit in Mental Health Data

## Overview
This project analyzes fairness and bias in a mental health treatment dataset by comparing two machine learning models: Logistic Regression and a Neural Network (MLPClassifier). The goal is to predict treatment outcome and evaluate whether the models perform differently across gender groups.

The project also applies threshold adjustment as a fairness mitigation method and compares fairness metrics before and after mitigation.

## Dataset
The dataset used in this project is `HW3_mental_health_diagnosis_treatment.csv`.

- 341 rows
- 17 columns

This is a binary classification task:
- Improved = 1
- Deteriorated = 0

Protected attribute:
- Gender
  - Female = 0
  - Male = 1

## Features Used
The protected attribute was not included in the feature set.

Features:
- Age
- Diagnosis
- Symptom Severity (1-10)
- Mood Score (1-10)
- Sleep Quality (1-10)
- Physical Activity (hrs/week)
- Medication
- Therapy Type
- Treatment Duration (weeks)
- Stress Level (1-10)
- Treatment Progress (1-10)
- AI-Detected Emotional State
- Adherence to Treatment (%)

## Models
This project compares:
1. Logistic Regression
2. Neural Network (MLPClassifier) with two hidden layers of 8 neurons each

## Fairness Evaluation
Overall metrics:
- Accuracy
- Precision
- Recall

Group-wise fairness metrics:
- Accuracy by group
- True Positive Rate (TPR)
- False Positive Rate (FPR)

## Fairness Mitigation
Threshold adjustment was applied to the Logistic Regression model.

Thresholds used:
- Female threshold = 0.50
- Male threshold = 0.40

## Files Included
- `hw3_fairness_audit.ipynb` — notebook with full code and outputs
- `hw3_fairness_audit.py` — Python script version of the project
- `HW3_mental_health_diagnosis_treatment.csv` — dataset
- `overall_results.csv` — overall model results
- `group_fairness_results.csv` — fairness metrics by gender group
- `mitigation_results.csv` — before vs after mitigation results
- `fairness_gap_comparison.csv` — fairness gap comparison
- `HW3_Report.tex` — LaTeX source file for the report
- `HW3_Report.pdf` — final report
- `README.md` — project documentation

## Required Libraries
- pandas
- numpy
- scikit-learn

Install with:

```bash
pip install pandas numpy scikit-learn
