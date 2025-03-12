# Attrition Model

## Overview
This repository contains code for an attrition model designed to predict whether employees will leave the company. The model is built using logistic regression, which is a statistical method for analyzing datasets in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes).

## Model Explanation
The attrition model utilizes logistic regression to analyze employee data and predict the likelihood of attrition. Logistic regression is particularly suited for this type of binary classification problem as it estimates the probability of a certain event (employee attrition) occurring.

## How to Use
1. **Data Preparation**:
   - Ensure your dataset is cleaned and preprocessed for analysis.
   - The dataset should contain relevant features that are likely to influence employee attrition.

2. **Model Training**:
   - Run the provided code to train the logistic regression model on your dataset.

3. **Prediction**:
   - Use the trained model to predict employee attrition on new data.

## Requirements
- Python 3.x
- Pandas
- Matplotlib
- Scikit-learn

## Installation
```bash
pip install pandas matplotlib scikit-learn