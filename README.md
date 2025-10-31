# Diabetes Prediction Model

A comprehensive machine learning project that predicts diabetes diagnosis using clinical and lifestyle data. The project implements multiple machine learning algorithms to identify the most important risk factors for diabetes and achieve high prediction accuracy.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Workflow](#analysis-workflow)
- [Model Performance](#model-performance)
- [Visualizations](#visualizations)
- [Contributing](#contributing)

## Project Overview

This project analyzes diabetes health indicators to predict diabetes diagnosis using advanced machine learning techniques. The analysis includes:

- Comprehensive exploratory data analysis (EDA)
- Feature engineering and selection
- Multiple model testing and comparison
- Hyperparameter optimization
- Model evaluation and performance assessment

The project achieves approximately 92% accuracy in predicting diabetes diagnosis using Random Forest and XGBoost algorithms.

## Dataset

The dataset contains 100,000 entries with 31 features related to diabetes indicators:

- Clinical measurements (BMI, blood pressure, cholesterol, glucose levels, insulin, HbA1c)
- Demographics (age, gender, ethnicity, education, income)
- Lifestyle factors (smoking, alcohol consumption, physical activity, diet score)
- Medical history (family history, hypertension, cardiovascular history)
- Additional metrics (sleep hours, screen time, waist-to-hip ratio)

## Features

Key features analyzed in the dataset include:
- `age` - Age of the patient
- `gender` - Gender of the patient
- `ethnicity` - Ethnic background
- `bmi` - Body Mass Index
- `hba1c` - Glycated hemoglobin levels
- `glucose_fasting` - Fasting glucose levels
- `glucose_postprandial` - Post-meal glucose levels
- `insulin_level` - Insulin levels
- `physical_activity_minutes_per_week` - Weekly physical activity
- `diet_score` - Diet quality score
- `family_history_diabetes` - Family history of diabetes
- `diagnosed_diabetes` - Target variable (0 = No, 1 = Yes)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd diabetes_modelling
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

If no requirements.txt exists, install the following packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

## Usage

1. Place the diabetes dataset file `diabetes-health-indicators-dataset.zip` in the `Data/` directory

2. Run the Jupyter notebook:
```bash
jupyter notebook diabetes_modelling.ipynb
```

## Analysis Workflow

1. **Data Loading**: Extract and load the dataset from the zip file
2. **Data Preprocessing**: Handle missing values, transform features, and scale data
3. **Exploratory Data Analysis**: Visualize distributions and relationships
4. **Feature Engineering**: 
   - Create new feature `free_hours_per_week` (24 - sleep hours - screen time)
   - Apply skewness transformation to numerical features
   - Use one-hot encoding for categorical variables
5. **Feature Selection**: Use Random Forest to identify important features
6. **Model Training**: 
   - Split data using StratifiedShuffleSplit (80/20 train/test split)
   - Train multiple algorithms (Random Forest, XGBoost, SGD, Logistic Regression)
   - Use pipelines for preprocessing and modeling
7. **Hyperparameter Tuning**: Optimize model parameters using GridSearchCV
8. **Model Evaluation**: Assess performance using accuracy, precision, recall, F1-score, ROC curves, and confusion matrices
9. **Model Persistence**: Save the best performing model

## Model Performance

The models achieved the following results:

| Model | Accuracy | Key Observations |
|-------|----------|------------------|
| Random Forest | ~92% | Best overall performance; high precision and recall balance |
| XGBoost | ~92% | AUC of approximately 0.942; excellent discrimination |
| SGD | ~86.5% | Good baseline performance |
| Logistic Regression | ~85.6% | Lower accuracy but interpretable results |

The Random Forest model was selected as the best performing model based on its balance of accuracy (~92%) and good precision-recall characteristics.

## Visualizations

The project includes various visualizations:

- Distribution histograms for all features
- Correlation matrices
- Box plots for different feature groups
- ROC curves for model performance
- Precision-Recall curves
- Pair plots for key variables

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request