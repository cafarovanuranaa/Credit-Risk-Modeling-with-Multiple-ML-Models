# Credit-Risk-Multimodel
Comprehensive credit risk modeling project with multiple machine learning models, feature engineering, hyperparameter tuning, and model interpretability.

## Dataset

The dataset used in this project is stored on Google Drive. You can access it here:  
[Credit Risk Application Dataset](https://drive.google.com/file/d/18Cy9Hoov-zyvi30iPsPHbsiHNi3kssD0/view?usp=drive_link)

## Project Workflow Summary


1. Data Preparation
- Loaded the application dataset and created a 10% random sample.
- Initial exploration: basic statistics, value counts, and missing value overview.
- Dropped low-variance columns and rows with missing TARGET.
- Removed SK_ID_CURR and columns with >50% missing values.
- Filled remaining missing values:
  • numeric count features → 0
  • selected categorical features → 'Unknown'
  • other categorical features → mode
  • remaining numeric features → mean
- Removed single-unique-value features.
- Grouped categorical variables:
  • ORGANIZATION_TYPE → organization_group
  • OCCUPATION_TYPE → occupation_group
- Created engineered features: credit_to_income, age_years, is_large_family, is_single_parent.
- Dropped redundant original columns: DAYS_BIRTH, OCCUPATION_TYPE, ORGANIZATION_TYPE.

2. Dataset Copies for Model Pipelines
- Created separate copies for different models:
  • data_lr → Logistic Regression with WOE transformation
  • data_knn → KNN with outlier handling and scaling
  • data_rf_xgb_lgb_cb → RF, XGB, LGBM with label encoding
  • data_cbc → CatBoost with native categorical features

3. Feature Transformation and Selection
- For data_lr:
  • Applied WOE encoding to numeric and categorical variables.
  • Calculated correlations with TARGET and removed low-correlation features.
  • Checked multicollinearity (intercorrelation) and removed highly correlated variables.
- For data_knn:
  • Detected and capped outliers using IQR method.
  • Removed low-correlation and highly correlated variables.
  • Calculated VIF and dropped multicollinear features.
  • Applied one-hot encoding for categorical variables.
- For data_rf_xgb_lgb_cb:
  • Applied label encoding for all features.
- For data_cbc:
  • Identified categorical features for native CatBoost handling.

4. Model Inputs Preparation
- Defined X (features) and y (TARGET) for each pipeline.
- Scaled data for KNN using StandardScaler.
- Split all datasets into train/test sets (80/20).

5. Model Training and Evaluation
- Logistic Regression:
  • Trained full and univariate models.
  • Selected variables based on train/test Gini scores and stability.
- KNN, RF, XGBoost, LGBM, CatBoost:
  • Hyperparameter tuning with Optuna for each model.
- Trained optimized models for each pipeline and collected train/test Gini scores.
- Selected XGBoost as the best-performing model and analyzed feature importance accordingly.
- Evaluated feature importance for XGBoost and selected top features.
- Retrained XGBoost on selected features.

6. Model Interpretation
- SHAP analysis for XGBoost selected features.
- Visualized feature importance and SHAP summary plot.
- Excluded a feature with all zero values from SHAP analysis.

7. Ensemble Modeling
- Voting Classifier:
  • Hard and soft voting using optimized RF, XGB, CatBoost, and Logistic Regression.
- Stacking Classifier:
  • Base models: RF, XGB, CatBoost
  • Meta-model: Logistic Regression
  • Used probability outputs for stacking.

