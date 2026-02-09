"""
MLOPS WORKFLOW - COMPLETE PIPELINE GUIDE
=========================================

This project implements a SUPERVISED BINARY CLASSIFICATION model for credit risk prediction.

TARGET: 'target' column
- 0 = Loan NOT Defaulted (Good Customer) - 96.86%
- 1 = Loan Defaulted (Bad Customer - HIGH RISK) - 3.14%

PROBLEM TYPE: Supervised Learning - Binary Classification
- Supervised: We have labeled training data with known outcomes
- Binary: Only 2 classes (0 or 1)
- Imbalanced: 31:1 ratio (needs SMOTE to balance)

WHY SUPERVISED?
- We have historical data with known loan outcomes
- We want to predict future loan defaults
- Model learns from past patterns

WHY CLASSIFICATION (not Regression)?
- Target is categorical (default or not), not continuous
- We predict discrete classes, not numeric values
- Output is probability of default

EVALUATION METRICS:
- AUC-ROC: Area Under ROC Curve (higher = better)
- F1-Score: Balance of Precision and Recall
- Precision: Of predicted defaults, how many are correct?
- Recall: Of actual defaults, how many did we catch?
- Confusion Matrix: True/False Positives/Negatives

=========================================
STEP-BY-STEP EXECUTION
=========================================

Run each script in order:

STEP 1: DATA COLLECTION
------------------------
Load all 32 parquet files
$ python scripts/step1_data_collection.py

Output: data_processed/step1_base_collected.parquet


STEP 2: DATA MERGING  
--------------------
Merge all tables into single dataset
- Static tables: 1:1 joins (person, credit bureau, tax)
- Dynamic tables: 1:N aggregations (applprev, deposits)

$ python scripts/step2_data_merging.py

Output: data_processed/step2_data_merged.parquet


STEP 3: DATA PREPROCESSING
--------------------------
Clean merged data
- Drop columns with >80% missing values
- Create missing indicators
- Impute missing values (median/mode)
- Remove duplicates

$ python scripts/step3_data_preprocessing.py

Output: data_processed/step3_data_cleaned.parquet


STEP 4: FEATURE ENGINEERING
---------------------------
Prepare features for modeling
- Encode categorical variables (one-hot)
- Train-test split (80/20, stratified)
- Scale features (StandardScaler)
- Balance classes with SMOTE

$ python scripts/step4_feature_engineering.py

Output: 
- data_processed/step4_X_train.parquet
- data_processed/step4_X_test.parquet
- data_processed/step4_y_train.parquet
- data_processed/step4_y_test.parquet


STEP 5: MODEL TRAINING
----------------------
Train multiple classification models
1. Logistic Regression (baseline)
2. LightGBM (gradient boosting)
3. XGBoost (extreme gradient boosting)

$ python scripts/step5_model_training.py

Output: 
- models/logistic_regression.pkl
- models/lightgbm.pkl
- models/xgboost.pkl


STEP 6: MODEL EVALUATION
------------------------
Evaluate all models on test set
- Calculate AUC-ROC, F1, Precision, Recall
- Generate confusion matrices
- Compare models
- Identify best performer

$ python scripts/step6_model_evaluation.py

Output:
- outputs/reports/step6_model_comparison.csv
- outputs/reports/step6_evaluation_results.json
- outputs/reports/step6_best_model.txt


STEP 7: VISUALIZATION (OPTIONAL)
--------------------------------
Generate plots and charts
- ROC curves
- Feature importance
- Confusion matrix heatmaps

$ python scripts/step7_visualization.py

Output: outputs/plots/*.png


=========================================
QUICK START - RUN ALL STEPS
=========================================

To run all steps automatically:

$ python scripts/run_all_steps.py

This will execute steps 1-6 in sequence.

=========================================
EXPECTED RESULTS
=========================================

Based on the dataset characteristics:
- Expected AUC-ROC: 0.70 - 0.80
- Expected F1-Score: 0.60 - 0.75
- Best Model: Likely LightGBM or XGBoost

=========================================
TROUBLESHOOTING
=========================================

Error: "File not found"
→ Check that parquet_files/train/ contains all data files

Error: "Missing optional dependency"
→ Run: pip install -r requirements.txt

Error: "Memory error"
→ Reduce batch size or use fewer features

Error: "Convergence warning"
→ Increase max_iter in model parameters

=========================================
"""

print(__doc__)
