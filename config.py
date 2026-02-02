"""
Configuration file for Credit Risk Prediction System
Contains all global settings, paths, and hyperparameters
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Root directory
ROOT_DIR = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = ROOT_DIR / "csv_files"
PARQUET_DIR = ROOT_DIR / "parquet_files"
TRAIN_DATA_DIR = DATA_DIR / "train"
TEST_DATA_DIR = DATA_DIR / "test"
PARQUET_TRAIN_DIR = PARQUET_DIR / "train"
PARQUET_TEST_DIR = PARQUET_DIR / "test"

# Output directories
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# Create directories if they don't exist
for directory in [MODELS_DIR, FIGURES_DIR, REPORTS_DIR, PREDICTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA SETTINGS
# ============================================================================

# File format preference
PREFERRED_FORMAT = 'parquet'  # 'parquet' or 'csv'

# Column names
TARGET_COL = 'target'
ID_COL = 'case_id'
DATE_COLS = ['date_decision']

# Data quality thresholds
MISSING_THRESHOLD = 0.80  # Drop columns with >80% missing values
HIGH_CARDINALITY_THRESHOLD = 100  # Threshold for target encoding

# ============================================================================
# PREPROCESSING SETTINGS
# ============================================================================

# Missing value imputation
NUMERICAL_IMPUTE_STRATEGY = 'median'  # 'mean', 'median', 'most_frequent'
CATEGORICAL_IMPUTE_STRATEGY = 'most_frequent'
CREATE_MISSING_INDICATORS = True

# Feature engineering
AGGREGATION_FUNCTIONS = {
    'numerical': ['mean', 'median', 'std', 'min', 'max', 'sum', 'count'],
    'categorical': ['mode', 'count', 'nunique']
}

# Scaling
SCALING_METHOD = 'standard'  # 'standard', 'minmax', 'robust'

# Encoding
ENCODING_METHODS = {
    'binary': 'label',
    'ordinal': 'ordinal',
    'low_cardinality': 'onehot',
    'high_cardinality': 'target'
}

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Random seed for reproducibility
RANDOM_STATE = 42

# Train/validation/test split
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Cross-validation
CV_FOLDS = 5
CV_STRATEGY = 'stratified'

# Class imbalance handling
IMBALANCE_METHOD = 'smote'  # 'smote', 'class_weight', 'threshold'
SMOTE_SAMPLING_STRATEGY = 0.5  # Ratio of minority to majority after resampling

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Logistic Regression
LOGISTIC_PARAMS = {
    'C': 1.0,
    'penalty': 'l2',
    'solver': 'lbfgs',
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'
}

# LightGBM
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 7,
    'num_leaves': 50,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1
}

# XGBoost
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 7,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': 0
}

# CatBoost
CATBOOST_PARAMS = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 7,
    'l2_leaf_reg': 3,
    'subsample': 0.8,
    'random_state': RANDOM_STATE,
    'verbose': 0,
    'thread_count': -1
}

# Early stopping
EARLY_STOPPING_ROUNDS = 50

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Metrics to calculate
METRICS = [
    'roc_auc',
    'precision',
    'recall',
    'f1',
    'accuracy',
    'average_precision'
]

# Classification threshold
DEFAULT_THRESHOLD = 0.5

# SHAP settings
SHAP_SAMPLE_SIZE = 1000  # Number of samples for SHAP calculation
SHAP_MAX_DISPLAY = 20  # Number of features to show in plots

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
DEFAULT_FIGSIZE = (12, 8)

# Color palette
COLOR_PALETTE = 'Set2'

# Plot settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

# Logging configuration
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = OUTPUTS_DIR / 'logs' / 'pipeline.log'

# Create log directory
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TABLE DEFINITIONS
# ============================================================================

# Static tables (1:1 relationship with base)
STATIC_TABLES = [
    'static_0_0',
    'static_0_1',
    'static_0_2',
    'static_cb_0',
    'person_1',
    'person_2'
]

# Dynamic tables (1:N relationship - require aggregation)
DYNAMIC_TABLES = [
    'applprev_1_0',
    'applprev_1_1',
    'applprev_1_2',
    'applprev_2',
    'credit_bureau_a_1_0',
    'credit_bureau_a_1_1',
    'credit_bureau_a_1_2',
    'credit_bureau_a_1_3',
    'credit_bureau_a_1_4',
    'credit_bureau_a_2_0',
    'credit_bureau_a_2_1',
    'credit_bureau_a_2_2',
    'credit_bureau_a_2_3',
    'credit_bureau_a_2_4',
    'credit_bureau_a_2_5',
    'credit_bureau_a_2_6',
    'credit_bureau_a_2_7',
    'credit_bureau_a_2_8',
    'credit_bureau_a_2_9',
    'credit_bureau_a_2_10',
    'credit_bureau_a_2_11',
    'credit_bureau_b_1',
    'credit_bureau_b_2',
    'debitcard_1',
    'deposit_1',
    'other_1',
    'tax_registry_a_1',
    'tax_registry_b_1',
    'tax_registry_c_1'
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_table_path(table_name, data_type='train', file_format='parquet'):
    """
    Get the full path for a specific table.
    
    Parameters:
    -----------
    table_name : str
        Name of the table (e.g., 'base', 'static_0_0')
    data_type : str
        Either 'train' or 'test'
    file_format : str
        Either 'parquet' or 'csv'
    
    Returns:
    --------
    Path object for the table file
    """
    if file_format == 'parquet':
        base_dir = PARQUET_TRAIN_DIR if data_type == 'train' else PARQUET_TEST_DIR
    else:
        base_dir = TRAIN_DATA_DIR if data_type == 'train' else TEST_DATA_DIR
    
    filename = f"{data_type}_{table_name}.{file_format}"
    return base_dir / filename


def get_model_path(model_name, version='latest'):
    """
    Get the path for saving/loading a model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model (e.g., 'lightgbm', 'xgboost')
    version : str
        Version identifier
    
    Returns:
    --------
    Path object for the model file
    """
    return MODELS_DIR / f"{model_name}_{version}.pkl"


def print_config():
    """Print current configuration settings."""
    print("=" * 80)
    print("CREDIT RISK PREDICTION SYSTEM - CONFIGURATION")
    print("=" * 80)
    print(f"\nRoot Directory: {ROOT_DIR}")
    print(f"Data Format: {PREFERRED_FORMAT}")
    print(f"Random State: {RANDOM_STATE}")
    print(f"Train/Val/Test Split: {TRAIN_SIZE}/{VAL_SIZE}/{TEST_SIZE}")
    print(f"CV Folds: {CV_FOLDS}")
    print(f"Imbalance Method: {IMBALANCE_METHOD}")
    print(f"Missing Value Threshold: {MISSING_THRESHOLD}")
    print("=" * 80)


if __name__ == "__main__":
    print_config()
