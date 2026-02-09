"""
Complete MLOps Pipeline - End-to-End ML Workflow
Covers: Data Collection → Preprocessing → Training → Evaluation → Tuning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import joblib
import json

# ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, f1_score, accuracy_score
)
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from src.data import DataLoader, DataMerger, DataPreprocessor
from src.visualization import plots

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUTS_DIR / 'logs' / 'mlops_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create directories
(OUTPUTS_DIR / 'logs').mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / 'models').mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / 'reports').mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / 'plots').mkdir(parents=True, exist_ok=True)


class MLOpsPipeline:
    """Complete MLOps pipeline for credit risk modeling."""
    
    def __init__(self):
        self.raw_data = None
        self.merged_data = None
        self.cleaned_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.scaler = None
        
    def step_1_data_collection(self):
        """Step 1: Collect and load all raw data."""
        logger.info("=" * 80)
        logger.info("STEP 1: DATA COLLECTION")
        logger.info("=" * 80)
        
        loader = DataLoader(data_type='train', file_format='parquet')
        self.raw_data = loader.load_base_table()
        
        logger.info(f"✓ Loaded base data: {self.raw_data.shape}")
        logger.info(f"  Rows: {len(self.raw_data):,}")
        logger.info(f"  Columns: {len(self.raw_data.columns)}")
        logger.info(f"  Target distribution:\n{self.raw_data[TARGET_COL].value_counts()}")
        logger.info(f"  Class imbalance: {(self.raw_data[TARGET_COL]==0).sum()}/{(self.raw_data[TARGET_COL]==1).sum()} ratio")
        
        return self.raw_data
    
    def step_2_data_merging(self):
        """Step 2: Merge all 32 tables into single dataset."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: DATA MERGING")
        logger.info("=" * 80)
        
        merger = DataMerger(data_type='train', file_format='parquet')
        
        # Start with base table
        self.merged_data = self.raw_data.copy()
        logger.info(f"Starting with base: {self.merged_data.shape}")
        
        # Merge static tables (1:1 joins)
        logger.info("\n### Merging Static Tables (1:1 joins) ###")
        static_tables = [
            'static_cb_0', 'static_0_0', 'static_0_1', 'static_0_2'
        ]
        
        for table in static_tables:
            try:
                logger.info(f"  Merging {table}...")
                self.merged_data = merger.merge_static_table(self.merged_data, table)
                logger.info(f"    Shape after merge: {self.merged_data.shape}")
            except FileNotFoundError:
                logger.warning(f"    {table} not found, skipping")
            except Exception as e:
                logger.error(f"    Error merging {table}: {str(e)}")
        
        # Merge dynamic tables (1:N aggregations)
        logger.info("\n### Merging Dynamic Tables (1:N aggregations) ###")
        dynamic_tables = [
            'applprev_1', 'applprev_2',
            'person_1', 'person_2',
            'credit_bureau_a_1', 'credit_bureau_a_2',
            'credit_bureau_b_1', 'credit_bureau_b_2',
            'other_1', 'tax_registry_a_1', 'tax_registry_b_1', 'tax_registry_c_1',
            'deposit_1', 'debitcard_1'
        ]
        
        for table in dynamic_tables:
            try:
                logger.info(f"  Aggregating {table}...")
                self.merged_data = merger.aggregate_dynamic_table(self.merged_data, table)
                logger.info(f"    Shape after aggregation: {self.merged_data.shape}")
            except FileNotFoundError:
                logger.warning(f"    {table} not found, skipping")
            except Exception as e:
                logger.error(f"    Error aggregating {table}: {str(e)}")
        
        logger.info(f"\n✓ Final merged dataset: {self.merged_data.shape}")
        logger.info(f"  Total features: {self.merged_data.shape[1] - 1}")  # Exclude target
        
        # Save merged data
        merged_path = DATA_PROCESSED_DIR / 'train_merged.parquet'
        self.merged_data.to_parquet(merged_path, index=False)
        logger.info(f"✓ Saved merged data to: {merged_path}")
        
        return self.merged_data
    
    def step_3_data_preprocessing(self):
        """Step 3: Clean and preprocess merged data."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: DATA PREPROCESSING")
        logger.info("=" * 80)
        
        preprocessor = DataPreprocessor(missing_threshold=MISSING_THRESHOLD)
        
        # Analyze missing values
        logger.info("\n### Missing Value Analysis ###")
        missing_report = preprocessor.analyze_missing_values(self.merged_data)
        logger.info(f"Total missing values: {missing_report['total_missing']:,}")
        logger.info(f"Columns with >80% missing: {missing_report['high_missing_cols']}")
        
        # Drop high missing columns
        logger.info("\n### Dropping High Missing Columns ###")
        self.cleaned_data = preprocessor.drop_high_missing_columns(self.merged_data)
        logger.info(f"Shape after dropping: {self.cleaned_data.shape}")
        
        # Create missing indicators
        logger.info("\n### Creating Missing Indicators ###")
        self.cleaned_data = preprocessor.create_missing_indicators(self.cleaned_data)
        logger.info(f"Shape after indicators: {self.cleaned_data.shape}")
        
        # Impute missing values
        logger.info("\n### Imputing Missing Values ###")
        self.cleaned_data = preprocessor.impute_missing_values(self.cleaned_data)
        logger.info(f"Remaining missing: {self.cleaned_data.isnull().sum().sum()}")
        
        # Validate data
        logger.info("\n### Data Validation ###")
        is_valid = preprocessor.validate_data(self.cleaned_data)
        logger.info(f"Data validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
        
        # Save cleaned data
        cleaned_path = DATA_PROCESSED_DIR / 'train_cleaned.parquet'
        self.cleaned_data.to_parquet(cleaned_path, index=False)
        logger.info(f"✓ Saved cleaned data to: {cleaned_path}")
        
        return self.cleaned_data
    
    def step_4_feature_engineering(self):
        """Step 4: Engineer additional features."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: FEATURE ENGINEERING")
        logger.info("=" * 80)
        
        # Separate features and target
        X = self.cleaned_data.drop(columns=[TARGET_COL, CASE_ID_COL])
        y = self.cleaned_data[TARGET_COL]
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Feature columns: {list(X.columns)}")
        
        # Handle categorical columns
        logger.info("\n### Encoding Categorical Features ###")
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        logger.info(f"Categorical columns: {categorical_cols}")
        
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            logger.info(f"Shape after encoding: {X.shape}")
        
        # Split data
        logger.info("\n### Train-Test Split ###")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
        
        logger.info(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        logger.info(f"Train target distribution:\n{self.y_train.value_counts()}")
        logger.info(f"Test target distribution:\n{self.y_test.value_counts()}")
        
        # Scale features
        logger.info("\n### Feature Scaling ###")
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        logger.info("✓ Features scaled")
        
        # Handle class imbalance with SMOTE
        logger.info("\n### Handling Class Imbalance (SMOTE) ###")
        smote = SMOTE(random_state=RANDOM_STATE)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        logger.info(f"After SMOTE: {self.X_train.shape}")
        logger.info(f"New target distribution:\n{self.y_train.value_counts()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def step_5_model_selection_training(self):
        """Step 5: Select and train multiple models."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: MODEL SELECTION & TRAINING")
        logger.info("=" * 80)
        
        # 1. Logistic Regression (Baseline)
        logger.info("\n### Training Logistic Regression (Baseline) ###")
        lr = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
        lr.fit(self.X_train, self.y_train)
        self.models['logistic_regression'] = lr
        logger.info("✓ Logistic Regression trained")
        
        # 2. LightGBM
        logger.info("\n### Training LightGBM ###")
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_model = lgb.train(
            LIGHTGBM_PARAMS,
            lgb_train,
            num_boost_round=100,
            valid_sets=[lgb_train],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)]
        )
        self.models['lightgbm'] = lgb_model
        logger.info("✓ LightGBM trained")
        
        # 3. XGBoost
        logger.info("\n### Training XGBoost ###")
        xgb_model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        xgb_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            early_stopping_rounds=10,
            verbose=10
        )
        self.models['xgboost'] = xgb_model
        logger.info("✓ XGBoost trained")
        
        return self.models
    
    def step_6_model_evaluation(self):
        """Step 6: Evaluate all models."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: MODEL EVALUATION")
        logger.info("=" * 80)
        
        for model_name, model in self.models.items():
            logger.info(f"\n### Evaluating {model_name.upper()} ###")
            
            # Make predictions
            if model_name == 'lightgbm':
                y_pred_proba = model.predict(self.X_test)
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            auc_roc = roc_auc_score(self.y_test, y_pred_proba)
            f1 = f1_score(self.y_test, y_pred)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            logger.info(f"  AUC-ROC: {auc_roc:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            
            # Classification report
            logger.info(f"\n{classification_report(self.y_test, y_pred)}")
            
            # Store results
            self.results[model_name] = {
                'auc_roc': auc_roc,
                'f1_score': f1,
                'accuracy': accuracy,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Save model
            model_path = OUTPUTS_DIR / 'models' / f'{model_name}.pkl'
            joblib.dump(model, model_path)
            logger.info(f"✓ Model saved: {model_path}")
        
        return self.results
    
    def step_7_generate_reports(self):
        """Step 7: Generate visualizations and reports."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: GENERATING REPORTS")
        logger.info("=" * 80)
        
        # Model comparison
        logger.info("\n### Model Comparison ###")
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'AUC-ROC': [r['auc_roc'] for r in self.results.values()],
            'F1-Score': [r['f1_score'] for r in self.results.values()],
            'Accuracy': [r['accuracy'] for r in self.results.values()]
        })
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        # Save comparison
        comparison_path = OUTPUTS_DIR / 'reports' / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"✓ Saved comparison: {comparison_path}")
        
        # Generate plots for best model
        best_model = comparison_df.loc[comparison_df['AUC-ROC'].idxmax(), 'Model']
        logger.info(f"\n### Best Model: {best_model.upper()} ###")
        
        y_pred = self.results[best_model]['y_pred']
        y_pred_proba = self.results[best_model]['y_pred_proba']
        
        # ROC Curve
        plots.plot_roc_curve(self.y_test, y_pred_proba, 
                            save_path=OUTPUTS_DIR / 'plots' / 'roc_curve.png')
        logger.info("✓ ROC curve saved")
        
        # Confusion Matrix
        plots.plot_confusion_matrix(self.y_test, y_pred,
                                   save_path=OUTPUTS_DIR / 'plots' / 'confusion_matrix.png')
        logger.info("✓ Confusion matrix saved")
        
        # Feature importance (if tree-based model)
        if best_model in ['lightgbm', 'xgboost']:
            if best_model == 'lightgbm':
                importance = self.models[best_model].feature_importance()
                feature_names = self.X_train.columns
            else:
                importance = self.models[best_model].feature_importances_
                feature_names = self.X_train.columns
            
            plots.plot_feature_importance(
                importance, feature_names, top_n=20,
                save_path=OUTPUTS_DIR / 'plots' / 'feature_importance.png'
            )
            logger.info("✓ Feature importance saved")
        
        logger.info("\n✓ All reports generated!")
        
        return comparison_df
    
    def run_full_pipeline(self):
        """Execute complete MLOps pipeline."""
        start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE MLOPS PIPELINE")
        logger.info(f"Start time: {start_time}")
        logger.info("=" * 80)
        
        try:
            # Step 1: Data Collection
            self.step_1_data_collection()
            
            # Step 2: Data Merging
            self.step_2_data_merging()
            
            # Step 3: Data Preprocessing
            self.step_3_data_preprocessing()
            
            # Step 4: Feature Engineering
            self.step_4_feature_engineering()
            
            # Step 5: Model Selection & Training
            self.step_5_model_selection_training()
            
            # Step 6: Model Evaluation
            self.step_6_model_evaluation()
            
            # Step 7: Generate Reports
            comparison = self.step_7_generate_reports()
            
            # Summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Duration: {duration}")
            logger.info(f"Best Model: {comparison.loc[comparison['AUC-ROC'].idxmax(), 'Model']}")
            logger.info(f"Best AUC-ROC: {comparison['AUC-ROC'].max():.4f}")
            logger.info("\n✓ All outputs saved to: outputs/")
            
            return comparison
            
        except Exception as e:
            logger.error(f"\n✗ Pipeline failed: {str(e)}")
            raise


if __name__ == '__main__':
    pipeline = MLOpsPipeline()
    results = pipeline.run_full_pipeline()
