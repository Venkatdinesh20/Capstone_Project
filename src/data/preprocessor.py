"""
Data preprocessing utilities for Credit Risk Prediction System
Handles missing values, outliers, and data validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess and clean data."""
    
    def __init__(self):
        """Initialize DataPreprocessor."""
        self.imputers = {}
        self.scalers = {}
        self.columns_dropped = []
        self.missing_indicators = {}
        logger.info("DataPreprocessor initialized")
    
    
    def analyze_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze missing values in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
        
        Returns:
        --------
        pd.DataFrame : Summary of missing values
        """
        missing_summary = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isnull().sum().values,
            'missing_pct': (df.isnull().sum() / len(df) * 100).values,
            'dtype': df.dtypes.values
        })
        
        missing_summary = missing_summary[missing_summary['missing_count'] > 0]
        missing_summary = missing_summary.sort_values('missing_pct', ascending=False)
        
        logger.info(f"Found {len(missing_summary)} columns with missing values")
        logger.info(f"Total missing values: {missing_summary['missing_count'].sum():,}")
        
        return missing_summary
    
    
    def drop_high_missing_columns(self, df: pd.DataFrame, 
                                  threshold: float = MISSING_THRESHOLD) -> pd.DataFrame:
        """
        Drop columns with missing values above threshold.
        
        Parameters:
        -----------
        df : pd.DataFrame
        threshold : float
            Drop columns with missing percentage above this (0-1)
        
        Returns:
        --------
        pd.DataFrame : DataFrame with high-missing columns removed
        """
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        self.columns_dropped.extend(cols_to_drop)
        
        df_cleaned = df.drop(columns=cols_to_drop)
        
        logger.info(f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing")
        if cols_to_drop:
            logger.info(f"Dropped columns: {cols_to_drop[:5]}{'...' if len(cols_to_drop) > 5 else ''}")
        
        return df_cleaned
    
    
    def create_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary indicator columns for missing values.
        
        Parameters:
        -----------
        df : pd.DataFrame
        
        Returns:
        --------
        pd.DataFrame : DataFrame with missing indicator columns added
        """
        if not CREATE_MISSING_INDICATORS:
            return df
        
        df_with_indicators = df.copy()
        
        # Find columns with missing values
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        
        # Exclude ID and target columns
        cols_with_missing = [col for col in cols_with_missing 
                            if col not in [ID_COL, TARGET_COL]]
        
        # Create indicators
        for col in cols_with_missing:
            indicator_col = f"{col}_missing"
            df_with_indicators[indicator_col] = df[col].isnull().astype(int)
            self.missing_indicators[col] = indicator_col
        
        logger.info(f"Created {len(cols_with_missing)} missing value indicators")
        
        return df_with_indicators
    
    
    def impute_missing_values(self, df: pd.DataFrame, 
                             fit: bool = True) -> pd.DataFrame:
        """
        Impute missing values using specified strategies.
        
        Parameters:
        -----------
        df : pd.DataFrame
        fit : bool
            Whether to fit imputers (True for train, False for test)
        
        Returns:
        --------
        pd.DataFrame : DataFrame with imputed values
        """
        df_imputed = df.copy()
        
        # Separate numerical and categorical columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove ID and target from imputation
        num_cols = [col for col in num_cols if col not in [ID_COL, TARGET_COL]]
        cat_cols = [col for col in cat_cols if col not in [ID_COL, TARGET_COL]]
        
        # Impute numerical columns
        if num_cols:
            if fit:
                self.imputers['numerical'] = SimpleImputer(
                    strategy=NUMERICAL_IMPUTE_STRATEGY
                )
                df_imputed[num_cols] = self.imputers['numerical'].fit_transform(
                    df_imputed[num_cols]
                )
                logger.info(f"Fitted and imputed {len(num_cols)} numerical columns")
            else:
                if 'numerical' in self.imputers:
                    df_imputed[num_cols] = self.imputers['numerical'].transform(
                        df_imputed[num_cols]
                    )
                    logger.info(f"Imputed {len(num_cols)} numerical columns")
        
        # Impute categorical columns
        if cat_cols:
            if fit:
                self.imputers['categorical'] = SimpleImputer(
                    strategy=CATEGORICAL_IMPUTE_STRATEGY
                )
                df_imputed[cat_cols] = self.imputers['categorical'].fit_transform(
                    df_imputed[cat_cols]
                )
                logger.info(f"Fitted and imputed {len(cat_cols)} categorical columns")
            else:
                if 'categorical' in self.imputers:
                    df_imputed[cat_cols] = self.imputers['categorical'].transform(
                        df_imputed[cat_cols]
                    )
                    logger.info(f"Imputed {len(cat_cols)} categorical columns")
        
        return df_imputed
    
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                       threshold: float = 3.0) -> Dict[str, pd.Series]:
        """
        Detect outliers in numerical columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
        method : str
            'iqr' or 'zscore'
        threshold : float
            Threshold for outlier detection
        
        Returns:
        --------
        dict : {column: boolean series indicating outliers}
        """
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [col for col in num_cols if col not in [ID_COL, TARGET_COL]]
        
        outliers = {}
        
        for col in num_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers[col] = z_scores > threshold
        
        total_outliers = sum(mask.sum() for mask in outliers.values())
        logger.info(f"Detected {total_outliers:,} outliers across {len(outliers)} columns")
        
        return outliers
    
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True,
                      method: str = SCALING_METHOD) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Parameters:
        -----------
        df : pd.DataFrame
        fit : bool
            Whether to fit scaler (True for train, False for test)
        method : str
            'standard', 'minmax', or 'robust'
        
        Returns:
        --------
        pd.DataFrame : DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        # Get numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [col for col in num_cols if col not in [ID_COL, TARGET_COL]]
        
        if not num_cols:
            logger.warning("No numerical columns to scale")
            return df_scaled
        
        # Select scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit or transform
        if fit:
            self.scalers[method] = scaler
            df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
            logger.info(f"Fitted and scaled {len(num_cols)} columns using {method} scaler")
        else:
            if method in self.scalers:
                df_scaled[num_cols] = self.scalers[method].transform(df_scaled[num_cols])
                logger.info(f"Scaled {len(num_cols)} columns using {method} scaler")
        
        return df_scaled
    
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the dataset for common issues.
        
        Parameters:
        -----------
        df : pd.DataFrame
        
        Returns:
        --------
        bool : True if validation passes
        """
        logger.info("Validating dataset...")
        
        issues = []
        
        # Check for duplicate IDs
        if ID_COL in df.columns:
            if df[ID_COL].duplicated().any():
                issues.append(f"Duplicate {ID_COL} found")
        
        # Check for infinite values
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if np.isinf(df[col]).any():
                issues.append(f"Infinite values in {col}")
        
        # Check target variable (if exists)
        if TARGET_COL in df.columns:
            if df[TARGET_COL].isnull().any():
                issues.append(f"Missing values in {TARGET_COL}")
            
            unique_targets = df[TARGET_COL].unique()
            if not set(unique_targets).issubset({0, 1}):
                issues.append(f"Invalid target values: {unique_targets}")
        
        # Report results
        if issues:
            logger.error("Validation FAILED:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        else:
            logger.info("✓ Validation PASSED")
            return True
    
    
    def preprocess_pipeline(self, df: pd.DataFrame, 
                          fit: bool = True) -> pd.DataFrame:
        """
        Full preprocessing pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
        fit : bool
            Whether to fit transformers (True for train, False for test)
        
        Returns:
        --------
        pd.DataFrame : Preprocessed DataFrame
        """
        logger.info("=" * 80)
        logger.info(f"PREPROCESSING PIPELINE ({'TRAIN' if fit else 'TEST'} MODE)")
        logger.info("=" * 80)
        
        df_processed = df.copy()
        
        # Step 1: Drop high-missing columns (only on train)
        if fit:
            df_processed = self.drop_high_missing_columns(df_processed)
        else:
            # Drop same columns as train
            df_processed = df_processed.drop(columns=self.columns_dropped, errors='ignore')
        
        # Step 2: Create missing indicators
        df_processed = self.create_missing_indicators(df_processed)
        
        # Step 3: Impute missing values
        df_processed = self.impute_missing_values(df_processed, fit=fit)
        
        # Step 4: Validate
        self.validate_data(df_processed)
        
        logger.info("=" * 80)
        logger.info("PREPROCESSING COMPLETE")
        logger.info(f"Shape: {df.shape} → {df_processed.shape}")
        logger.info("=" * 80)
        
        return df_processed


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("DATA PREPROCESSOR - EXAMPLE USAGE")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'case_id': range(1000),
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': ['A', 'B', 'C'] * 333 + ['A'],
        'target': np.random.choice([0, 1], 1000)
    })
    
    # Add missing values
    df.loc[np.random.choice(1000, 100), 'feature1'] = np.nan
    df.loc[np.random.choice(1000, 50), 'feature3'] = np.nan
    
    print(f"\nOriginal shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess_pipeline(df, fit=True)
    
    print(f"\nProcessed shape: {df_processed.shape}")
    print(f"Missing values after processing:\n{df_processed.isnull().sum().sum()}")
