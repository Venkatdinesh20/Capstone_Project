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
    
    def __init__(self, missing_threshold: float = MISSING_THRESHOLD):
        """Initialize DataPreprocessor.
        
        Parameters:
        -----------
        missing_threshold : float
            Threshold for dropping columns with too many missing values (default: 0.80)
        """
        self.missing_threshold = missing_threshold
        self.imputers = {}
        self.scalers = {}
        self.columns_dropped = []
        self.missing_indicators = {}
        logger.info(f"DataPreprocessor initialized with missing_threshold={missing_threshold}")
    
    
    def analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """
        Analyze missing values in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
        
        Returns:
        --------
        dict : Summary with total_missing, missing_percentage, high_missing_cols, etc.
        """
        total_missing = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = (total_missing / total_cells) * 100
        
        missing_by_col = df.isnull().sum()
        high_missing_cols = missing_by_col[missing_by_col / len(df) > self.missing_threshold].index.tolist()
        
        logger.info(f"Total missing values: {total_missing:,} ({missing_pct:.2f}%)")
        logger.info(f"Columns with >{self.missing_threshold*100}% missing: {len(high_missing_cols)}")
        
        return {
            'total_missing': total_missing,
            'missing_percentage': missing_pct,
            'total_cols': df.shape[1],
            'high_missing_cols': high_missing_cols
        }
    
    
    def drop_high_missing_columns(self, df: pd.DataFrame, 
                                  threshold: float = None) -> pd.DataFrame:
        """
        Drop columns with missing values above threshold.
        
        Parameters:
        -----------
        df : pd.DataFrame
        threshold : float
            Drop columns with missing percentage above this (0-1). 
            If None, uses self.missing_threshold
        
        Returns:
        --------
        pd.DataFrame : DataFrame with high-missing columns removed
        """
        if threshold is None:
            threshold = self.missing_threshold
            
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
        Create binary indicator columns for missing values - OPTIMIZED.
        
        Parameters:
        -----------
        df : pd.DataFrame
        
        Returns:
        --------
        pd.DataFrame : DataFrame with missing indicator columns added
        """
        if not CREATE_MISSING_INDICATORS:
            return df
        
        # Find columns with missing values
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        
        # Exclude ID and target columns
        cols_with_missing = [col for col in cols_with_missing 
                            if col not in [ID_COL, TARGET_COL]]
        
        # Limit to columns with moderate missing (not too sparse)
        # Skip if >50% missing (already dropped) or <5% missing (not useful)
        useful_cols = []
        for col in cols_with_missing:
            missing_pct = df[col].isnull().sum() / len(df)
            if 0.05 <= missing_pct <= 0.50:
                useful_cols.append(col)
        
        if not useful_cols:
            logger.info("No useful columns for missing indicators")
            return df
        
        # Create all indicators at once using concat (memory efficient)
        indicator_dfs = []
        for col in useful_cols:
            indicator_col = f"{col}_missing"
            indicator_dfs.append(pd.DataFrame({
                indicator_col: df[col].isnull().astype('int8')  # Use int8 to save memory
            }))
            self.missing_indicators[col] = indicator_col
        
        # Concatenate all at once
        if indicator_dfs:
            indicators_combined = pd.concat(indicator_dfs, axis=1)
            df_with_indicators = pd.concat([df, indicators_combined], axis=1)
            logger.info(f"Created {len(useful_cols)} missing value indicators (5-50% missing range)")
        else:
            df_with_indicators = df
        
        return df_with_indicators
    
    
    def impute_missing_values(self, df: pd.DataFrame, 
                             fit: bool = True) -> pd.DataFrame:
        """
        Impute missing values using specified strategies - MEMORY OPTIMIZED WITH BATCHING.
        
        Parameters:
        -----------
        df : pd.DataFrame
        fit : bool
            Whether to fit imputers (True for train, False for test)
        
        Returns:
        --------
        pd.DataFrame : DataFrame with imputed values
        """
        logger.info("Starting imputation (batch processing to save memory)...")
        
        # Separate numerical and categorical columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove ID and target from imputation
        num_cols = [col for col in num_cols if col not in [ID_COL, TARGET_COL]]
        cat_cols = [col for col in cat_cols if col not in [ID_COL, TARGET_COL]]
        
        # Only impute columns with missing values
        num_cols_with_missing = [col for col in num_cols if df[col].isnull().any()]
        cat_cols_with_missing = [col for col in cat_cols if df[col].isnull().any()]
        
        logger.info(f"Numerical columns to impute: {len(num_cols_with_missing)}")
        logger.info(f"Categorical columns to impute: {len(cat_cols_with_missing)}")
        
        # Impute numerical columns in batches
        if num_cols_with_missing:
            batch_size = 50  # Process 50 columns at a time
            num_batches = (len(num_cols_with_missing) + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(num_cols_with_missing))
                batch_cols = num_cols_with_missing[start_idx:end_idx]
                
                logger.info(f"Processing numerical batch {i+1}/{num_batches} ({len(batch_cols)} columns)")
                
                if fit:
                    imputer = SimpleImputer(strategy=NUMERICAL_IMPUTE_STRATEGY)
                    df[batch_cols] = imputer.fit_transform(df[batch_cols])
                    self.imputers[f'numerical_batch_{i}'] = imputer
                else:
                    if f'numerical_batch_{i}' in self.imputers:
                        df[batch_cols] = self.imputers[f'numerical_batch_{i}'].transform(df[batch_cols])
            
            logger.info(f"✓ Imputed all {len(num_cols_with_missing)} numerical columns")
        
        # Impute categorical columns in batches
        if cat_cols_with_missing:
            batch_size = 20  # Smaller batch for categorical
            num_batches = (len(cat_cols_with_missing) + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(cat_cols_with_missing))
                batch_cols = cat_cols_with_missing[start_idx:end_idx]
                
                logger.info(f"Processing categorical batch {i+1}/{num_batches} ({len(batch_cols)} columns)")
                
                if fit:
                    imputer = SimpleImputer(strategy=CATEGORICAL_IMPUTE_STRATEGY)
                    df[batch_cols] = imputer.fit_transform(df[batch_cols])
                    self.imputers[f'categorical_batch_{i}'] = imputer
                else:
                    if f'categorical_batch_{i}' in self.imputers:
                        df[batch_cols] = self.imputers[f'categorical_batch_{i}'].transform(df[batch_cols])
            
            logger.info(f"✓ Imputed all {len(cat_cols_with_missing)} categorical columns")
        
        logger.info("Imputation complete")
        return df
    
    
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
