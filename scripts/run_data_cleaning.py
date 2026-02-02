"""
Data Cleaning Pipeline - MLOps Workflow
Run this to clean and prepare data for modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from src.data import DataLoader, DataMerger, DataPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUTS_DIR / 'logs' / 'data_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory
(OUTPUTS_DIR / 'logs').mkdir(parents=True, exist_ok=True)


def load_base_data():
    """Step 1: Load base training data."""
    logger.info("=" * 80)
    logger.info("STEP 1: LOADING BASE DATA")
    logger.info("=" * 80)
    
    loader = DataLoader(data_type='train', file_format='parquet')
    base_df = loader.load_base_table()
    
    logger.info(f"✓ Loaded base data: {base_df.shape}")
    logger.info(f"  Columns: {list(base_df.columns)}")
    logger.info(f"  Memory: {base_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return base_df


def analyze_data_quality(df):
    """Step 2: Analyze data quality."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: DATA QUALITY ANALYSIS")
    logger.info("=" * 80)
    
    # Basic info
    logger.info(f"\n### BASIC INFO ###")
    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Columns: {len(df.columns)}")
    logger.info(f"Duplicates: {df.duplicated().sum():,}")
    
    # Data types
    logger.info(f"\n### DATA TYPES ###")
    logger.info(f"Numerical: {len(df.select_dtypes(include=[np.number]).columns)}")
    logger.info(f"Categorical: {len(df.select_dtypes(include=['object', 'category']).columns)}")
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'column': df.columns,
        'missing': missing.values,
        'missing_pct': missing_pct.values
    })
    missing_df = missing_df[missing_df['missing'] > 0].sort_values('missing_pct', ascending=False)
    
    logger.info(f"\n### MISSING VALUES ###")
    logger.info(f"Columns with missing: {len(missing_df)}")
    logger.info(f"Total missing: {missing.sum():,}")
    
    if len(missing_df) > 0:
        logger.info(f"\nTop 10 columns with most missing:")
        for _, row in missing_df.head(10).iterrows():
            logger.info(f"  {row['column']:30s} {row['missing_pct']:6.2f}%")
    
    # Target distribution
    if TARGET_COL in df.columns:
        logger.info(f"\n### TARGET DISTRIBUTION ###")
        target_dist = df[TARGET_COL].value_counts()
        target_pct = df[TARGET_COL].value_counts(normalize=True) * 100
        for val in target_dist.index:
            logger.info(f"  {val}: {target_dist[val]:,} ({target_pct[val]:.2f}%)")
        
        # Imbalance ratio
        if len(target_dist) == 2:
            ratio = target_dist.max() / target_dist.min()
            logger.info(f"  Imbalance ratio: {ratio:.2f}:1")
    
    return missing_df


def clean_data(df):
    """Step 3: Clean the data."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: DATA CLEANING")
    logger.info("=" * 80)
    
    preprocessor = DataPreprocessor()
    
    # Drop high missing columns
    logger.info(f"\n1. Dropping columns with >{MISSING_THRESHOLD*100}% missing...")
    df_cleaned = preprocessor.drop_high_missing_columns(df)
    logger.info(f"   Dropped {len(preprocessor.columns_dropped)} columns")
    
    # Create missing indicators
    logger.info(f"\n2. Creating missing value indicators...")
    df_cleaned = preprocessor.create_missing_indicators(df_cleaned)
    logger.info(f"   Created {len(preprocessor.missing_indicators)} indicator flags")
    
    # Impute missing values
    logger.info(f"\n3. Imputing missing values...")
    df_cleaned = preprocessor.impute_missing_values(df_cleaned, fit=True)
    logger.info(f"   ✓ Imputation complete")
    
    # Validate
    logger.info(f"\n4. Validating cleaned data...")
    preprocessor.validate_data(df_cleaned)
    
    return df_cleaned, preprocessor


def save_cleaned_data(df, suffix='cleaned'):
    """Step 4: Save cleaned data."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: SAVING CLEANED DATA")
    logger.info("=" * 80)
    
    # Create processed data directory
    processed_dir = ROOT_DIR / 'data_processed'
    processed_dir.mkdir(exist_ok=True)
    
    # Save as parquet
    output_path = processed_dir / f'train_base_{suffix}.parquet'
    df.to_parquet(output_path, index=False)
    logger.info(f"✓ Saved to: {output_path}")
    logger.info(f"  Final shape: {df.shape}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024**2:.2f} MB")
    
    return output_path


def generate_cleaning_report(original_df, cleaned_df, missing_df):
    """Generate cleaning summary report."""
    logger.info("\n" + "=" * 80)
    logger.info("DATA CLEANING SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"\n### BEFORE vs AFTER ###")
    logger.info(f"Rows:          {len(original_df):,} → {len(cleaned_df):,}")
    logger.info(f"Columns:       {len(original_df.columns)} → {len(cleaned_df.columns)}")
    logger.info(f"Missing total: {original_df.isnull().sum().sum():,} → {cleaned_df.isnull().sum().sum():,}")
    
    logger.info(f"\n### CHANGES ###")
    cols_dropped = len(original_df.columns) - len(cleaned_df.columns)
    cols_added = len(cleaned_df.columns) - len(original_df.columns) + cols_dropped
    logger.info(f"Columns dropped: {cols_dropped}")
    logger.info(f"Columns added (indicators): {cols_added}")
    
    logger.info(f"\n### NEXT STEPS ###")
    logger.info(f"1. ✓ Data loaded and cleaned")
    logger.info(f"2. → Next: Merge other tables (run merger.py)")
    logger.info(f"3. → Next: Feature engineering")
    logger.info(f"4. → Next: Model training")


def main():
    """Main data cleaning pipeline."""
    start_time = datetime.now()
    
    logger.info("=" * 80)
    logger.info("DATA CLEANING PIPELINE - STARTED")
    logger.info(f"Time: {start_time}")
    logger.info("=" * 80)
    
    try:
        # Step 1: Load data
        df = load_base_data()
        
        # Step 2: Analyze
        missing_df = analyze_data_quality(df)
        
        # Step 3: Clean
        df_cleaned, preprocessor = clean_data(df)
        
        # Step 4: Save
        output_path = save_cleaned_data(df_cleaned)
        
        # Step 5: Report
        generate_cleaning_report(df, df_cleaned, missing_df)
        
        # Success
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ DATA CLEANING COMPLETED SUCCESSFULLY!")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 80)
        
        return df_cleaned, preprocessor
    
    except Exception as e:
        logger.error(f"\n❌ ERROR: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    df_cleaned, preprocessor = main()
