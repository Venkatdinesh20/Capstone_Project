"""
STEP 3: DATA PREPROCESSING
Clean merged data: handle missing values, outliers, duplicates
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from src.data import DataPreprocessor

print("=" * 80)
print("STEP 3: DATA PREPROCESSING")
print("=" * 80)

# Load merged data from Step 2
print("\n### Loading Merged Data from Step 2 ###")
merged_df = pd.read_parquet(DATA_PROCESSED_DIR / 'step2_data_merged.parquet')
print(f"Input shape: {merged_df.shape}")
print(f"Memory: {merged_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Initialize preprocessor
preprocessor = DataPreprocessor(missing_threshold=MISSING_THRESHOLD)

# Step 3.1: Analyze missing values
print("\n### Step 3.1: Missing Value Analysis ###")
missing_report = preprocessor.analyze_missing_values(merged_df)
print(f"Total missing: {missing_report['total_missing']:,}")
print(f"Missing percentage: {missing_report['missing_percentage']:.2f}%")
print(f"Columns analyzed: {missing_report['total_cols']}")
print(f"Columns with >80% missing: {len(missing_report['high_missing_cols'])}")

if missing_report['high_missing_cols']:
    print(f"\nColumns to drop (>80% missing):")
    for col in missing_report['high_missing_cols'][:10]:  # Show first 10
        pct = (merged_df[col].isnull().sum() / len(merged_df)) * 100
        print(f"  - {col}: {pct:.1f}%")
    if len(missing_report['high_missing_cols']) > 10:
        print(f"  ... and {len(missing_report['high_missing_cols']) - 10} more")

# Step 3.2: Drop high missing columns
print("\n### Step 3.2: Dropping High Missing Columns ###")
cleaned_df = preprocessor.drop_high_missing_columns(merged_df)
print(f"Before: {merged_df.shape[1]} columns")
print(f"After: {cleaned_df.shape[1]} columns")
print(f"Dropped: {merged_df.shape[1] - cleaned_df.shape[1]} columns")

# Step 3.3: Check duplicates
print("\n### Step 3.3: Duplicate Check ###")
if ID_COL in cleaned_df.columns:
    duplicate_ids = cleaned_df[ID_COL].duplicated().sum()
    print(f"Duplicate case_ids: {duplicate_ids:,}")
    if duplicate_ids > 0:
        print("Removing duplicate case_ids...")
        cleaned_df = cleaned_df.drop_duplicates(subset=[ID_COL], keep='first')
        print(f"After removal: {cleaned_df.shape}")
    else:
        print("✓ No duplicates found")
else:
    print("No case_id column found, skipping duplicate check")

# Step 3.4: Create missing indicators
print("\n### Step 3.4: Creating Missing Indicators ###")
cols_before = cleaned_df.shape[1]
cleaned_df = preprocessor.create_missing_indicators(cleaned_df)
cols_after = cleaned_df.shape[1]
print(f"Added {cols_after - cols_before} missing indicator columns")

# Step 3.5: Impute missing values
print("\n### Step 3.5: Imputing Missing Values (Pandas fillna) ###")
print(f"Missing before imputation: {cleaned_df.isnull().sum().sum():,}")

# Use fillna without inplace (returns new DataFrame)
print("Imputing numerical columns with median...")
num_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [col for col in num_cols if col not in [ID_COL, TARGET_COL]]

for col in num_cols:
    if cleaned_df[col].isnull().any():
        median_val = cleaned_df[col].median()
        cleaned_df[col] = cleaned_df[col].fillna(median_val)

print("Imputing categorical columns with mode...")
cat_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols = [col for col in cat_cols if col not in [ID_COL, TARGET_COL]]

for col in cat_cols:
    if cleaned_df[col].isnull().any():
        mode_val = cleaned_df[col].mode()[0] if len(cleaned_df[col].mode()) > 0 else 'MISSING'
        cleaned_df[col] = cleaned_df[col].fillna(mode_val)

print(f"Missing after imputation: {cleaned_df.isnull().sum().sum():,}")

# Step 3.6: Validate data
print("\n### Step 3.6: Data Validation ###")
is_valid = preprocessor.validate_data(cleaned_df)
if is_valid:
    print("✓ Data validation PASSED")
    print(f"  - No missing values")
    print(f"  - Target column present")
    print(f"  - No duplicate case_ids")
else:
    print("✗ Data validation FAILED")

# Summary
print("\n" + "=" * 80)
print("PREPROCESSING SUMMARY")
print("=" * 80)
print(f"Input shape: {merged_df.shape}")
print(f"Output shape: {cleaned_df.shape}")
print(f"Rows unchanged: {cleaned_df.shape[0] == merged_df.shape[0]}")
print(f"Columns: {merged_df.shape[1]} → {cleaned_df.shape[1]}")
print(f"Missing values: {merged_df.isnull().sum().sum():,} → {cleaned_df.isnull().sum().sum():,}")
print(f"Memory: {cleaned_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Data types summary
print(f"\n### Data Types ###")
print(cleaned_df.dtypes.value_counts())

# Save cleaned data
output_path = DATA_PROCESSED_DIR / 'step3_data_cleaned.parquet'
cleaned_df.to_parquet(output_path, index=False)

print("\n" + "=" * 80)
print("✓ STEP 3 COMPLETED")
print("=" * 80)
print(f"✓ Cleaned dataset: {cleaned_df.shape}")
print(f"✓ Ready for feature engineering")
print(f"✓ Saved to: {output_path}")
print("\nNext: Run step4_feature_engineering.py")
print("=" * 80)
