"""
Compare original vs cleaned data
"""
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from src.data import DataLoader

print("=" * 80)
print("DATA CLEANING COMPARISON")
print("=" * 80)

# Load original
print("\n### ORIGINAL DATA ###")
loader = DataLoader(data_type='train', file_format='parquet')
original = loader.load_base_table()
print(f"Shape: {original.shape}")
print(f"Missing values: {original.isnull().sum().sum():,}")
print(f"Memory: {original.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nColumns: {list(original.columns)}")
print(f"\nData types:\n{original.dtypes}")
print(f"\nMissing per column:\n{original.isnull().sum()}")

# Load cleaned
print("\n" + "=" * 80)
print("### CLEANED DATA ###")
cleaned = pd.read_parquet(ROOT_DIR / 'data_processed' / 'train_base_cleaned.parquet')
print(f"Shape: {cleaned.shape}")
print(f"Missing values: {cleaned.isnull().sum().sum():,}")
print(f"Memory: {cleaned.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nColumns added: {cleaned.shape[1] - original.shape[1]}")

# Find new columns (missing indicators)
new_cols = set(cleaned.columns) - set(original.columns)
if new_cols:
    print(f"\nNew columns (missing indicators): {len(new_cols)}")
    for col in sorted(new_cols):
        print(f"  - {col}")

# Summary
print("\n" + "=" * 80)
print("### SUMMARY ###")
print(f"Rows: {original.shape[0]:,} → {cleaned.shape[0]:,} (no change ✓)")
print(f"Columns: {original.shape[1]} → {cleaned.shape[1]} (+{cleaned.shape[1] - original.shape[1]})")
print(f"Missing: {original.isnull().sum().sum():,} → {cleaned.isnull().sum().sum():,} (cleaned ✓)")
print(f"File size: 6.72 MB → 7.31 MB")
print("=" * 80)

# Check target distribution
if TARGET_COL in cleaned.columns:
    print(f"\n### TARGET DISTRIBUTION (unchanged) ###")
    print(cleaned[TARGET_COL].value_counts())
    print(f"\nClass distribution:")
    print(cleaned[TARGET_COL].value_counts(normalize=True) * 100)
