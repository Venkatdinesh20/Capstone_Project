"""
STEP 1: DATA COLLECTION
Load all raw parquet files and understand the data structure
"""
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from src.data import DataLoader

print("=" * 80)
print("STEP 1: DATA COLLECTION")
print("=" * 80)

# Initialize loader
loader = DataLoader(data_type='train', file_format='parquet')

# List all available tables
print("\n### Available Training Files ###")
available_tables = loader.list_available_tables()
print(f"Total files: {len(available_tables)}")
for i, table in enumerate(available_tables, 1):
    print(f"{i:2}. {table}")

# Load base table
print("\n### Loading Base Table ###")
base_df = loader.load_base_table()
print(f"Shape: {base_df.shape}")
print(f"Columns: {list(base_df.columns)}")
print(f"Memory: {base_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\n### Sample Data ###")
print(base_df.head())

print(f"\n### Data Types ###")
print(base_df.dtypes)

print(f"\n### Target Distribution ###")
print(base_df[TARGET_COL].value_counts())
print(f"\nClass Imbalance: {base_df[TARGET_COL].value_counts()[0] / base_df[TARGET_COL].value_counts()[1]:.1f}:1")

print(f"\n### Missing Values ###")
print(base_df.isnull().sum())

# Save base data
output_path = DATA_PROCESSED_DIR / 'step1_base_collected.parquet'
base_df.to_parquet(output_path, index=False)

print("\n" + "=" * 80)
print("✓ STEP 1 COMPLETED")
print("=" * 80)
print(f"✓ Collected: {len(base_df):,} records")
print(f"✓ Features: {len(base_df.columns)} columns")
print(f"✓ Saved to: {output_path}")
print("\nNext: Run step2_data_merging.py")
print("=" * 80)
