"""
STEP 2: DATA MERGING
Merge all 32 parquet files into single dataset
Static tables: 1:1 joins
Dynamic tables: 1:N aggregations
"""
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from src.data import DataLoader, DataMerger

print("=" * 80)
print("STEP 2: DATA MERGING")
print("=" * 80)

# Load base data from Step 1
print("\n### Loading Base Data from Step 1 ###")
base_df = pd.read_parquet(DATA_PROCESSED_DIR / 'step1_base_collected.parquet')
print(f"Starting shape: {base_df.shape}")

# Initialize merger
merger = DataMerger(data_type='train', file_format='parquet')
merged_df = base_df.copy()

# Merge static tables (1:1 joins)
print("\n### Merging Static Tables (1:1 joins) ###")
static_tables = ['static_cb_0', 'static_0_0', 'static_0_1', 'static_0_2']

for table in static_tables:
    try:
        print(f"\n  Merging {table}...")
        before_shape = merged_df.shape
        merged_df = merger.merge_static_table(table, merged_df)
        after_shape = merged_df.shape
        print(f"    Before: {before_shape} → After: {after_shape}")
        print(f"    Added {after_shape[1] - before_shape[1]} columns")
    except FileNotFoundError:
        print(f"    ⚠ {table} not found, skipping")
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")

# Merge dynamic tables (1:N aggregations)
print("\n### Merging Dynamic Tables (1:N aggregations) ###")
dynamic_tables = [
    'applprev_1', 'applprev_2',
    'person_1', 'person_2',
    'credit_bureau_a_1', 'credit_bureau_a_2',
    'credit_bureau_b_1', 'credit_bureau_b_2',
    'other_1', 
    'tax_registry_a_1', 'tax_registry_b_1', 'tax_registry_c_1',
    'deposit_1', 'debitcard_1'
]

for table in dynamic_tables:
    try:
        print(f"\n  Aggregating {table}...")
        before_shape = merged_df.shape
        agg_df = merger.aggregate_dynamic_table(table)
        if agg_df is not None:
            merged_df = merged_df.merge(agg_df, on=CASE_ID_COL, how='left')
            after_shape = merged_df.shape
            print(f"    Before: {before_shape} → After: {after_shape}")
            print(f"    Added {after_shape[1] - before_shape[1]} aggregated columns")
        else:
            print(f"    ⚠ {table} returned None")
    except FileNotFoundError:
        print(f"    ⚠ {table} not found, skipping")
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")

# Summary
print("\n" + "=" * 80)
print("MERGE SUMMARY")
print("=" * 80)
print(f"Starting columns: {base_df.shape[1]}")
print(f"Final columns: {merged_df.shape[1]}")
print(f"Features added: {merged_df.shape[1] - base_df.shape[1]}")
print(f"Final shape: {merged_df.shape}")
print(f"Memory: {merged_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Check missing values
print(f"\n### Missing Values Summary ###")
missing_pct = (merged_df.isnull().sum() / len(merged_df) * 100).sort_values(ascending=False)
high_missing = missing_pct[missing_pct > 80]
print(f"Columns with >80% missing: {len(high_missing)}")
print(f"Total missing values: {merged_df.isnull().sum().sum():,}")

# Save merged data
output_path = DATA_PROCESSED_DIR / 'step2_data_merged.parquet'
merged_df.to_parquet(output_path, index=False)

print("\n" + "=" * 80)
print("✓ STEP 2 COMPLETED")
print("=" * 80)
print(f"✓ Merged dataset: {merged_df.shape}")
print(f"✓ Total features: {merged_df.shape[1] - 1} (excluding target)")
print(f"✓ Saved to: {output_path}")
print("\nNext: Run step3_data_preprocessing.py")
print("=" * 80)
