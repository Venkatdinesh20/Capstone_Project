"""
Analyze target variable to determine problem type
"""
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *

# Load base data
print("=" * 80)
print("TARGET VARIABLE ANALYSIS")
print("=" * 80)

df = pd.read_parquet(PARQUET_TRAIN_DIR / 'train_base.parquet')

print(f"\n### Dataset Info ###")
print(f"Total records: {len(df):,}")
print(f"Columns: {list(df.columns)}")

print(f"\n### Target Variable: '{TARGET_COL}' ###")
print(f"Data type: {df[TARGET_COL].dtype}")
print(f"Unique values: {sorted(df[TARGET_COL].unique())}")
print(f"Number of unique values: {df[TARGET_COL].nunique()}")

print(f"\n### Value Distribution ###")
print(df[TARGET_COL].value_counts().to_string())

print(f"\n### Percentage Distribution ###")
print((df[TARGET_COL].value_counts(normalize=True) * 100).to_string())

print(f"\n### Problem Type Determination ###")
if df[TARGET_COL].nunique() == 2:
    print("✓ BINARY CLASSIFICATION")
    print("  - Target has 2 classes (0 and 1)")
    print("  - 0 = Loan NOT Defaulted (Good Customer)")
    print("  - 1 = Loan Defaulted (Bad Customer - RISK)")
    print("  - Model Type: SUPERVISED LEARNING - BINARY CLASSIFICATION")
    print("  - Algorithms: Logistic Regression, Decision Trees, Random Forest,")
    print("               LightGBM, XGBoost, Neural Networks")
    print("  - Evaluation: AUC-ROC, Precision, Recall, F1-Score, Confusion Matrix")
elif df[TARGET_COL].nunique() < 10:
    print("✓ MULTI-CLASS CLASSIFICATION")
else:
    print("✓ REGRESSION")

print(f"\n### Class Imbalance ###")
class_ratio = df[TARGET_COL].value_counts()[0] / df[TARGET_COL].value_counts()[1]
print(f"Ratio (0:1): {class_ratio:.1f}:1")
if class_ratio > 10:
    print(f"⚠ HIGHLY IMBALANCED! Need to handle:")
    print(f"  - Use SMOTE (Synthetic Minority Over-sampling)")
    print(f"  - Use class_weight='balanced'")
    print(f"  - Use stratified sampling")
    print(f"  - Evaluate with F1-Score, not just Accuracy")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("✓ Problem Type: SUPERVISED BINARY CLASSIFICATION")
print("✓ Goal: Predict loan default risk (0 or 1)")
print("✓ Supervised: We have labeled data (target column)")
print("✓ Classification: Target is categorical (not continuous)")
print("=" * 80)
