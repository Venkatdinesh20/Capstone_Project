"""
STEP 4: FEATURE ENGINEERING
Prepare features for modeling: encoding, scaling, train-test split
Class imbalance handled via model class_weight parameter
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("=" * 80)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 80)

# Load cleaned data from Step 3
print("\n### Loading Cleaned Data from Step 3 ###")
cleaned_df = pd.read_parquet(DATA_PROCESSED_DIR / 'step3_data_cleaned.parquet')
print(f"Input shape: {cleaned_df.shape}")

# Step 4.1: Separate features and target
print("\n### Step 4.1: Separating Features and Target ###")
if ID_COL in cleaned_df.columns:
    X = cleaned_df.drop(columns=[TARGET_COL, ID_COL])
    case_ids = cleaned_df[ID_COL]
else:
    X = cleaned_df.drop(columns=[TARGET_COL])
    case_ids = None

y = cleaned_df[TARGET_COL]

print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")
print(f"Feature columns: {list(X.columns)[:10]}... ({len(X.columns)} total)")

# Step 4.2: Handle categorical features
print("\n### Step 4.2: Encoding Categorical Features ###")
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Categorical columns: {len(categorical_cols)}")

if categorical_cols:
    print(f"Columns: {categorical_cols}")
    
    # Identify date columns and high-cardinality categoricals
    date_cols = [col for col in categorical_cols if 'date' in col.lower() or col in ['birthdate_574D', 'dateofbirth_337D']]
    print(f"\nDropping {len(date_cols)} date columns (too high cardinality): {date_cols}")
    X = X.drop(columns=date_cols)
    
    # Update categorical columns list
    categorical_cols = [col for col in categorical_cols if col not in date_cols]
    print(f"Remaining categorical columns: {len(categorical_cols)}")
    
    if categorical_cols:
        # Check cardinality for remaining categoricals
        high_card_threshold = 100
        high_card_cols = []
        for col in categorical_cols:
            n_unique = X[col].nunique()
            if n_unique > high_card_threshold:
                high_card_cols.append(col)
        
        if high_card_cols:
            print(f"Dropping {len(high_card_cols)} high-cardinality columns (>{high_card_threshold} categories)")
            X = X.drop(columns=high_card_cols)
            categorical_cols = [col for col in categorical_cols if col not in high_card_cols]
        
        if categorical_cols:
            print(f"Applying one-hot encoding to {len(categorical_cols)} columns...")
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            # Note: One-hot encoded columns are already 0/1, don't need scaling
            if hasattr(X, 'sparse'):
                X = X.sparse.to_dense()
            print(f"Shape after encoding: {X.shape}")
        else:
            print("No categorical columns remaining after filtering")
    else:
        print("No categorical columns remaining after dropping dates")
else:
    print("No categorical columns found")

# Step 4.3: Train-Test Split
print("\n### Step 4.3: Train-Test Split (Stratified) ###")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\nTrain target distribution:")
print(y_train.value_counts())
print(f"Test target distribution:")
print(y_test.value_counts())

# Verify stratification
train_ratio = y_train.value_counts()[1] / len(y_train) * 100
test_ratio = y_test.value_counts()[1] / len(y_test) * 100
print(f"\nDefault rate - Train: {train_ratio:.2f}%, Test: {test_ratio:.2f}%")

# Step 4.4: Feature Scaling (Only numerical features)
print("\n### Step 4.4: Feature Scaling (StandardScaler on numerical features) ###")

# Identify numerical columns (exclude one-hot encoded binary columns)
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Filter out binary columns (one-hot encoded have only 0/1 values)
numerical_cols_to_scale = [col for col in numerical_cols if X_train[col].nunique() > 2]

print(f"Scaling {len(numerical_cols_to_scale)} numerical features (excluding binary one-hot encoded)")

if numerical_cols_to_scale:
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Scale only numerical features
    X_train_scaled[numerical_cols_to_scale] = scaler.fit_transform(X_train[numerical_cols_to_scale])
    X_test_scaled[numerical_cols_to_scale] = scaler.transform(X_test[numerical_cols_to_scale])
    
    print(f"✓ Numerical features scaled (mean≈0, std≈1)")
    print(f"Scaled features mean: {X_train_scaled[numerical_cols_to_scale].mean().mean():.6f}")
    print(f"Scaled features std: {X_train_scaled[numerical_cols_to_scale].std().mean():.6f}")
else:
    print("No numerical features to scale")
    X_train_scaled = X_train
    X_test_scaled = X_test
    scaler = None

# Save scaler
if scaler is not None:
    scaler_path = MODELS_DIR / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved: {scaler_path}")
    # Also save numerical columns list
    joblib.dump(numerical_cols_to_scale, MODELS_DIR / 'numerical_cols.pkl')
    print(f"✓ Numerical columns list saved")

# Step 4.5: Class Imbalance Handling
print("\n### Step 4.5: Class Imbalance Strategy ###")
print(f"Dataset statistics:")
print(f"  Train shape: {X_train_scaled.shape}")
print(f"  Class 0: {(y_train==0).sum():,}")
print(f"  Class 1: {(y_train==1).sum():,}")
print(f"  Ratio: {(y_train==0).sum() / (y_train==1).sum():.1f}:1")
print(f"\n⚠️  SMOTE skipped due to memory constraints with 1.3M rows")
print(f"✓ Will use class_weight='balanced' in models instead")
print(f"   This approach scales loss function by class frequency")
print(f"   and is more memory-efficient for large datasets")

# Use original data (no resampling)
X_train_final = X_train_scaled
y_train_final = y_train

# Summary
print("\n" + "=" * 80)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 80)
print(f"Total features: {X_train_final.shape[1]}")
print(f"Train samples: {X_train_final.shape[0]:,}")
print(f"Test samples: {X_test_scaled.shape[0]:,}")
print(f"Features scaled: ✓")
print(f"Class imbalance: Will handle via model class_weight parameter")

# Save processed data
print("\n### Saving Processed Data ###")
output_dir = DATA_PROCESSED_DIR

X_train_final.to_parquet(output_dir / 'step4_X_train.parquet', index=False)
X_test_scaled.to_parquet(output_dir / 'step4_X_test.parquet', index=False)
y_train_final.to_frame().to_parquet(output_dir / 'step4_y_train.parquet', index=False)
y_test.to_frame().to_parquet(output_dir / 'step4_y_test.parquet', index=False)

print(f"✓ X_train: {output_dir / 'step4_X_train.parquet'}")
print(f"✓ X_test: {output_dir / 'step4_X_test.parquet'}")
print(f"✓ y_train: {output_dir / 'step4_y_train.parquet'}")
print(f"✓ y_test: {output_dir / 'step4_y_test.parquet'}")

print("\n" + "=" * 80)
print("✓ STEP 4 COMPLETED")
print("=" * 80)
print(f"✓ Data ready for modeling")
print(f"✓ Features: {X_train_final.shape[1]}")
print(f"✓ Train samples: {X_train_final.shape[0]:,}")
print(f"✓ Test samples: {X_test_scaled.shape[0]:,}")
print("\nNext: Run step5_model_training.py")
print("=" * 80)
