"""
STEP 6: MODEL EVALUATION
Evaluate all trained models: AUC-ROC, F1-Score, Precision, Recall, Confusion Matrix
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *

from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import joblib
import json

print("=" * 80)
print("STEP 6: MODEL EVALUATION")
print("=" * 80)

# Load test data
print("\n### Loading Test Data ###")
X_test = pd.read_parquet(DATA_PROCESSED_DIR / 'step4_X_test.parquet')
y_test = pd.read_parquet(DATA_PROCESSED_DIR / 'step4_y_test.parquet')[TARGET_COL]
print(f"Test samples: {len(X_test):,}")
print(f"Test target distribution: {y_test.value_counts().to_dict()}")

# Load trained models
print("\n### Loading Trained Models ###")
models = {}
model_files = ['logistic_regression.pkl', 'lightgbm.pkl', 'xgboost.pkl']

for model_file in model_files:
    model_path = MODELS_DIR / model_file
    if model_path.exists():
        model_name = model_file.replace('.pkl', '')
        models[model_name] = joblib.load(model_path)
        print(f"OK Loaded: {model_name}")
    else:
        print(f"X  Not found: {model_file}")

if not models:
    print("\nX  No models found! Run step5_model_training.py first")
    sys.exit(1)

# Evaluate each model
results = {}

for model_name, model in models.items():
    print("\n" + "=" * 80)
    print(f"EVALUATING: {model_name.upper()}")
    print("=" * 80)
    
    # Make predictions
    if model_name == 'lightgbm':
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"\n### Primary Metrics ###")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n### Confusion Matrix ###")
    print(f"                 Predicted")
    print(f"                 0         1")
    print(f"Actual  0    {cm[0,0]:>6}   {cm[0,1]:>6}")
    print(f"        1    {cm[1,0]:>6}   {cm[1,1]:>6}")
    
    # True/False Positives/Negatives
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negatives:  {tn:>6}")
    print(f"False Positives: {fp:>6}")
    print(f"False Negatives: {fn:>6}")
    print(f"True Positives:  {tp:>6}")
    
    # Classification Report
    print(f"\n### Classification Report ###")
    print(classification_report(y_test, y_pred, digits=4))
    
    # Store results
    results[model_name] = {
        'auc_roc': float(auc_roc),
        'f1_score': float(f1),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }

# Model Comparison
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'AUC-ROC': [r['auc_roc'] for r in results.values()],
    'F1-Score': [r['f1_score'] for r in results.values()],
    'Accuracy': [r['accuracy'] for r in results.values()],
    'Precision': [r['precision'] for r in results.values()],
    'Recall': [r['recall'] for r in results.values()]
})

comparison_df = comparison_df.sort_values('AUC-ROC', ascending=False)
print(f"\n{comparison_df.to_string(index=False)}")

# Best Model
best_model_name = comparison_df.iloc[0]['Model']
best_auc = comparison_df.iloc[0]['AUC-ROC']

print("\n" + "=" * 80)
print("BEST MODEL")
print("=" * 80)
print(f"Winner: {best_model_name.upper()}")
print(f"   AUC-ROC: {best_auc:.4f}")
print(f"   F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")

# Save results
print("\n### Saving Evaluation Results ###")
reports_dir = OUTPUTS_DIR / 'reports'
reports_dir.mkdir(parents=True, exist_ok=True)

# Save comparison CSV
comparison_path = reports_dir / 'step6_model_comparison.csv'
comparison_df.to_csv(comparison_path, index=False)
print(f"OK Model comparison: {comparison_path}")

# Save detailed results JSON
results_path = reports_dir / 'step6_evaluation_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"OK Detailed results: {results_path}")

# Save best model name
best_model_path = reports_dir / 'step6_best_model.txt'
with open(best_model_path, 'w') as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"AUC-ROC: {best_auc:.4f}\n")
    f.write(f"F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}\n")
print(f"OK Best model info: {best_model_path}")

print("\n" + "=" * 80)
print("OK STEP 6 COMPLETED")
print("=" * 80)
print(f"OK All {len(models)} models evaluated")
print(f"OK Best model: {best_model_name}")
print(f"OK Results saved to: {reports_dir}/")
print("=" * 80)
