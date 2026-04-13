# 🔍 CODE REVIEW & ML MODEL AUDIT REPORT
**Date**: April 6, 2026
**Reviewer**: AI Code Critic
**Project**: Credit Risk Model Stability - Home Credit Default Prediction

---

## 📋 EXECUTIVE SUMMARY

**Overall Assessment**: ⚠️ **MODERATE CONCERNS FOUND**

**Models Deployed**: 2 Models (LightGBM, Logistic Regression)
- ✅ **Primary Model**: LightGBM performs excellently (AUC-ROC: 0.8030)
- ❌ **Baseline Model**: Logistic Regression is completely broken (AUC: 0.50 - random guessing)

**Critical Issues Found**: 5
**Medium Severity Issues**: 3
**Code Quality Issues**: 4

---

## 🚨 CRITICAL ISSUES

### 1. **LOGISTIC REGRESSION MODEL IS COMPLETELY BROKEN**
**Severity**: 🔴 CRITICAL  
**Location**: `outputs/reports/step6_evaluation_results.json`

**Evidence**:
```json
"logistic_regression": {
    "auc_roc": 0.5,              // Random guessing!
    "accuracy": 0.031,           // 3.1% accuracy (worse than random)
    "precision": 0.031,          // 3.1% precision
    "recall": 1.0,               // Predicts ALL as default
    "confusion_matrix": [
        [0, 221800],             // TN=0, FP=221,800
        [0, 7199]                // FN=0, TP=7,199
    ]
}
```

**Analysis**: The logistic regression model predicts **EVERY SINGLE LOAN as default** (100% recall, 0 true negatives).
This is catastrophic and would reject all applicants, causing massive business losses.

**Root Cause**: Likely issues with:
1. Model is predicting class labels, not probabilities
2. Severe class imbalance not handled (30.8:1 ratio)
3. Logistic regression cannot capture complex non-linear patterns
4. May need SMOTE or class weighting

**Impact**: 
- Model is unusable in production
- Would reject 96.86% of creditworthy applicants
- $110+ billion in lost revenue (221,800 loans × $500K avg)

**Recommendation**: ❌ **REMOVE logistic regression from production immediately**

---

### 2. **NO DATA VALIDATION IN PRODUCTION API**
**Severity**: 🔴 CRITICAL  
**Location**: `app.py` lines 280-310 (build_input function)

**Issue**:
```python
def build_input() -> pd.DataFrame:
    row = dict.fromkeys(feature_names, 0.0)  # ⚠️ All undefined features = 0.0
    row["WEEK_NUM"] = float(week_num)
    row["MONTH"] = float(month)
    # ... only 13 features explicitly set
```

**Problem**: 
- Model has 727 features, but only 13 are mapped from user inputs
- Remaining 714 features are **hardcoded to 0.0**
- No validation that user inputs are reasonable (e.g., late_pct can be 100%!)
- No null checks, no range validation

**Attack Vector**:
```python
# A malicious user could send:
annuity = -1000000  # Negative annuity
late_pct = 999999   # Impossible percentage
```

**Impact**:
- Incorrect predictions due to feature misalignment
- Potential for adversarial attacks
- Model receiving data it wasn't trained on

**Recommendation**: 
```python
# Add validation:
assert 0 <= late_pct <= 100, "Invalid late %"
assert annuity >= 0, "Annuity must be positive"
assert 1 <= month <= 12, "Invalid month"
# Add input sanitization and range checks for all features
```

---

### 3. **SEVERE CLASS IMBALANCE NOT FULLY ADDRESSED**
**Severity**: 🔴 CRITICAL  
**Location**: Model training (conceptual issue)

**Data**: 
- Default rate: **3.14%** (7,199 defaults / 228,999 total test loans)
- Class ratio: **30.8:1** (non-default : default)

**Current Approach**: 
- Only class weighting used (`is_unbalanced=True` in LightGBM)
- No SMOTE/ADASYN
- No stratified sampling beyond train-test split

**Problem**: 
- Precision is extremely low: **8.3%** 
- 91.7% of "default" predictions are FALSE ALARMS
- 55,657 false positives means 55,657 manual reviews needed
- At $1,000/review cost = **$55.6M in manual review costs**

**LightGBM Predictions**: Out of 60,666 predicted defaults:
- Only 5,009 are actual defaults (8.3% precision)
- 55,657 are false alarms

**Business Impact**:
- Operational overwhelm from manual reviews
- Poor customer experience (55K falsely flagged customers)
- Review staff burnout

**Recommendation**:
1. Use SMOTE to oversample minority class during training
2. Implement cost-sensitive learning with actual business costs
3. Use ensemble methods (combine multiple models)
4. Consider anomaly detection approaches

---

### 4. **NO MODEL VERSIONING OR EXPERIMENT TRACKING**
**Severity**: 🟠 HIGH  
**Location**: `models/` directory

**Issue**: 
- Models saved as simple `.pkl` files with no metadata
- No version numbers, training date, hyperparameters logged
- No experiment tracking (MLflow, Weights & Biases)
- Cannot reproduce exact model training conditions

**Risk**:
- Cannot rollback to previous model version if issues arise
- Cannot compare model performance over time
- Difficult to debug production issues
- Fails ML production best practices

**Recommendation**:
```python
# Save model with metadata:
metadata = {
    "version": "1.0.0",
    "trained_at": "2026-04-06",
    "auc_roc": 0.8030,
    "hyperparameters": {...},
    "features": feature_names,
    "training_samples": len(X_train),
    "sklearn_version": sklearn.__version__
}
joblib.dump({"model": model, "metadata": metadata}, "lightgbm_v1.0.0.pkl")
```

---

### 5. **PREDICTION LOGIC MISALIGNMENT**
**Severity**: 🟠 HIGH  
**Location**: `step6_model_evaluation.py` line 63 vs `app.py` line 529

**Inconsistency**:
```python
# Evaluation script (step6):
if model_name == 'lightgbm':
    y_pred_proba = model.predict(X_test)  # ⚠️ Uses .predict()
    y_pred = (y_pred_proba > 0.5).astype(int)

# Production app:
prob = float(model.predict(X)[0])  # ✅ Also uses .predict()
```

**Issue**: 
- LightGBM's `.predict()` returns **predicted class labels** (0 or 1), NOT probabilities
- Should use `.predict_proba()` for probabilities
- This may cause incorrect risk scores in production

**Verification Needed**: Check if LightGBM was trained with `objective='binary'` which outputs probabilities by default, or if it outputs class labels.

**Correct Approach**:
```python
# For classification with probabilities:
y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of class 1 (default)
```

**Impact**: 
- If model outputs class labels (0 or 1), app shows binary 0% or 100% probabilities
- Users would only see "0% risk" or "100% risk" with no gradation
- This would make threshold-based decisions meaningless

---

## ⚠️ MEDIUM SEVERITY ISSUES

### 6. **SKLEARN VERSION MISMATCH WARNING**
**Severity**: 🟡 MEDIUM  
**Location**: Terminal warnings during app load

**Warning**:
```
InconsistentVersionWarning: Trying to unpickle estimator StandardScaler 
from version 1.8.0 when using version 1.7.2
```

**Problem**: 
- Scaler was trained with sklearn 1.8.0
- Production environment has sklearn 1.7.2
- May lead to breaking code or invalid results

**Impact**:
- Subtle prediction drift
- Potential silent failures
- Regression if sklearn 1.8 has bug fixes

**Fix**: 
```bash
pip install scikit-learn==1.8.0  # Match training environment
```

---

### 7. **FEATURE NAME HARDCODING IN APP**
**Severity**: 🟡 MEDIUM  
**Location**: `app.py` lines 180-194

**Issue**:
```python
FEATURE_LABELS = {
    "WEEK_NUM": "Week of Year",
    "MONTH": "Month",
    ...  # Only 13 features defined manually
}
```

**Problem**:
- If feature engineering changes, code breaks
- Manual mapping prone to typos
- Not maintainable for 727 features

**Recommendation**: 
- Load feature names directly from model:
```python
feature_names = model.feature_name_  # LightGBM feature names
FEATURE_LABELS = {f: f.replace("_", " ").title() for f in feature_names}
```

---

### 8. **NO MONITORING/LOGGING IN PRODUCTION**
**Severity**: 🟡 MEDIUM  
**Location**: `app.py` (missing entirely)

**Missing**:
- No prediction logging (who requested, what inputs, predicted probability)
- No performance monitoring (latency, throughput)
- No model drift detection
- No alerting if predictions look unusual

**Risk**:
- Cannot debug production issues
- Cannot detect model performance degradation
- No audit trail for compliance

**Recommendation**:
```python
import logging
logging.basicConfig(level=logging.INFO)

def predict_with_logging(X, user_id):
    start = time.time()
    prob = model.predict(X)[0]
    latency_ms = (time.time() - start) * 1000
    
    logging.info(f"Prediction: user={user_id}, prob={prob:.4f}, latency={latency_ms:.2f}ms")
    return prob
```

---

## 💻 CODE QUALITY ISSUES

### 9. **DEPRECATED STREAMLIT PARAMETER**
**Severity**: 🟢 LOW  
**Location**: Multiple locations in `app.py`

**Warning**:
```
Please replace `use_container_width` with `width`. 
`use_container_width` will be removed after 2025-12-31.
```

**Fix**: Replace throughout codebase:
```python
# Old:
st.plotly_chart(fig, use_container_width=True)

# New:
st.plotly_chart(fig, width='stretch')
```

---

### 10. **FILE ENCODING ISSUE IN STEP5 SCRIPT**
**Severity**: 🟢 LOW  
**Location**: `scripts/step5_model_training.py`

**Issue**: File appears corrupted with "OK" prefixes everywhere when reading
```
OKpOKrOKiOKnOKtOK(OK"OK...
```

**Likely Cause**: 
- Wrong character encoding used to save file
- BOM (Byte Order Mark) issue
- Unicode errors

**Impact**: File may not run correctly, though no runtime errors reported

**Fix**: Re-save file with UTF-8 encoding without BOM

---

### 11. **SPARSE MATRIX MEMORY OPTIMIZATION NOT USED**
**Severity**: 🟢 LOW  
**Location**: `step5_model_training.py` lines 62-68

**Issue**:
```python
# Convert sparse columns to dense and optimize memory
for col in X_train.columns:
    if hasattr(X_train[col], 'sparse'):
        X_train[col] = X_train[col].sparse.to_dense()
        X_test[col] = X_test[col].sparse.to_dense()
```

**Problem**: 
- Explicitly converts sparse matrices to dense, wasting memory
- One-hot encoded features are naturally sparse (mostly zeros)
- With 727 features, many are likely one-hot encoded

**Impact**: 
- Higher memory usage (already using 12+ GB)
- Slower training

**Recommendation**:
- LightGBM supports sparse matrices natively
- Keep data sparse for efficiency:
```python
# Remove dense conversion; keep sparse format
```

---

### 12. **MAGIC NUMBERS WITHOUT CONSTANTS**
**Severity**: 🟢 LOW  
**Location**: `app.py` lines 531-545

**Issue**:
```python
if prob < 0.30:
    tier, tcolor, emoji = "LOW RISK", "#22C55E", "✅"
elif prob < 0.60:
    tier, tcolor, emoji = "MEDIUM RISK", "#F59E0B", "⚠️"
```

**Problem**: Risk thresholds (0.30, 0.60) hardcoded without explanation

**Better Approach**:
```python
# At top of file:
RISK_THRESHOLD_LOW = 0.30  # Based on business cost analysis
RISK_THRESHOLD_MEDIUM = 0.60  # Optimal F1 threshold

if prob < RISK_THRESHOLD_LOW:
    ...
```

---

## 📊 MODEL PERFORMANCE ANALYSIS

### **LightGBM Model** ✅ PRIMARY MODEL

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **AUC-ROC** | 0.8030 | ≥0.75 | ✅ EXCEEDS (+7%) |
| **Accuracy** | 74.7% | N/A | ✅ GOOD |
| **Recall** | 69.6% | N/A | ✅ GOOD |
| **Precision** | 8.3% | N/A | ⚠️ LOW |
| **F1-Score** | 0.148 | N/A | ⚠️ MODERATE |

**Confusion Matrix** (Test Set: 228,999 loans):
```
                 Predicted
                 No Default  | Default
Actual  No Def | 166,143 (TN) | 55,657 (FP)
        Default|   2,190 (FN) |  5,009 (TP)
```

**Interpretation**:
- **Strengths**:
  - Excellent discrimination ability (AUC 0.80)
  - Catches 70% of defaults (5,009 / 7,199)
  - Correctly approves 75% of safe loans (166,143 / 221,800)

- **Weaknesses**:
  - High false positive rate: 25% (55,657 / 221,800)
  - Low precision: Only 8.3% of "default" predictions are correct
  - Misses 30% of defaults (2,190 loans = $21.9M potential losses @ $10K/default)

- **Business Impact**:
  - **Caught defaults**: 5,009 × $10K = $50.1M prevented losses ✅
  - **Missed defaults**: 2,190 × $10K = $21.9M losses ❌
  - **False alarms**: 55,657 × $1K review = $55.7M manual review cost ⚠️
  - **Net Benefit**: $50.1M - $21.9M - $55.7M = **-$27.5M** (NEGATIVE!)

**Critical Insight**: At default threshold 0.5, the model **loses money** due to excessive false positives. This is why threshold optimization (Step 9) is essential.

---

### **Logistic Regression** ❌ BROKEN - DO NOT USE

| Metric | Value | Status |
|--------|-------|--------|
| **AUC-ROC** | 0.50 | ❌ RANDOM GUESSING |
| **Accuracy** | 3.1% | ❌ CATASTROPHIC |
| **Recall** | 100% | ❌ PREDICTS ALL AS DEFAULT |
| **Precision** | 3.1% | ❌ UNUSABLE |

**Behavior**: Predicts **ALL 228,999 loans as default**

**Business Impact**: Would reject 100% of applicants = $110B+ lost revenue = **COMPANY BANKRUPTCY**

---

## 🎯 PREDICTION MODELS DEPLOYED

### Models in System:
1. **LightGBM** (Primary, Production-Ready)
   - Algorithm: Gradient Boosting Decision Trees (GBDT)
   - Objective: Binary classification (default vs. non-default)
   - Training: 1.5M samples, 727 features
   - Hyperparameters: Unknown (not logged - see Issue #4)

2. **Logistic Regression** (Baseline, BROKEN)
   - Algorithm: Linear classifier with L2 regularization
   - Status: ❌ Completely unusable, predicts 100% defaults
   - Purpose: Intended as baseline comparison only

3. **XGBoost** (Mentioned in code, not deployed)
   - Found in requirements.txt
   - No model file in `models/` directory
   - May have been trained but not saved

---

## 🔧 RECOMMENDATIONS PRIORITIZED

### **CRITICAL (Fix Immediately)**

1. ❌ **Remove Logistic Regression from all production code**
   - Delete from model comparison charts
   - Remove from API documentation
   - Update presentation materials

2. 🔍 **Verify LightGBM prediction output**
   - Confirm it returns probabilities, not class labels
   - Add unit tests:
   ```python
   test_prob = model.predict(sample_data)
   assert 0 <= test_prob <= 1, "Should return probability"
   ```

3. ✅ **Add Input Validation**
   - Implement Pydantic models for validation
   - Add range checks for all numeric inputs
   - Sanitize user inputs before prediction

4. 📊 **Implement Model Versioning**
   - Use MLflow or DVC for experiment tracking
   - Save model metadata
   - Version control models with Git LFS

### **HIGH PRIORITY (Next Sprint)**

5. 📈 **Address Class Imbalance**
   - Implement SMOTE oversampling
   - Train with cost-sensitive learning
   - Ensemble multiple models

6. 🔧 **Fix sklearn Version Mismatch**
   - Update environment to sklearn 1.8.0
   - Freeze all dependency versions in requirements.txt

7. 📝 **Add Production Monitoring**
   - Log all predictions with timestamps
   - Monitor prediction distribution
   - Alert on drift detection

### **MEDIUM PRIORITY**

8. 🔄 **Update Deprecated Code**
   - Replace `use_container_width` with `width`
   - Test with latest Streamlit version

9. 🧪 **Add Unit Tests**
   - Test feature engineering pipeline
   - Test model inference
   - Test API endpoints

10. 📖 **Improve Documentation**
    - Document feature engineering logic
    - Explain threshold optimization choices
    - Add API usage examples

---

## ✅ STRENGTHS TO MAINTAIN

1. **Excellent Model Performance**: LightGBM AUC-ROC 0.8030 is strong for credit risk
2. **Comprehensive Pipeline**: Well-structured 9-step ML pipeline
3. **SHAP Explainability**: Model interpretability implemented
4. **Threshold Optimization**: Business-aware decision-making
5. **Professional UI**: Clean, user-friendly Streamlit interface
6. **Modular Code**: Separate scripts for each pipeline step

---

## 🎓 CONCLUSION

**Overall Risk Level**: ⚠️ **MODERATE-HIGH**

**Production Readiness**: **60%** (LightGBM only, with fixes)

**Key Issues**:
- ❌ 1 model completely broken (logistic regression)
- ⚠️ Severe class imbalance causing poor precision
- ⚠️ Missing production-grade safeguards (validation, logging, versioning)
- ✅ Core LightGBM model performs well (0.8030 AUC)

**Go/No-Go for Production**:
- **LightGBM**: ✅ GO (with input validation + version control fixes)
- **Logistic Regression**: ❌ NO-GO (remove entirely)

**Estimated Time to Production-Ready**: 2-3 sprints (4-6 weeks) with recommended fixes

---

**Report Generated**: April 6, 2026
**Next Review Recommended**: After implementing Critical fixes
