# Technical Plan: Credit Risk Prediction System

## Document Information

**Project**: Credit Risk Prediction System  
**Version**: 1.0  
**Date**: February 2, 2026  
**Authors**: Capstone Project Team  
**Status**: Approved

---

## Table of Contents

1. [Technical Overview](#technical-overview)
2. [System Architecture](#system-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Modeling Strategy](#modeling-strategy)
6. [Evaluation Framework](#evaluation-framework)
7. [Technology Stack](#technology-stack)
8. [Development Workflow](#development-workflow)
9. [Performance Requirements](#performance-requirements)
10. [Risk Mitigation](#risk-mitigation)

---

## 1. Technical Overview

### 1.1 Problem Formulation

**Problem Type**: Binary Classification  
**Task**: Predict probability of loan default  
**Target Variable**: `target` (0 = No Default, 1 = Default)  
**Evaluation Metric**: ROC-AUC (Area Under ROC Curve)

### 1.2 Input Data Characteristics

- **Format**: Parquet and CSV files
- **Tables**: 68 interconnected tables
- **Features**: 1,139 total columns
- **Samples**: ~1.5 million loan applications
- **Class Distribution**: Highly imbalanced (5% positive class)

### 1.3 Output Requirements

- Probability score [0, 1] for each application
- Feature importance rankings
- SHAP explanations for predictions
- Model performance metrics

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Parquet Files│  │  CSV Files   │  │   Metadata   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Processing Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Data Loader  │→ │    Merger    │→ │ Preprocessor │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Imputer    │→ │  Aggregator  │→ │   Engineer   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Modeling Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Logistic    │  │  LightGBM    │  │   XGBoost    │      │
│  │ Regression   │  │              │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │  CatBoost    │  │   Ensemble   │                        │
│  │              │  │              │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Evaluation Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Metrics    │  │    SHAP      │  │ Validation   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Predictions  │  │    Plots     │  │   Reports    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Responsibility | Output |
|-----------|---------------|--------|
| Data Loader | Read files from disk | Raw DataFrames |
| Merger | Join tables on case_id | Combined DataFrame |
| Preprocessor | Clean, validate data | Clean DataFrame |
| Imputer | Handle missing values | Complete DataFrame |
| Aggregator | Summarize multi-row tables | Aggregated features |
| Engineer | Create new features | Feature matrix |
| Models | Train and predict | Probabilities |
| Evaluator | Calculate metrics | Performance scores |
| Explainer | Generate SHAP values | Interpretations |

---

## 3. Data Pipeline

### 3.1 Data Loading Strategy

**Approach**: Lazy loading with memory optimization

```python
# Pseudocode
def load_data(file_path, file_format='parquet'):
    if file_format == 'parquet':
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, chunksize=100000)
    return df
```

**Optimization Techniques**:
- Use `dtype` specification to reduce memory
- Load only required columns when possible
- Use chunking for very large files
- Leverage Parquet's columnar format

### 3.2 Data Merging Strategy

**Table Relationships**:

```
Base Table (train_base.csv)
    ├── case_id (Primary Key)
    │
    ├─── Static Tables (1:1 relationship)
    │    ├── train_static_0_0.csv
    │    ├── train_static_0_1.csv
    │    ├── train_static_cb_0.csv
    │    ├── train_person_1.csv
    │    └── train_person_2.csv
    │
    └─── Dynamic Tables (1:N relationship - require aggregation)
         ├── train_applprev_*.csv (previous applications)
         ├── train_credit_bureau_a_*.csv (credit bureau data)
         ├── train_credit_bureau_b_*.csv
         ├── train_debitcard_1.csv
         ├── train_deposit_1.csv
         ├── train_other_1.csv
         └── train_tax_registry_*.csv
```

**Merge Logic**:

1. **Static Table Merge** (Simple left joins)
   ```python
   base_df.merge(static_df, on='case_id', how='left')
   ```

2. **Dynamic Table Aggregation** (Group and aggregate first)
   ```python
   agg_df = dynamic_df.groupby('case_id').agg({
       'feature1': ['mean', 'max', 'min', 'std', 'count'],
       'feature2': ['sum', 'mean'],
       ...
   })
   base_df.merge(agg_df, on='case_id', how='left')
   ```

### 3.3 Missing Value Strategy

**Three-Tier Approach**:

| Severity | Condition | Action |
|----------|-----------|--------|
| Critical | >80% missing | Drop column |
| High | 50-80% missing | Impute + add flag |
| Moderate | <50% missing | Impute only |

**Imputation Methods**:

```python
# Numerical features
SimpleImputer(strategy='median')

# Categorical features
SimpleImputer(strategy='most_frequent')

# Missing indicator
MissingIndicator(features='all')
```

### 3.4 Data Validation

**Validation Checks**:
- ✓ No duplicate case_id in final dataset
- ✓ No NaN in target variable
- ✓ Target values in {0, 1}
- ✓ Feature dtypes consistent
- ✓ No infinite values
- ✓ Reasonable value ranges

---

## 4. Feature Engineering

### 4.1 Aggregation Features

For tables with multiple rows per case_id:

| Feature Type | Aggregations |
|--------------|--------------|
| Numerical | mean, median, std, min, max, sum, count |
| Categorical | mode, count, nunique |
| Temporal | most_recent, oldest, time_difference |

### 4.2 Derived Features

**Domain-Specific Features**:

1. **Credit Utilization**
   ```python
   credit_utilization = current_balance / credit_limit
   ```

2. **Payment Ratio**
   ```python
   payment_ratio = payment_amount / loan_amount
   ```

3. **Application Frequency**
   ```python
   app_frequency = num_applications / time_period
   ```

4. **Debt-to-Income Ratio**
   ```python
   dti = total_debt / annual_income
   ```

### 4.3 Interaction Features

**Key Interactions**:
- Age × Income
- Loan Amount × Credit History Length
- Employment Status × Income Stability

### 4.4 Encoding Strategies

| Feature Type | Encoding Method |
|--------------|-----------------|
| Binary | Label Encoding |
| Ordinal | Ordinal Encoding |
| Nominal (low cardinality) | One-Hot Encoding |
| Nominal (high cardinality) | Target Encoding |

---

## 5. Modeling Strategy

### 5.1 Model Selection

**Candidates**:

| Model | Strengths | Use Case |
|-------|-----------|----------|
| Logistic Regression | Interpretable, fast, baseline | Benchmark |
| LightGBM | Fast, handles missing values, accurate | Primary candidate |
| XGBoost | Robust, well-tested, accurate | Primary candidate |
| CatBoost | Handles categoricals well | Alternative |

### 5.2 Hyperparameter Search Space

**LightGBM**:
```python
param_grid = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, -1],
    'num_leaves': [31, 50, 100],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0, 0.1, 1.0]
}
```

**XGBoost**:
```python
param_grid = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [1, 1.5, 2.0]
}
```

### 5.3 Class Imbalance Handling

**Approach 1: SMOTE (Synthetic Minority Over-sampling)**
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**Approach 2: Class Weights**
```python
class_weights = {0: 1, 1: 19}  # Inverse of class ratio
model = LGBMClassifier(class_weight=class_weights)
```

**Approach 3: Threshold Adjustment**
```python
# Instead of 0.5, find optimal threshold
optimal_threshold = find_threshold_by_f1(y_val, y_pred_proba)
```

### 5.4 Training Process

**Steps**:
1. Split data: 70% train, 15% validation, 15% test
2. Apply SMOTE on training set only
3. Train model with early stopping
4. Tune hyperparameters on validation set
5. Evaluate final model on test set

**Cross-Validation**:
- 5-fold stratified cross-validation
- Track mean and std of metrics

---

## 6. Evaluation Framework

### 6.1 Primary Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| ROC-AUC | ∫ TPR d(FPR) | > 0.75 |
| Precision | TP / (TP + FP) | > 0.70 |
| Recall | TP / (TP + FN) | > 0.65 |
| F1-Score | 2 × (P × R) / (P + R) | > 0.67 |

### 6.2 Confusion Matrix Analysis

```
                Predicted
                0       1
Actual  0      TN      FP
        1      FN      TP
```

**Business Interpretation**:
- **FP (False Positive)**: Predict default but customer repays → Lost opportunity
- **FN (False Negative)**: Predict repayment but customer defaults → Financial loss

**Cost-Sensitive Analysis**:
```python
cost = (FP × cost_of_false_positive) + (FN × cost_of_false_negative)
```

### 6.3 Model Interpretability

**SHAP (SHapley Additive exPlanations)**:

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global interpretation
shap.summary_plot(shap_values, X_test)

# Local interpretation
shap.force_plot(explainer.expected_value, shap_values[i], X_test.iloc[i])
```

**Feature Importance**:
- Tree-based built-in importance
- Permutation importance
- SHAP feature importance

---

## 7. Technology Stack

### 7.1 Core Libraries

| Purpose | Library | Version |
|---------|---------|---------|
| Data Manipulation | pandas | 2.0+ |
| Numerical Computing | numpy | 1.24+ |
| Machine Learning | scikit-learn | 1.3+ |
| Gradient Boosting | lightgbm | 4.0+ |
| Gradient Boosting | xgboost | 2.0+ |
| Gradient Boosting | catboost | 1.2+ |
| Imbalance Handling | imbalanced-learn | 0.11+ |
| Model Interpretation | shap | 0.43+ |
| Visualization | matplotlib | 3.7+ |
| Visualization | seaborn | 0.12+ |
| Data Reading | pyarrow | 13.0+ |

### 7.2 Development Tools

| Tool | Purpose |
|------|---------|
| Jupyter | Interactive notebooks |
| VS Code | Code editor |
| Git | Version control |
| GitHub | Code hosting |
| Jira | Project management |
| Python venv | Environment management |

### 7.3 System Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 16 GB
- Storage: 20 GB
- OS: Windows 10/11, Linux, macOS

**Recommended**:
- CPU: 8+ cores
- RAM: 32 GB
- Storage: 50 GB SSD
- GPU: Optional (for XGBoost GPU support)

---

## 8. Development Workflow

### 8.1 Git Workflow

**Branch Strategy**:
```
main (production-ready code)
  ├── develop (integration branch)
  │   ├── feature/data-pipeline
  │   ├── feature/model-training
  │   ├── feature/evaluation
  │   └── feature/documentation
  └── hotfix/bug-fixes
```

**Commit Convention**:
```
<type>(<scope>): <subject>

Types: feat, fix, docs, style, refactor, test, chore
Example: feat(data): add missing value imputation
```

### 8.2 Code Review Process

1. Create feature branch
2. Implement changes
3. Write tests
4. Create pull request
5. At least 1 team member reviews
6. Address feedback
7. Merge to develop

### 8.3 Testing Strategy

**Unit Tests**:
```python
def test_missing_value_imputation():
    df = pd.DataFrame({'A': [1, np.nan, 3]})
    imputed = impute_missing(df)
    assert imputed['A'].isna().sum() == 0
```

**Integration Tests**:
```python
def test_full_pipeline():
    df = load_data()
    processed = preprocess(df)
    features = engineer_features(processed)
    assert features.shape[1] > df.shape[1]
```

---

## 9. Performance Requirements

### 9.1 Training Performance

| Metric | Requirement |
|--------|-------------|
| Training Time | < 30 minutes per model |
| Memory Usage | < 16 GB |
| Disk I/O | Efficient (use Parquet) |

### 9.2 Prediction Performance

| Metric | Requirement |
|--------|-------------|
| Batch Prediction | 1000 cases/second |
| Single Prediction | < 100ms |

### 9.3 Model Size

- Serialized model: < 500 MB
- Feature matrix: Sparse representation if possible

---

## 10. Risk Mitigation

### 10.1 Technical Risks

| Risk | Mitigation |
|------|------------|
| Memory overflow | Use chunking, optimize dtypes |
| Overfitting | Cross-validation, regularization, early stopping |
| Long training time | Start with subset, use early stopping |
| Data leakage | Careful train/test split, no target in features |

### 10.2 Data Quality Risks

| Risk | Mitigation |
|------|------------|
| Missing values | Comprehensive imputation strategy |
| Outliers | Robust scaling, outlier detection |
| Incorrect joins | Validation checks, unit tests |
| Feature shift | Monitor feature distributions |

---

## Appendix A: File Naming Conventions

### A.1 Code Files
- `snake_case` for Python files: `data_loader.py`
- `PascalCase` for classes: `DataLoader`
- `UPPERCASE` for constants: `MAX_FEATURES`

### A.2 Notebooks
- Sequential numbering: `01_data_exploration.ipynb`
- Descriptive names: `02_missing_value_analysis.ipynb`

### A.3 Data Files
- Preserve original names from Kaggle
- Generated files: `processed_train_data.parquet`

---

## Appendix B: Useful Commands

### B.1 Environment Setup
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### B.2 Running Scripts
```bash
python scripts/run_pipeline.py
python scripts/train_model.py --model lightgbm
```

### B.3 Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

---

**Document Version**: 1.0  
**Last Updated**: February 2, 2026  
**Approved By**: Technical Team  
**Next Review**: February 16, 2026
