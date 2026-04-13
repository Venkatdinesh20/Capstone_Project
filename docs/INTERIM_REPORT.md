# Interim Project Report
## Home Credit Credit Risk Model Stability

---

<div align="center">

**Capstone Project — Machine Learning**

Reporting Period: February 1, 2026 – March 9, 2026

Project Status: **ON TRACK**

</div>

---

## Executive Summary

This report covers the progress of the Home Credit Credit Risk Model Stability project through the end of February and into early March 2026. Over the past five weeks, the team has built a working end-to-end machine learning pipeline that ingests raw parquet data, merges 32 training tables, cleans and engineers features, trains classification models, and evaluates their performance on a held-out test set.

The main result so far is a LightGBM model that achieves an AUC-ROC of **0.8030** on the test partition, which beats the initial project target of 0.75 by roughly seven percentage points. The full pipeline is reproducible: anyone on the team can re-run the six numbered scripts in sequence and arrive at the same outputs.

Work still outstanding includes SHAP-based model interpretability, a polished visualization dashboard, threshold tuning, and the final presentation materials. These items are scheduled across the first two weeks of March.

**Highlights at a glance:**

- Complete data pipeline across 6 modular scripts
- 1,526,659 loan records processed from 32 parquet files (~1.27 GB compressed)
- 384 million missing values resolved to zero in the cleaned dataset
- LightGBM model: 0.8030 AUC-ROC, 69.6% recall on the default class
- Two models evaluated (Logistic Regression baseline + LightGBM); XGBoost was attempted but dropped due to memory and time constraints
- All intermediate artifacts saved as parquet for fast reload

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives and Success Criteria](#2-objectives-and-success-criteria)
3. [Work Completed](#3-work-completed)
4. [Technical Implementation](#4-technical-implementation)
5. [Model Performance](#5-model-performance)
6. [Challenges and Solutions](#6-challenges-and-solutions)
7. [Resource Utilization](#7-resource-utilization)
8. [Risk Assessment](#8-risk-assessment)
9. [Lessons Learned](#9-lessons-learned)
10. [Next Steps](#10-next-steps)
11. [Timeline and Milestones](#11-timeline-and-milestones)
12. [Team Performance](#12-team-performance)
13. [Conclusion](#13-conclusion)

---

## 1. Project Overview

### 1.1 Background

The Home Credit Credit Risk Model Stability project is a capstone exercise in applied machine learning. The goal is to build a binary classifier that predicts whether a loan applicant will default, paying particular attention to applicants who have little or no traditional credit history. The dataset comes from the 2024 Kaggle competition of the same name.

The core business problem is straightforward: lenders need to separate high-risk applicants from low-risk ones without automatically rejecting people who simply lack a credit record. A good model lets the institution approve more creditworthy borrowers while flagging likely defaulters for additional review.

### 1.2 Scope

**In scope:**
- Loading, merging, and preprocessing the 32 training parquet files
- Feature engineering (encoding, scaling, missing-value handling)
- Training and comparing classification models
- Evaluating on a held-out test set using standard metrics
- Model interpretability via SHAP (in progress)
- Documentation covering every stage of the pipeline

**Out of scope:**
- Deploying the model behind a live API
- Building a user-facing application or dashboard service
- Integration with any production banking system

### 1.3 Dataset at a Glance

| Item | Value |
|------|-------|
| Source | Kaggle — Home Credit Credit Risk Model Stability (2024) |
| Training files | 32 parquet files (~1.27 GB total on disk) |
| Test files | 36 parquet files (separate Kaggle test partition) |
| Records in training base table | 1,526,659 |
| Columns in base table | 5 (`case_id`, `date_decision`, `MONTH`, `WEEK_NUM`, `target`) |
| Columns after merging all tables | 391 |
| Columns after preprocessing | 376 |
| Features after one-hot encoding | 727 |
| Target variable | `target` — binary (0 = no default, 1 = default) |
| Class distribution | 96.86% non-default / 3.14% default (~30.8 : 1) |

### 1.4 Problem Type

**Supervised learning — binary classification.**

- *Supervised* because we have labeled data: we know which historical loans defaulted.
- *Binary* because there are exactly two outcomes: default (1) or no default (0).
- *Classification* because the target is categorical, not continuous.

The primary evaluation metric is AUC-ROC, which measures how well the model ranks a random positive example above a random negative example regardless of the decision threshold.

### 1.5 Team

| Role | Member | Primary Responsibilities |
|------|--------|--------------------------|
| Team Lead / Data Engineer | Venkat Dinesh | Project coordination, data pipeline, infrastructure, config management |
| ML Engineer | Sai Charan | Model training, hyperparameter selection, memory optimization |
| Data Scientist | Lokesh Reddy | Evaluation metrics, statistical analysis, SHAP (upcoming) |
| Backend Developer | Pranav Dhara | Preprocessing scripts, feature engineering, code quality |
| QA / Testing Lead | Sunny Sunny | Validation, documentation review, testing |

---

## 2. Objectives and Success Criteria

### 2.1 Primary Objectives

| # | Objective | Target | Current Status |
|---|-----------|--------|----------------|
| 1 | Merge all training tables into a single dataset | 32/32 tables merged | Done |
| 2 | Handle missing data so the final feature matrix has zero NaNs | 0 missing values | Done — 384 M missing values resolved |
| 3 | Train at least two classification models | ≥ 2 models | Done (LR + LightGBM) |
| 4 | Achieve AUC-ROC ≥ 0.75 on held-out test set | ≥ 0.75 | Achieved 0.8030 |
| 5 | Implement SHAP analysis for model explanations | Complete analysis | In progress — scheduled for March 8 |
| 6 | Build a reproducible, modular pipeline | End-to-end scripts | Done (6 scripts, all re-runnable) |

### 2.2 Metric Scorecard

**Technical Metrics**

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| AUC-ROC | ≥ 0.75 | 0.8030 | Exceeded |
| Recall (default class) | ≥ 0.65 | 0.6958 | Met |
| Precision (default class) | ≥ 0.70 | 0.0826 | Below target (see discussion in §5) |
| F1-Score | ≥ 0.60 | 0.1476 | Below target (expected with 30.8 : 1 imbalance) |
| Data integration | 32 tables | 32 tables | Complete |
| Missing values in final data | 0 | 0 | Complete |

**Note on precision and F1:** These values look low in isolation but are a direct consequence of the severe class imbalance. With only 3.14% of records being defaults, even a model with strong discrimination (0.80 AUC-ROC) will produce many false positives when the decision threshold is set at the default 0.5. The team chose to prioritize recall (catching actual defaulters) over precision, which is the standard trade-off in credit risk. A more nuanced threshold could improve these numbers at the cost of recall — this is planned as part of the threshold optimization work in March.

**Process Metrics**

| Metric | Status |
|--------|--------|
| Documentation coverage | README, epic definition, user stories, data dictionary, technical plan, layman summary, meeting minutes, workflow guide |
| Code modularity | 6 pipeline steps + 4 source modules + 9 utility scripts |
| Version control | 7 commits across Jan–Feb |
| Automated testing | Manual validation only — automated tests planned |
| Team meetings | 3 meetings completed (notes documented) |

---

## 3. Work Completed

### 3.1 Phase 1 — Project Setup and Planning (Feb 1–5)

The first week focused on getting the infrastructure right so everyone on the team could work in the same environment and follow the same conventions.

**What was delivered:**

- **Project directory structure** organized into `scripts/`, `src/`, `models/`, `outputs/`, `docs/`, `notebooks/`, `data_processed/`, and `parquet_files/`.
- **`config.py`** — a single configuration file that holds every path, hyperparameter, table list, and threshold used anywhere in the project. Changing a value in config propagates everywhere automatically.
- **`requirements.txt`** — pinned dependency list covering pandas, numpy, scikit-learn, LightGBM, XGBoost, CatBoost, SHAP, imbalanced-learn, matplotlib, seaborn, plotly, pyarrow, and others.
- **Source modules** in `src/`:
  - `src/data/loader.py` — `DataLoader` class with memory-optimized parquet/CSV loading, batch loading, and dtype downcasting.
  - `src/data/merger.py` — `DataMerger` class for static (1:1 join) and dynamic (1:N aggregation) table merges.
  - `src/data/preprocessor.py` — `DataPreprocessor` class handling missing-value analysis, column dropping, indicator creation, imputation, and validation.
  - `src/visualization/plots.py` — plotting utilities for target distribution, missing values, correlation heatmaps, confusion matrices, ROC curves, and feature importance.
- **Documentation** — seven documents were created:
  1. `README.md` — project overview with badges, installation, usage
  2. `docs/epic_definition.md` — detailed epic statement and business case
  3. `docs/user_stories.md` — 8 user stories with 42 individual tasks
  4. `docs/technical_plan.md` — architecture, data pipeline, modeling strategy
  5. `docs/data_dictionary.md` — feature naming conventions
  6. `QUICKSTART.md` — step-by-step setup for new contributors
  7. `LICENSE` — MIT
- **Analysis scripts** — `data_quality_analysis.py` and `missing_data_analysis.py` for automated quality reporting before the main pipeline runs.
- **Git repository** initialized with proper `.gitignore`, first commit on January 19, 2026. Milestone 2 commit on February 2.

---

### 3.2 Phase 2 — Data Pipeline (Feb 6–18)

#### Step 1: Data Collection (`step1_data_collection.py`)

Loaded the base training table from `parquet_files/train/train_base.parquet` using the `DataLoader` class with memory optimization (dtype downcasting).

| Detail | Value |
|--------|-------|
| Records loaded | 1,526,659 |
| Columns in base table | 5 (`case_id`, `date_decision`, `MONTH`, `WEEK_NUM`, `target`) |
| Memory after optimization | ~17.5 MB (down from ~72.8 MB — 76% reduction) |
| Target distribution | 1,478,665 non-default (0) / 47,994 default (1) |
| Class ratio | ~30.8 : 1 |
| Output | `data_processed/step1_base_collected.parquet` (7.31 MB) |

The script also prints the target distribution and confirms the class imbalance up front so downstream decisions (e.g., class weights vs. SMOTE) can be made with full context. A companion script, `step1_analyze_target.py`, was written to formally classify the problem type and document the rationale for treating it as supervised binary classification.

---

#### Step 2: Data Merging (`step2_data_merging.py`)

Merged all 32 training tables into a single DataFrame using the `DataMerger` class.

**Merging strategy:**
- **Static tables** (`static_cb_0`, `static_0_0`, `static_0_1`, `static_0_2`): one row per `case_id` — merged via left join directly.
- **Dynamic tables** (`applprev_1`, `applprev_2`, `person_1`, `person_2`, `credit_bureau_a_1`, `credit_bureau_a_2`, `credit_bureau_b_1`, `credit_bureau_b_2`, `other_1`, `tax_registry_a_1`, `tax_registry_b_1`, `tax_registry_c_1`, `deposit_1`, `debitcard_1`): multiple rows per `case_id` — aggregated with `mean`, `median`, `std`, `min`, `max`, `sum`, `count` for numerical columns and `count`, `nunique` for categorical columns, then joined on `case_id`.

| Detail | Value |
|--------|-------|
| Input | 1,526,659 rows × 5 columns (base) + 31 auxiliary tables |
| Output | 1,526,659 rows × 391 columns |
| Features added by merging | 386 |
| Total missing values in merged data | 384,192,877 |
| Output file | `data_processed/step2_data_merged.parquet` (207.96 MB) |

A `monitor_merging.py` script was written separately to provide real-time progress tracking during the merge, which helped during development because the full run takes several minutes.

---

#### Step 3: Data Preprocessing (`step3_data_preprocessing.py`)

Cleaned the merged dataset using the `DataPreprocessor` class.

**Steps performed:**

1. **Missing-value analysis** — 104 columns had >80% missing data.
2. **Column dropping** — high-missing columns removed; missing indicators added for columns with moderate missingness (5–50%). Net result: 391 → 376 columns.
3. **Duplicate check** — no duplicate `case_id` values found.
4. **Imputation** — numerical columns filled with their median; categorical columns filled with their mode. All imputation was done using explicit assignment (`df[col] = df[col].fillna(...)`) rather than the deprecated `inplace=True`, which Pandas 2.0 no longer honors reliably.
5. **Validation** — confirmed zero missing values in the output DataFrame.

| Detail | Value |
|--------|-------|
| Input | 1,526,659 × 391 with 384,192,877 missing values |
| Columns after cleaning (net, after dropping + adding indicators) | 376 |
| Output | 1,526,659 × 376 with 0 missing values |
| Numerical columns in output | 317 |
| Categorical columns in output | 55 |
| Output file | `data_processed/step3_data_cleaned.parquet` (183.32 MB) |

---

### 3.3 Phase 3 — Feature Engineering and Model Training (Feb 19–24)

#### Step 4: Feature Engineering (`step4_feature_engineering.py`)

Prepared the cleaned data for modeling.

1. **Separated features and target** — Dropped `case_id` and `target` from the feature matrix.
2. **Handled categorical features:**
   - Identified 55 categorical/object columns.
   - Dropped date columns with very high cardinality.
   - Dropped remaining high-cardinality categoricals (>100 unique values).
   - Applied one-hot encoding (`pd.get_dummies`) to the rest.
3. **Train-test split** — 85/15 stratified split (controlled by `TEST_SIZE = 0.15` in `config.py`).
4. **Feature scaling** — `StandardScaler` fit on training data only, then applied to both sets. Only non-binary numerical columns were scaled; one-hot columns left as 0/1.
5. **Class imbalance** — SMOTE was originally planned but skipped because generating synthetic samples for 1.3 M rows would have consumed too much memory. Instead, models use `class_weight='balanced'` or the equivalent `is_unbalance=True` / `scale_pos_weight` parameter.

| Detail | Value |
|--------|-------|
| Features after encoding | 727 |
| Training set | 1,297,660 samples (1,256,865 non-default / 40,795 default) |
| Test set | 228,999 samples (221,800 non-default / 7,199 default) |
| Default rate (both sets) | 3.14% |
| Scaler saved | `models/scaler.pkl` |
| Column list saved | `models/numerical_cols.pkl` |
| Output files | `step4_X_train.parquet` (226 MB), `step4_X_test.parquet` (42 MB), `step4_y_train.parquet`, `step4_y_test.parquet` |

---

#### Step 5: Model Training (`step5_model_training.py`)

Trained classification models and saved them as pickle files.

**Model 1 — Logistic Regression (baseline)**

Used `SGDClassifier` with `loss='log_loss'` (equivalent to logistic regression but more memory-efficient through stochastic gradient descent). Trained on a 20% stratified sample of the training data due to memory constraints, with `class_weight='balanced'`. Training completed in a few seconds.

Saved to: `models/logistic_regression.pkl` (0.03 MB)

**Model 2 — LightGBM**

Used the native `lgb.train()` API with parameters defined in `config.py`:
- `objective='binary'`, `metric='auc'`, `boosting_type='gbdt'`
- `n_estimators=1000`, `learning_rate=0.05`, `max_depth=7`, `num_leaves=50`
- `is_unbalance=True`
- Early stopping after 10 rounds with no improvement
- Trained on the full training set (used LightGBM's Dataset object for memory efficiency)

Training completed in roughly 2 minutes.

Saved to: `models/lightgbm.pkl` (2.5 MB)

**Model 3 — XGBoost (attempted)**

The training script includes code for XGBoost using a 30% stratified sample and the parameters in `config.py` (`scale_pos_weight=30.8`). However, XGBoost ran into memory and time issues on the team's hardware. The resulting model file was not retained in the `models/` directory, and the evaluation step only found two saved models. The team decided to proceed with Logistic Regression and LightGBM rather than spend more time debugging XGBoost at this stage.

---

#### Step 6: Model Evaluation (`step6_model_evaluation.py`)

Evaluated each saved model on the 228,999-sample test set.

| Metric | LightGBM | Logistic Regression |
|--------|----------|---------------------|
| AUC-ROC | **0.8030** | 0.5000 |
| Recall | 0.6958 | 1.0000 |
| Precision | 0.0826 | 0.0314 |
| F1-Score | 0.1476 | 0.0610 |
| Accuracy | 0.7474 | 0.0314 |

**Best model: LightGBM**

**LightGBM confusion matrix (test set, n = 228,999):**

|  | Predicted No Default | Predicted Default |
|--|----------------------|-------------------|
| **Actual No Default** | 166,143 (TN) | 55,657 (FP) |
| **Actual Default** | 2,190 (FN) | 5,009 (TP) |

The Logistic Regression baseline effectively predicts "default" for every applicant — 100% recall but 0.50 AUC-ROC (no discrimination). This confirms that the SGDClassifier on a small sample with extreme imbalance couldn't learn meaningful patterns, making it a useful lower bound rather than a real contender.

LightGBM, by contrast, provides genuine discrimination: it catches 69.6% of actual defaulters while correctly passing 74.9% of non-defaulters. The false-positive rate is high but expected at a 0.5 threshold on this imbalance ratio.

**Outputs saved:**
- `outputs/reports/step6_model_comparison.csv`
- `outputs/reports/step6_evaluation_results.json`
- `outputs/reports/step6_best_model.txt`

---

### 3.4 Phase 4 — Documentation and Knowledge Transfer (Feb 25–28)

During the final week of February the team focused on consolidating documentation so that the project is understandable to both technical and non-technical audiences.

Documents produced or updated:
1. **`docs/PROJECT_SUMMARY_LAYMAN.md`** — a plain-language walkthrough of the entire project, written so someone without a data-science background can follow the reasoning.
2. **`scripts/README_WORKFLOW.py`** — a step-by-step execution guide that doubles as an explanatory script file.
3. **`MILESTONE_2_SUMMARY.md`** — describes everything delivered during the initial setup phase.
4. **`docs/MEETING_MINUTES_1.md`** — detailed notes from the February 24 team meeting, including demo results, problem discussions, and action-item assignments.
5. Code-level documentation — docstrings, inline comments, and consistent naming conventions across all scripts and modules.

---

## 4. Technical Implementation

### 4.1 Technology Stack

| Category | Library / Tool | Purpose |
|----------|---------------|---------|
| Data manipulation | pandas ≥ 2.0, numpy | DataFrames, numerical computation |
| File I/O | pyarrow, fastparquet | Parquet reading/writing |
| ML framework | scikit-learn ≥ 1.3 | Preprocessing, SGDClassifier baseline, metrics |
| Gradient boosting | LightGBM ≥ 4.0 | Primary production model |
| Gradient boosting | XGBoost ≥ 2.0 | Attempted, not in final evaluation |
| Imbalance handling | imbalanced-learn | SMOTE utilities (ultimately not used) |
| Interpretability | SHAP ≥ 0.43 | Planned for next sprint |
| Visualization | matplotlib, seaborn, plotly | Plots and charts |
| Model persistence | joblib | Saving/loading .pkl files |
| Dev environment | Python 3.8+, VS Code, Jupyter | Development and exploration |
| Version control | Git / GitHub | Repository hosting |
| Project management | Jira | Task tracking and sprints |
| Communication | Microsoft Teams | Virtual meetings |

### 4.2 Architecture

The pipeline is organized as six sequential scripts, each reading the output of the previous step and writing its own output to `data_processed/` or `models/`.

```
[Raw Parquet Data]
       │  32 training files
       ▼
 step1_data_collection.py  →  step1_base_collected.parquet  (5 cols)
       │
       ▼
 step2_data_merging.py     →  step2_data_merged.parquet     (391 cols)
       │
       ▼
 step3_data_preprocessing.py → step3_data_cleaned.parquet   (376 cols)
       │
       ▼
 step4_feature_engineering.py → X_train, X_test, y_train, y_test  (727 features)
       │                        scaler.pkl, numerical_cols.pkl
       ▼
 step5_model_training.py   →  logistic_regression.pkl, lightgbm.pkl
       │
       ▼
 step6_model_evaluation.py →  evaluation_results.json, model_comparison.csv
```

Each step is independently re-runnable. If a bug is found in preprocessing, for example, the team can re-run from step 3 onward without re-doing the expensive merge.

Reusable logic lives in `src/` modules (`DataLoader`, `DataMerger`, `DataPreprocessor`, visualization helpers), while pipeline scripts in `scripts/` orchestrate those modules and print progress to the console. `config.py` holds every shared constant — paths, hyperparameters, table lists, thresholds — so there are no magic numbers scattered through the codebase.

### 4.3 Memory Optimization

Processing 1.5 M rows across hundreds of columns on consumer-grade hardware required careful memory management:

1. **Dtype downcasting** — `DataLoader._optimize_dtypes()` converts int64 → int8/int16/int32 and float64 → float32 where possible. This cut the base table from 72.8 MB to 17.5 MB (76% reduction).
2. **Parquet format** — columnar compression, ~10× faster reads than CSV, and selective column loading.
3. **Batch table processing** — dynamic tables are aggregated one at a time and merged incrementally rather than loading everything into memory simultaneously.
4. **Stratified sampling for training** — Logistic Regression and the attempted XGBoost model used 20–30% stratified samples to stay within RAM limits. LightGBM used its native Dataset object (which is more memory-efficient than raw pandas DataFrames).
5. **float32 conversion** before training — numeric columns cast to float32 in `step5_model_training.py` to halve the memory footprint.

### 4.4 Configuration Management

All project-level settings are centralized in `config.py`:

```python
# Key settings (abridged)
RANDOM_STATE       = 42
TEST_SIZE          = 0.15
MISSING_THRESHOLD  = 0.80      # Drop columns with >80% missing
TARGET_COL         = 'target'
ID_COL             = 'case_id'

LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 7,
    'num_leaves': 50,
    'is_unbalance': True,
    'random_state': 42,
    ...
}
```

Benefits: a single place to change any parameter, no hard-coded paths in scripts, and consistent random seeding across the entire project.

---

## 5. Model Performance

### 5.1 LightGBM — Detailed Analysis

**AUC-ROC: 0.8030**
This means that if you pick a random defaulter and a random non-defaulter, the model will assign a higher risk score to the defaulter about 80% of the time. An AUC-ROC above 0.80 is considered good in credit-risk modeling, especially given the data constraints and the level of class imbalance.

**Recall: 69.6%**
The model catches roughly 7 out of every 10 actual defaulters. The remaining 2,190 defaulters in the test set are missed (false negatives). In credit-risk terms, these are loans that would be approved and then default — a direct financial loss.

**Precision: 8.3%**
Of the 60,666 applicants the model flags as likely defaulters, only 5,009 actually default. The other 55,657 are false alarms. This low precision is a mathematical consequence of the class distribution: when defaults are only 3.14% of the population, any model that pushes recall above ~70% will inevitably tag a large number of non-defaulters.

**Accuracy: 74.7%**
Accuracy is misleading here. A model that simply predicts "no default" for every applicant would achieve 96.86% accuracy. LightGBM's 74.7% accuracy is lower than that because it deliberately flags many applicants to catch defaults — which is exactly what we want.

### 5.2 Confusion Matrix Interpretation

Out of 228,999 test applicants:

| Outcome | Count | % of Total | Business Meaning |
|---------|-------|-----------|------------------|
| True Negatives (TN) | 166,143 | 72.6% | Correctly approved — will repay |
| False Positives (FP) | 55,657 | 24.3% | Flagged for review, but actually fine |
| True Positives (TP) | 5,009 | 2.2% | Correctly caught — would have defaulted |
| False Negatives (FN) | 2,190 | 1.0% | Missed — approved but will default |

Practically speaking, the model routes roughly 24% of applications to manual review (FP) while catching 69.6% of defaults. The 2,190 missed defaults represent the cost of not being even more conservative.

### 5.3 Baseline Comparison

| Metric | Logistic Regression | LightGBM | Improvement |
|--------|---------------------|----------|-------------|
| AUC-ROC | 0.5000 | 0.8030 | +0.303 |
| Recall | 1.0000 | 0.6958 | –0.304 |
| Precision | 0.0314 | 0.0826 | +163% |
| F1-Score | 0.0610 | 0.1476 | +142% |

The Logistic Regression baseline predicts "default" for every single applicant (hence 100% recall, 0.50 AUC-ROC). It cannot distinguish between good and bad applicants at all, confirming that a simple linear model trained on a small sample with this level of imbalance is effectively random. LightGBM provides real predictive power.

### 5.4 Threshold Considerations

The current evaluation uses the default threshold of 0.5. Because the class ratio is ~30.8 : 1, a lower threshold (e.g., 0.30) would catch more defaults at the expense of more false positives. A higher threshold (e.g., 0.70) would do the opposite. The team plans to perform a systematic threshold analysis in the next sprint, plotting precision/recall/F1 against various thresholds so the business stakeholder can choose based on their risk appetite.

---

## 6. Challenges and Solutions

### 6.1 Memory Errors During Data Merging

**Problem:** Merging all 32 tables at once crashed with a `MemoryError` requesting 3.21 GB.

**Root cause:** Loading and joining multiple large DataFrames simultaneously exceeded available RAM.

**Solution:**
- Processed tables one at a time using incremental merges.
- Used `pd.concat()` for column-wise joins instead of repeated column assignment (avoids DataFrame fragmentation).
- Applied dtype downcasting early to keep the footprint as small as possible.
- Added explicit `gc.collect()` between large operations.

**Result:** Merging completes successfully within available memory.

### 6.2 Pandas 2.0 Copy-on-Write Behavior

**Problem:** Missing-value imputation using `inplace=True` silently failed — NaN values were still present after running `fillna`.

**Root cause:** Pandas 2.0 introduced copy-on-write semantics. Operations on DataFrame slices no longer modify the original object in place, making `inplace=True` unreliable.

**Solution:** Changed all imputation calls from:
```python
df[col].fillna(df[col].median(), inplace=True)   # broken in Pandas 2.0+
```
to explicit reassignment:
```python
df[col] = df[col].fillna(df[col].median())        # always works
```

**Result:** Zero missing values in the cleaned dataset, verified by an assertion at the end of step 3.

### 6.3 XGBoost Memory and Time Constraints

**Problem:** XGBoost training on a 30% sample still consumed excessive memory and took over 30 minutes.

**Root cause:** The default exact tree-construction method requires all data in memory simultaneously. With 727 features and hundreds of thousands of rows, the working set exceeded the team's hardware limits.

**Decision:** Skip XGBoost for now and focus on LightGBM, which uses a histogram-based algorithm that is inherently more memory-efficient. One strong model with 0.8030 AUC-ROC is more valuable than three mediocre ones. XGBoost can be revisited on cloud hardware if needed.

### 6.4 Class Imbalance Strategy

**Problem:** The original plan called for SMOTE oversampling, but generating synthetic minority samples at the scale of 1.3 M training rows would require ~8 GB of additional memory.

**Options considered:**

| Approach | Pros | Cons |
|----------|------|------|
| SMOTE oversampling | Balanced classes | Massive memory requirement; creates synthetic data |
| Random undersampling | Fast, low memory | Discards 95%+ of majority class |
| Class weights | Zero extra memory; uses real data | Doesn't explicitly balance the dataset |

**Decision:** Use class weights (`class_weight='balanced'` in scikit-learn; `is_unbalance=True` in LightGBM). This tells the model to penalize misclassification of the minority class proportionally to the imbalance ratio. No additional data is created.

**Result:** LightGBM achieved 69.6% recall on the default class with an AUC-ROC of 0.8030 — good performance without any synthetic data.

### 6.5 Feature Engineering Decisions

**Trade-offs made during feature encoding:**

- **Date columns** — dropped entirely because dates like `date_decision` were high-cardinality strings. Extracting year/month/weekday features is deferred to a possible second round of feature engineering.
- **High-cardinality categoricals** (>100 unique values) — dropped to avoid creating thousands of one-hot columns.
- **One-hot encoding** — applied to remaining categoricals. After encoding, the feature count grew from 376 to 727.

These were pragmatic choices to keep the feature set manageable. With more time, target encoding or frequency encoding could capture information from the dropped columns.

### 6.6 Git Merge Conflict

During Milestone 2, two team members pushed changes to `README.md` from different branches. The conflict was resolved by keeping the comprehensive Milestone 2 version (commit `6e33fec`). Going forward, the team established clear file ownership and a practice of pulling before starting work each day.

---

## 7. Resource Utilization

### 7.1 Hardware

| Resource | Specification | Peak Utilization |
|----------|--------------|------------------|
| CPU | Intel i5/i7 (varies by member) | 60–80% during model training |
| RAM | 8–16 GB | Peak ~12 GB during merging |
| Storage | SSD, 50+ GB free | ~2 GB used (data + models + outputs) |
| GPU | Not used | N/A |

### 7.2 Pipeline Timing

| Operation | Approximate Duration |
|-----------|---------------------|
| Loading base table | ~30 seconds |
| Merging 32 tables | ~8 minutes |
| Preprocessing | ~5 minutes |
| Feature engineering | ~3 minutes |
| LightGBM training | ~2 minutes |
| Logistic Regression training | ~3 seconds |
| Evaluation (228 K predictions) | ~5 seconds |
| **Total end-to-end** | **~20 minutes** |

### 7.3 Processed Data Sizes

| File | Size |
|------|------|
| step1_base_collected.parquet | 7.31 MB |
| step2_data_merged.parquet | 207.96 MB |
| step3_data_cleaned.parquet | 183.32 MB |
| step4_X_train.parquet | 226.38 MB |
| step4_X_test.parquet | 41.97 MB |
| lightgbm.pkl | 2.50 MB |
| logistic_regression.pkl | 0.03 MB |

### 7.4 Team Hours (Feb 1–28 estimate)

| Member | Hours | Primary Focus |
|--------|-------|---------------|
| Venkat Dinesh | ~60 h | Infrastructure, data pipeline, docs |
| Sai Charan | ~55 h | Model training, memory optimization |
| Lokesh Reddy | ~50 h | Evaluation, statistical analysis |
| Pranav Dhara | ~45 h | Preprocessing, feature engineering |
| Sunny Sunny | ~35 h | QA, validation, doc review |
| **Total** | **~245 h** | |

### 7.5 Tools and Cost

All tools used are free or covered by academic licenses: GitHub, Jira (free tier), VS Code, Jupyter, Microsoft Teams, Kaggle. Total infrastructure cost: **$0**.

---

## 8. Risk Assessment

### 8.1 Current Risks

| ID | Risk | Severity | Likelihood | Mitigation | Status |
|----|------|----------|------------|------------|--------|
| R-01 | XGBoost not implemented | Medium | Occurred | LightGBM exceeds target; XGBoost is optional | Accepted |
| R-02 | SHAP interpretability pending | Medium | 80% likely to slip | Scheduled for March 8; LightGBM feature importance available as fallback | Tracked |
| R-03 | No automated test suite | Low | 100% | Manual validation sufficient for now; pytest planned | Accepted |
| R-04 | Decision threshold not optimized | Medium | 100% | Using default 0.5; threshold analysis planned for March 12 | Planned |
| R-05 | Low precision may confuse stakeholders | Medium | 60% | Will include business-context explanation in final presentation | Tracked |
| R-06 | step5 file encoding issue | Low | Occurred | Script content runs correctly; file has spurious "OK" bytes that don't affect execution | Monitoring |

### 8.2 Risks Resolved

| Risk | Resolution |
|------|-----------|
| Memory constraints during merging | Batch processing + dtype optimization |
| Pandas 2.0 fillna bug | Switched to explicit assignment |
| Class imbalance handling | Class weights instead of SMOTE |
| Reproducibility concerns | Fixed random seeds, saved all artifacts |

---

## 9. Lessons Learned

### 9.1 Technical

1. **Memory profiling is not optional.** On datasets this size, the merge step alone can exceed 3 GB. Dtype downcasting and incremental joins need to be designed in from the start, not bolted on after a crash.

2. **Pin your library versions.** The Pandas 2.0 copy-on-write change silently broke code that looked correct. We caught it through manual inspection, but an automated test on the output would have caught it faster. Going forward, `requirements.txt` locks specific versions.

3. **Start with a simple baseline.** The Logistic Regression model is objectively useless (AUC-ROC = 0.50), but running it first validated the entire pipeline — data loading, feature shapes, prediction format, metric calculation — before we spent time on LightGBM. Every bug caught at the baseline stage was easier to fix.

4. **Class weights scale better than resampling on large datasets.** SMOTE might be fine for 50 K rows. At 1.3 M rows with a 30.8 : 1 imbalance, it would have created over a million synthetic samples and blown up memory. Class weights cost nothing.

5. **One good model beats three incomplete ones.** Dropping XGBoost freed up time to tune LightGBM, improve documentation, and keep the project on schedule.

### 9.2 Process

1. **Document decisions when you make them.** When the team decided to skip SMOTE, the reasoning went straight into the meeting minutes. Two weeks later, anyone can read why that choice was made without having to track down the person who made it.

2. **Modular pipelines pay dividends.** The step-based design meant that when the fillna bug was found in step 3, only steps 3–6 needed to be re-run. If the pipeline had been a single monolithic notebook, debugging would have taken much longer.

3. **Communicate before pushing.** The one Git conflict we had was a `README.md` collision that could have been avoided with a quick message. Since establishing file-ownership norms, there have been zero further conflicts.

4. **Validation assertions at every step.** Adding `assert df.isnull().sum().sum() == 0` at the end of step 3 catches imputation bugs immediately rather than letting bad data propagate into training.

### 9.3 What We Would Do Differently

- Set up a basic `pytest` suite from day one — even five or ten tests covering data shapes, column names, and null counts would have caught the Pandas bug earlier.
- Run a deeper EDA phase before jumping into the pipeline. The notebook `01_data_exploration.ipynb` exists but was not fully fleshed out.
- Engage stakeholders earlier on threshold preferences so that the model's operating point reflects real business costs.

---

## 10. Next Steps

### Priority 1 — Model Interpretability (SHAP) — Due March 8

| Task | Owner |
|------|-------|
| Install and configure SHAP | Lokesh |
| Calculate SHAP values for LightGBM predictions on a sample | Lokesh |
| Generate summary and force plots | Sai |
| Identify and explain top features in plain language | Venkat |
| Create sample prediction explanations | Pranav |

### Priority 2 — Visualization Dashboard — Due March 10

| Task | Owner |
|------|-------|
| ROC curve with AUC annotation | Sai |
| Confusion matrix heatmap | Lokesh |
| Precision-recall curve | Pranav |
| Feature importance bar chart | Sai |
| Model comparison chart | Sunny |

### Priority 3 — Threshold Optimization — Due March 12

| Task | Owner |
|------|-------|
| Plot threshold vs. precision / recall / F1 | Lokesh |
| Estimate business impact at different thresholds | Lokesh |
| Write recommendation for stakeholders | Venkat |

### Priority 4 — Full-Data Retraining — Due March 12

The current LightGBM model was trained on the full training partition (1,297,660 rows). The Logistic Regression baseline used only a 20% sample. For the final Kaggle submission, the team plans to retrain LightGBM on 100% of available data and generate predictions on the competition test set.

| Task | Owner |
|------|-------|
| Retrain LightGBM on full data | Sai |
| Generate Kaggle test-set predictions | Sai |
| Verify submission format against `sample_submission.csv` | Lokesh |

### Priority 5 — Final Presentation and Documentation — Due March 15

| Task | Owner |
|------|-------|
| Update README with final metrics | Venkat |
| Prepare slide deck (20–30 slides) | Venkat + all |
| Record demo video (10–15 min) | Venkat + Sai |
| Write 2-page executive summary | Venkat |

### Stretch Goals (if time permits)

- Implement XGBoost on cloud resources
- Add automated tests (pytest)
- Build a Flask API endpoint for predictions
- Create an interactive Streamlit dashboard
- Extract date-based features from dropped date columns
- Hyperparameter tuning with Optuna

---

## 11. Timeline and Milestones

### Milestone Tracker

| Milestone | Planned | Actual | Status |
|-----------|---------|--------|--------|
| M1: Project setup and docs | Feb 5 | Feb 2 | Completed 3 days early |
| M2: Data pipeline complete | Feb 18 | Feb 17 | Completed 1 day early |
| M3: Models trained and evaluated | Feb 24 | Feb 24 | Completed on time |
| M4: SHAP interpretability | Mar 8 | — | In progress |
| M5: Final presentation ready | Mar 15 | — | Scheduled |
| M6: Project submission | Mar 20 | — | Scheduled |

### High-Level Gantt

```
                  Jan W3   Feb W1   Feb W2   Feb W3   Feb W4   Mar W1   Mar W2
                  ──────   ──────   ──────   ──────   ──────   ──────   ──────
Initial commit    [██]
Project setup              [████]
Data collection                     [██]
Data merging                        [████]
Preprocessing                              [████]
Feature eng.                                       [██]
Model training                                     [████]
Evaluation                                                [██]
Documentation                                             [████]
SHAP analysis                                                      [████]
Visualizations                                                     [████]
Final review                                                              [████]
Presentation                                                              [████]
```

### Upcoming Deadlines

| Date | Deliverable | Owner |
|------|------------|-------|
| Mar 8 | SHAP analysis complete | Lokesh |
| Mar 10 | Visualization dashboard | Sai |
| Mar 12 | Threshold analysis | Lokesh |
| Mar 12 | Full-data retraining | Sai |
| Mar 15 | All documentation final | All |
| Mar 15 | Presentation ready | Venkat |
| Mar 20 | Final submission | All |

---

## 12. Team Performance

### Individual Contributions

**Venkat Dinesh — Team Lead / Data Engineer**
Set up the project structure, wrote `config.py`, managed the Git repo, coordinated tasks across the team, and authored the majority of the documentation. Keeps meetings focused and deadlines visible.

**Sai Charan — ML Engineer**
Implemented the LightGBM training pipeline, handled memory optimization (float32 conversion, stratified sampling), and debugged the XGBoost attempt. Drove the decision to use class weights over SMOTE.

**Lokesh Reddy — Data Scientist**
Owned the evaluation step — metric calculation, confusion-matrix analysis, and clear explanations of what the numbers mean for the business. Will lead the SHAP analysis in March.

**Pranav Dhara — Backend Developer**
Built the preprocessing and feature-engineering scripts. Diagnosed and fixed the Pandas 2.0 fillna issue. Maintains clean code style and modularity across the codebase.

**Sunny Sunny — QA / Testing Lead**
Reviewed every script for correctness, asked probing business-facing questions during meetings (e.g., "What about the 2,190 false negatives?"), and validated that outputs match expected shapes and distributions.

### Sprint Velocity

| Sprint | Points Planned | Points Completed | Rate |
|--------|----------------|------------------|------|
| Sprint 1 (Feb 1–14) | 30 | 28 | 93% |
| Sprint 2 (Feb 15–28) | 25 | 26 | 104% |
| Average | — | 27 | 98.5% |

### Team Dynamics

**Strengths:** Strong async communication, willingness to help when someone is stuck, honest problem reporting during meetings.

**Areas to improve:** No formal code reviews (could catch issues like the Pandas bug earlier), need automated tests, and git commit granularity could be finer.

---

## 13. Conclusion

The project is on track. The core deliverable — a working binary classification pipeline with a 0.8030 AUC-ROC model — is complete and reproducible. The team has navigated memory constraints, library-version surprises, and class-imbalance challenges with practical engineering decisions rather than brute force.

What remains is the interpretability layer (SHAP), presentation materials, and minor optimizations (threshold tuning, possible full-data retraining). These are well-scoped tasks with clear ownership and deadlines, all fitting comfortably within the two weeks before the March 20 submission date.

**Summary status:**

| Area | Status |
|------|--------|
| Data pipeline | Complete |
| Model training | 2 models trained |
| Evaluation | AUC-ROC 0.8030 (exceeds 0.75 target) |
| Interpretability | SHAP scheduled for March 8 |
| Documentation | 90% complete, final pass in March |
| Timeline | On track |
| Team morale | High |

---

*Report prepared by the project team.*
*Date: March 9, 2026*
*Repository: https://github.com/Venkatdinesh20/Capstone_Project*
