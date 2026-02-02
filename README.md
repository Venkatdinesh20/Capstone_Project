# Credit Risk Prediction System

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning system that predicts credit default risk for loan applications using advanced ensemble methods and feature engineering on the Home Credit dataset.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Epic Definition](#epic-definition)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [User Stories](#user-stories)
- [Technical Approach](#technical-approach)
- [Team](#team)
- [Timeline](#timeline)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project develops a machine learning model to predict credit default risk for loan applicants, particularly focusing on individuals with limited or no credit history. By analyzing behavioral patterns and financial indicators across 68 interconnected data tables, we aim to provide fair and accurate credit risk assessments.

### Key Features
- 🎯 **Risk Score Prediction**: Generate probability scores for loan default
- 🔍 **Model Interpretability**: SHAP-based explanations for predictions
- 📊 **Comprehensive Data Pipeline**: Automated processing of 68+ data tables
- ⚖️ **Fair Assessment**: Evaluation methodology for thin-file applicants
- 📈 **Multiple Models**: Comparison of Logistic Regression, LightGBM, XGBoost, and CatBoost

## 🎪 Epic Definition

**Epic Name**: Credit Risk Prediction System

**Epic Description**: Create a machine learning model that analyzes loan applicant information to predict default probability. Working with a comprehensive Kaggle dataset containing 68 tables and approximately 1.5 million loan records, the model evaluates factors including demographics, income, credit history, and past loan behavior to assess credit risk.

## 💼 Business Problem

Lenders face a critical dilemma:
- **Too Conservative**: Reject creditworthy applicants → Lost business opportunities
- **Too Lenient**: Approve high-risk applicants → Financial losses from defaults

**Additional Challenge**: Many applicants lack traditional credit scores due to limited borrowing history, leading to automatic rejection despite potentially responsible financial behavior.

**Our Solution**: Leverage behavioral pattern analysis and alternative data sources to make informed credit decisions, bridging the gap between risk management and financial inclusion.

## 📊 Dataset

**Source**: [Home Credit Credit Risk Model Stability - Kaggle Competition (2024)](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability)

| Characteristic | Value |
|----------------|-------|
| Total Loan Records | ~1.5 million |
| Number of Tables | 68 |
| Total Features | 1,139 |
| File Format | Parquet & CSV |
| Target Variable | Default (Binary: 0/1) |
| Default Rate | ~5% (Imbalanced) |
| Prize Pool | $105,000 |

### Data Categories
1. **Static Tables**: One row per application (demographics, static info)
2. **Dynamic Tables**: Multiple rows per application (credit bureau data, past applications, transaction history)

### Key Challenges
- ⚠️ Significant missing values across multiple features
- ⚠️ Severe class imbalance (95% non-default, 5% default)
- ⚠️ Complex relationships between 68 interconnected tables
- ⚠️ Encoded column names requiring interpretation

## 📁 Project Structure

```
home-credit-credit-risk-model-stability/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── config.py                         # Configuration settings
│
├── csv_files/                        # Original CSV data
│   ├── train/                        # Training datasets
│   └── test/                         # Test datasets
│
├── parquet_files/                    # Parquet format data
│   ├── train/
│   └── test/
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb    # Initial data analysis
│   ├── 02_data_quality.ipynb        # Quality assessment
│   ├── 03_feature_engineering.ipynb # Feature creation
│   ├── 04_model_training.ipynb      # Model development
│   └── 05_model_evaluation.ipynb    # Results & interpretability
│
├── src/                              # Source code
│   ├── data/                         # Data processing modules
│   │   ├── __init__.py
│   │   ├── loader.py                 # Data loading utilities
│   │   ├── merger.py                 # Table joining logic
│   │   └── preprocessor.py          # Cleaning & preprocessing
│   │
│   ├── features/                     # Feature engineering
│   │   ├── __init__.py
│   │   ├── aggregations.py          # Aggregate features
│   │   └── transformers.py          # Feature transformations
│   │
│   ├── models/                       # Model training & evaluation
│   │   ├── __init__.py
│   │   ├── train.py                 # Training pipeline
│   │   ├── evaluate.py              # Evaluation metrics
│   │   └── explain.py               # Model interpretability
│   │
│   └── visualization/                # Plotting utilities
│       ├── __init__.py
│       └── plots.py                 # Visualization functions
│
├── scripts/                          # Executable scripts
│   ├── run_pipeline.py              # Full pipeline execution
│   ├── train_model.py               # Model training script
│   └── generate_predictions.py      # Prediction generation
│
├── models/                           # Saved model artifacts
│   └── .gitkeep
│
├── outputs/                          # Generated outputs
│   ├── figures/                     # Plots and visualizations
│   ├── reports/                     # Analysis reports
│   └── predictions/                 # Model predictions
│
└── docs/                            # Additional documentation
    ├── epic_definition.md           # Detailed epic
    ├── user_stories.md              # All user stories
    ├── technical_plan.md            # Technical architecture
    └── data_dictionary.md           # Feature descriptions
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 16GB+ RAM recommended (for handling large datasets)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Venkatdinesh20/Capstone_Project.git
cd home-credit-credit-risk-model-stability
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
- Download from [Kaggle Competition Page](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data)
- Place files in `csv_files/` or `parquet_files/` directories

## 📖 Usage

### Quick Start

**1. Data Exploration**
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

**2. Run Full Pipeline**
```bash
python scripts/run_pipeline.py
```

**3. Train Models**
```bash
python scripts/train_model.py --model lightgbm --use-smote
```

**4. Generate Predictions**
```bash
python scripts/generate_predictions.py --model-path models/best_model.pkl
```

### Python API Example

```python
from src.data.loader import DataLoader
from src.features.aggregations import FeatureAggregator
from src.models.train import ModelTrainer

# Load and merge data
loader = DataLoader('parquet_files/train/')
df = loader.load_and_merge_all()

# Create features
aggregator = FeatureAggregator()
features = aggregator.transform(df)

# Train model
trainer = ModelTrainer(model_type='lightgbm')
model = trainer.fit(features, target)

# Evaluate
metrics = trainer.evaluate(X_test, y_test)
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

## 👥 User Stories

### Story 1: Risk Score Prediction
**As a** credit analyst  
**I want** to see a risk score for each loan application  
**So that** I can focus attention on problematic applications

### Story 2: Understanding Predictions
**As a** loan officer  
**I want** to know the reasons behind high-risk predictions  
**So that** I can have meaningful conversations with applicants

### Story 3: Combining Data Tables
**As a** data engineer  
**I want** all scattered data tables merged into one usable dataset  
**So that** we can train a model effectively

### Story 4: Handling Missing Data
**As a** data analyst  
**I want** a solid strategy for dealing with missing information  
**So that** incomplete applications don't break the system

### Story 5: Model Evaluation
**As a** data scientist  
**I want** proper evaluation metrics calculated  
**So that** we can trust the model's predictions

### Story 6: Feature Importance
**As a** business analyst  
**I want** to understand which factors drive defaults  
**So that** I can report meaningful insights to management

### Story 7: Fair Assessment
**As a** loan applicant with no credit history  
**I want** the system to consider my actual financial habits  
**So that** I'm not automatically rejected

### Story 8: Documentation
**As a** project evaluator  
**I want** everything properly documented  
**So that** future teams can continue the work

## 🔧 Technical Approach

### Data Processing Pipeline
1. **Data Loading**: Read 68 tables from Parquet/CSV files
2. **Data Merging**: Join tables using case_id keys
3. **Missing Value Handling**: 
   - Drop columns with >80% missing values
   - Impute numerical: median
   - Impute categorical: mode
   - Create missingness indicator flags
4. **Feature Aggregation**: Calculate statistics for multi-row tables (mean, sum, max, min, std)
5. **Feature Engineering**: Create interaction features and domain-specific features

### Modeling Strategy

| Model | Purpose | Key Parameters |
|-------|---------|----------------|
| Logistic Regression | Baseline | C=1.0, max_iter=1000 |
| LightGBM | Primary candidate | n_estimators=1000, learning_rate=0.05 |
| XGBoost | Primary candidate | n_estimators=1000, learning_rate=0.05 |
| CatBoost | Alternative | iterations=1000, learning_rate=0.05 |

### Class Imbalance Handling
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **Class Weights**: Adjust loss function
- **Stratified Sampling**: Maintain class distribution in train/test splits

### Evaluation Metrics
- **Primary**: ROC-AUC (competition metric)
- **Secondary**: Precision, Recall, F1-Score
- **Business Metric**: Confusion Matrix analysis
- **Validation**: 5-Fold Cross-Validation

### Model Interpretability
- **Feature Importance**: Tree-based importances
- **SHAP Values**: Understand individual predictions
- **Partial Dependence Plots**: Feature effect visualization

## 👨‍💻 Team

| Name | Role | Responsibilities |
|------|------|------------------|
| Venkat Dinesh | Data Engineer | Data acquisition, table merging, documentation |
| Sai Charan | Data Scientist | Model training, feature importance, evaluation |
| Lokesh Reddy | ML Engineer | Model optimization, cross-validation, testing |
| Pranav Dhara | Analyst | Feature aggregation, visualization, reporting |
| Sunny Sunny | Developer | Pipeline development, prediction generation |

## 📅 Timeline

| Weeks | Phase | Activities |
|-------|-------|------------|
| 1-2 | Data Understanding | EDA, quality assessment, relationship mapping |
| 3 | Feature Engineering | Aggregations, transformations, new features |
| 4-5 | Model Development | Training, tuning, comparison |
| 6-8 | Evaluation & Documentation | Testing, SHAP analysis, final documentation |

## 📊 Results

> Results will be updated as the project progresses

### Model Performance (Expected)

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| Logistic Regression | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |
| CatBoost | TBD | TBD | TBD | TBD |

### Top 10 Important Features

1. TBD
2. TBD
3. TBD
...

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

**Project Link**: [https://github.com/Venkatdinesh20/Capstone_Project](https://github.com/Venkatdinesh20/Capstone_Project)

**Jira Board**: [https://venkatdinesh60.atlassian.net/jira/core/projects/CP/board](https://venkatdinesh60.atlassian.net/jira/core/projects/CP/board)

## 🙏 Acknowledgments

- Home Credit for providing the dataset and hosting the Kaggle competition
- Kaggle community for insights and discussions
- Open-source contributors of scikit-learn, LightGBM, XGBoost, and SHAP libraries

---

**Last Updated**: February 2, 2026  
**Status**: 🚧 In Development - Milestone 2
