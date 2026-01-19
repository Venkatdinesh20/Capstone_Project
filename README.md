
# Credit Risk Scoring System with Temporal Stability Analysis

A machine learning system that predicts loan default risk while maintaining consistent performance over time. Built using the 2024 Home Credit Kaggle competition dataset.

## Project Overview

Traditional credit scoring models have two major problems. First, they exclude 1.4 billion people worldwide who lack formal credit history. Second, they degrade over time as economic conditions change – a model that works great today might perform poorly six months from now.

This project tackles both issues by building a credit risk scoring system that uses alternative data sources (payment behavior, transaction patterns, previous applications) and optimizes for temporal stability, not just accuracy.

## Why This Project is Different

Most credit scoring projects train a model, report accuracy on a test set, and call it done. Real production models don't work that way. They face distribution shift, changing customer behavior, and economic cycles that make yesterday's predictions unreliable today.

We're using the 2024 Home Credit competition dataset specifically because it measures model stability over time using a custom Gini stability metric. This directly addresses what banks actually care about – consistent performance across quarters, not just a single good test score.

## Dataset

**Source:** Home Credit – Credit Risk Model Stability (Kaggle 2024)

| Attribute | Value |
|-----------|-------|
| Total Records | ~1.5 million loan applications |
| Number of Tables | 68 |
| Total Features | 1,139 columns |
| Target Variable | Binary (1 = default, 0 = no default) |
| Class Distribution | ~3-5% positive |
| File Format | Parquet |

**Download:**
```bash
pip install kaggle
kaggle competitions download -c home-credit-credit-risk-model-stability
```

## Tech Stack

Everything here is free and open-source. No paid cloud services required.

| Category | Tool |
|----------|------|
| Data Processing | Pandas, DuckDB |
| ML Models | LightGBM, XGBoost, CatBoost, Scikit-learn |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API | FastAPI |
| Dashboard | Streamlit |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Monitoring | Evidently AI |

## Project Structure

```
credit-risk-stability/
│
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/                    # DVC tracked (not in Git)
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_stability_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── features.py
│   ├── model.py
│   └── explainability.py
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── dashboard/
│   └── app.py
│
├── tests/
│   ├── test_data_processing.py
│   ├── test_features.py
│   ├── test_model.py
│   └── test_api.py
│
├── models/                  # Saved model artifacts
│
└── mlruns/                  # MLflow experiment logs
```

## Installation

### Prerequisites
- Python 3.9+
- Git
- 8GB RAM minimum (16GB recommended)

### Setup

1. Clone the repository
```bash
git clone https://github.com/[your-username]/credit-risk-stability.git
cd credit-risk-stability
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download dataset
```bash
kaggle competitions download -c home-credit-credit-risk-model-stability
unzip home-credit-credit-risk-model-stability.zip -d data/raw/
```

5. Initialize DVC (optional, for data versioning)
```bash
dvc init
```

## Usage

### Running the Notebooks

Start with the notebooks in order:

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### Training the Model

```bash
python src/model.py --config configs/training_config.yaml
```

### Starting the API

```bash
uvicorn api.main:app --reload
```

API will be available at `http://localhost:8000`

API documentation at `http://localhost:8000/docs`

### Running the Dashboard

```bash
streamlit run dashboard/app.py
```

### Running Tests

```bash
pytest tests/ -v
```

## Features

### Feature Engineering

We engineer 200+ features from 68 source tables using DuckDB for efficient SQL processing:

- **Aggregation features:** Mean, max, min, count, std for numerical columns grouped by case_id
- **Temporal features:** Payment trends, days since last activity, velocity of changes
- **Behavioral features:** Early payment rates, missed payment patterns, credit utilization
- **Bureau features:** External credit history aggregations

Example DuckDB query:
```python
import duckdb

con = duckdb.connect()
df = con.execute("""
    SELECT 
        case_id,
        AVG(amt_payment) as avg_payment,
        MAX(days_past_due) as max_dpd,
        COUNT(*) as num_prev_apps
    FROM read_parquet('data/raw/train_applprev_*.parquet')
    GROUP BY case_id
""").fetchdf()
```

### Model Training

We train an ensemble of three gradient boosting models:

- **LightGBM** – Fast training, handles categorical features natively
- **XGBoost** – Robust, good regularization options
- **CatBoost** – Best for high-cardinality categoricals

Final predictions are a weighted average optimized for the Gini stability metric.

### Explainability

Every prediction includes SHAP explanations:

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Top 5 factors for a single prediction
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X.iloc[0]
))
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Get risk score for a single application |
| `/predict/batch` | POST | Get risk scores for multiple applications |
| `/explain/{case_id}` | GET | Get SHAP explanation for a prediction |
| `/health` | GET | Health check |
| `/metrics` | GET | Model performance metrics |

### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"case_id": 12345, "features": {...}}'
```

Response:
```json
{
    "case_id": 12345,
    "probability": 0.23,
    "risk_category": "Medium",
    "top_factors": [
        {"feature": "payment_history_score", "impact": -0.15},
        {"feature": "debt_to_income", "impact": 0.12},
        {"feature": "previous_defaults", "impact": 0.08}
    ]
}
```

## Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.78+ |
| Gini Stability Score | 0.45+ |
| Precision (at 50% recall) | TBD |
| Recall (at 80% precision) | TBD |
| Inference Latency (p95) | <100ms |

## Project Timeline

| Week | Focus | Status |
|------|-------|--------|
| 1 | Project Setup | ✅ Complete |
| 2 | EDA & Data Understanding | 🔄 In Progress |
| 3 | Feature Engineering | ⬜ Not Started |
| 4 | Model Development | ⬜ Not Started |
| 5 | Optimization & Stability | ⬜ Not Started |
| 6 | MLOps Pipeline | ⬜ Not Started |
| 7 | Deployment | ⬜ Not Started |
| 8 | Documentation | ⬜ Not Started |

## Contributing

This is a capstone project, but feedback is welcome. Feel free to open an issue if you spot something that could be improved.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Home Credit for providing the dataset and hosting the Kaggle competition
- The Kaggle community for starter notebooks and discussions

## Contact

[Venkat Dinesh Pasupuleti]
- GitHub: https://github.com/Venkatdinesh20
- LinkedIn: https://www.linkedin.com/in/venkat-dinesh-s206/
- Email: venkatdinesh60@gmail.com

---

*This project was developed as part of a Masters capstone in Data Science/Machine Learning.*

---
