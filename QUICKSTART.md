# Quick Start Guide

## Setup (First Time)

### 1. Clone Repository
```bash
git clone https://github.com/Venkatdinesh20/Capstone_Project.git
cd home-credit-credit-risk-model-stability
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Data
- Go to [Kaggle Competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data)
- Download the dataset
- Extract to `csv_files/` or `parquet_files/` directories

---

## Daily Workflow

### Option 1: Interactive Notebooks (Recommended for Exploration)

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_data_quality.ipynb
# 3. notebooks/03_feature_engineering.ipynb
# 4. notebooks/04_model_training.ipynb
# 5. notebooks/05_model_evaluation.ipynb
```

### Option 2: Run Scripts (Automated)

```bash
# Data quality analysis
python scripts/data_quality_analysis.py

# Missing data analysis
python scripts/missing_data_analysis.py

# Full pipeline (coming soon)
python scripts/run_pipeline.py
```

### Option 3: Python API

```python
# In a Python script or notebook
from src.data import DataLoader, DataMerger, DataPreprocessor
from config import *

# Load data
loader = DataLoader(data_type='train', file_format='parquet')
base_df = loader.load_base_table()

# Merge tables
merger = DataMerger(data_type='train')
full_df = merger.merge_all_tables()

# Preprocess
preprocessor = DataPreprocessor()
clean_df = preprocessor.preprocess_pipeline(full_df, fit=True)
```

---

## Common Tasks

### Check Configuration
```python
python config.py
```

### Load and Explore Data
```python
from src.data import load_sample_data
df = load_sample_data(n_rows=1000)
print(df.head())
```

### Get Dataset Summary
```python
from src.data import get_dataset_summary
summary = get_dataset_summary(data_type='train')
print(summary)
```

### Create Visualizations
```python
from src.visualization import plot_target_distribution, plot_missing_values

# Plot target
plot_target_distribution(df)

# Plot missing values
plot_missing_values(df, top_n=20)
```

---

## Project Structure Reference

```
home-credit-credit-risk-model-stability/
├── config.py                      # All configuration settings
├── requirements.txt               # Python dependencies
├── README.md                      # Main documentation
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_quality.ipynb
│   └── ...
│
├── src/                          # Source code
│   ├── data/                     # Data utilities
│   │   ├── loader.py            # Load data
│   │   ├── merger.py            # Merge tables
│   │   └── preprocessor.py      # Clean data
│   │
│   ├── features/                # Feature engineering
│   ├── models/                  # Model training
│   └── visualization/           # Plotting
│
├── scripts/                     # Executable scripts
│   ├── data_quality_analysis.py
│   └── missing_data_analysis.py
│
└── outputs/                     # Generated outputs
    ├── figures/                 # Plots
    ├── reports/                 # Analysis reports
    └── predictions/             # Model predictions
```

---

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the project root
cd d:\capestone2\home-credit-credit-risk-model-stability

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Memory Errors
```python
# Use chunking for large files
# Or work with a sample first
from src.data import load_sample_data
df = load_sample_data(n_rows=10000)
```

### Path Errors
```python
# Always use absolute paths from config
from config import ROOT_DIR, DATA_DIR
print(f"Root: {ROOT_DIR}")
print(f"Data: {DATA_DIR}")
```

---

## Team Workflow

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, then commit
git add .
git commit -m "feat: description of changes"

# Push to GitHub
git push origin feature/your-feature-name

# Create Pull Request on GitHub
```

### Before Committing
1. Run your code to ensure it works
2. Add docstrings to functions
3. Update documentation if needed
4. Don't commit large data files (.csv, .parquet)

---

## Resources

- **Jira Board**: https://venkatdinesh60.atlassian.net/jira/core/projects/CP/board
- **GitHub**: https://github.com/Venkatdinesh20/Capstone_Project
- **Kaggle**: https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability
- **Documentation**: See `docs/` folder

---

## Next Steps

1. ✅ Complete setup (you're here!)
2. 📊 Run `notebooks/01_data_exploration.ipynb`
3. 🔍 Run `scripts/data_quality_analysis.py`
4. 🧹 Start data preprocessing
5. 🚀 Build your first model

**Questions?** Check the documentation in `docs/` or ask the team!
