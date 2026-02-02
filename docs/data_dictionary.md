# Feature Definitions Data Dictionary

This file is provided by the Kaggle competition and contains descriptions of features across all tables.

For the actual feature definitions, please refer to:
- `feature_definitions.csv` in the root directory
- Kaggle competition data page: https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data

## Common Feature Patterns

### Numerical Features
- Features ending in `_A`: Aggregated statistics (mean, sum, etc.)
- Features ending in `_M`: Maximum values
- Features ending in `_sum`: Sum aggregations
- Features ending in `_mean`: Mean aggregations

### Categorical Features
- Features ending in `_type`: Category indicators
- Features starting with `is_`: Binary flags

### Temporal Features
- `date_decision`: Date of credit decision
- Features with `days_`: Time periods in days
- Features with `months_`: Time periods in months

## Key Tables

### Base Table
- `case_id`: Unique identifier for each application
- `target`: Default indicator (0 = No default, 1 = Default)
- `date_decision`: Date of decision
- `WEEK_NUM`: Week number for temporal split

### Static Tables
- `static_0_*`: Static applicant information
- `static_cb_0`: Credit bureau static data
- `person_1`, `person_2`: Personal information

### Dynamic Tables
- `applprev_*`: Previous application history
- `credit_bureau_a_*`, `credit_bureau_b_*`: Credit bureau history
- `debitcard_1`: Debit card transaction data
- `deposit_1`: Deposit account data
- `other_1`: Other financial products
- `tax_registry_*`: Tax registry information

## Feature Naming Convention

```
[source]_[category]_[aggregation]_[suffix]
```

Examples:
- `applprev_1_0_amount_mean`: Mean amount from application previous table 1_0
- `credit_bureau_a_2_5_days_max`: Maximum days from credit bureau A table 2_5
- `static_0_0_income_annual`: Annual income from static table 0_0

## Missing Value Indicators

When preprocessing, missing indicator flags are created:
```
[original_column]_missing
```

Example:
- `income_annual` → `income_annual_missing` (1 if missing, 0 otherwise)

## Data Types

- **int64/float64**: Numerical features
- **object**: Categorical features (will be encoded)
- **datetime64**: Temporal features
- **category**: Optimized categorical features

---

**Note**: For detailed feature descriptions, always refer to the official `feature_definitions.csv` file provided with the dataset.
