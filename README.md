# Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-3.x-green)
![Sklearn](https://img.shields.io/badge/ScikitLearn-1.x-orange)
![SMOTE](https://img.shields.io/badge/SMOTE-Imbalanced--Learn-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Overview

Binary classification project that predicts whether a telecom customer will churn, using the IBM Telco Customer Churn dataset. The project covers the full ML workflow: EDA, feature engineering, class imbalance handling, model comparison, and serialized model artifacts.

## Dataset

**IBM Telco Customer Churn**
- 7,043 customer records, 21 features
- Target: `Churn` (Yes / No)
- Class distribution: 73.4% No Churn, 26.6% Churn

Features include contract type, tenure, monthly charges, internet service type, payment method, and various add-on service subscriptions.

## Key EDA Findings

- **Month-to-month contracts** have the highest churn rate — customers on annual/two-year contracts rarely churn
- **Short-tenure customers (0–12 months)** churn at a far higher rate than long-tenure ones
- **Fiber optic** customers churn more than DSL customers despite paying more
- `tenure`, `MonthlyCharges`, and `Contract` are the strongest predictors

## Methodology

1. **Preprocessing** — `TotalCharges` cast to float (11 blank rows dropped), binary columns label-encoded, multi-category columns one-hot encoded
2. **Feature engineering** — `avg_monthly_charges = MonthlyCharges / (tenure + 1)` to capture early high-spend customers
3. **Scaling** — `StandardScaler` fit on training data only, then applied to test data (`transform`, not `fit_transform`)
4. **Class imbalance** — SMOTE applied to training set only (4130 → 4130 per class)

## Model Performance

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~0.74 | ~0.80 |
| Random Forest | 0.72 | 0.818 |
| **XGBoost** | **0.78** | **0.837** |

XGBoost is the best-performing model and is saved as the final artifact.

## Project Structure

```
Customer_churn/
├── Churn.ipynb              # Full notebook: EDA → feature engineering → modelling
├── models/
│   ├── xgb_churn_model.pkl  # Serialized XGBoost model
│   └── scaler.pkl           # Fitted StandardScaler
```

## Running the Notebook

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
jupyter notebook Churn.ipynb
```

Run all cells top-to-bottom. The dataset is loaded directly from a public GitHub URL — no local CSV required.

## Loading the Saved Model

```python
import joblib
import pandas as pd

model = joblib.load('models/xgb_churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# X must be preprocessed (encoded + feature-engineered) before scaling
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)
```

## Tools Used

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn, Imbalanced-learn (SMOTE)
- XGBoost

## ⚠️ Disclaimer

For educational purposes only. Not intended for production use without further validation.

---

**Author: Vishal Lohchab**
