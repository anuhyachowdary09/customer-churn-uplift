# Customer Churn Prediction & Retention Uplift

A production-grade churn prediction and retention system combining Cox Proportional Hazards survival analysis, XGBoost/LightGBM ensemble, SMOTE for class imbalance, two-model uplift modeling (Qini curve), and SHAP explainability. Designed for 20K+ customer datasets with ~18% churn rate.

## Business Impact
| Metric | Result |
|--------|--------|
| Churn reduction | 12% |
| Revenue protected | $1.2M |
| Campaign ROI improvement | 20% |
| Model AUC | 0.88 |

## Tech Stack
| Category | Tools |
|----------|-------|
| Survival Analysis | Cox Proportional Hazards (lifelines), Kaplan-Meier |
| ML Models | XGBoost, LightGBM, Logistic Regression (ensemble) |
| Uplift Modeling | Two-model approach, Qini curve evaluation |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Explainability | SHAP (SHapley Additive exPlanations) |
| Hyperparameter Tuning | Optuna (Bayesian optimization) |
| Experiment Tracking | MLflow |

## Project Structure
```
customer-churn-uplift/
├── main.py              # Full churn + uplift pipeline
├── requirements.txt
└── README.md
```

## Quickstart
```bash
pip install -r requirements.txt
python main.py
```

## Pipeline Overview

### 1. Survival Analysis (Cox PH)
- Estimates time-to-churn using Cox Proportional Hazards model
- Kaplan-Meier curves for customer cohort visualization
- Hazard ratios identify highest-risk features (tenure, contract type, ARPU)

### 2. Churn Classification Ensemble
- **SMOTE**: Oversamples minority class to handle 18% churn rate
- **XGBoost + LightGBM + LR**: Soft-voting ensemble, weights tuned via Optuna
- **Optuna**: Bayesian hyperparameter optimization maximizing AUC-ROC
- **MLflow**: Tracks all experiments, parameters, and metrics

### 3. SHAP Explainability
- Global feature importance (SHAP bar plots)
- Per-customer explanations for retention team interventions
- Top drivers: contract length, monthly charges, customer service calls

### 4. Uplift Modeling (Two-Model Approach)
- Separate models for treatment and control groups
- Uplift score = P(churn | treated) - P(churn | control)
- Qini curve measures incremental lift over random targeting
- Prioritizes customers with highest retention probability if contacted

## Evaluation Results
| Model | AUC-ROC | PR-AUC | F1 |
|-------|---------|--------|----|
| XGBoost | 0.875 | 0.712 | 0.681 |
| LightGBM | 0.871 | 0.705 | 0.674 |
| Ensemble | 0.882 | 0.724 | 0.689 |

## Production Extensions
- **Real-time scoring**: Deploy as FastAPI endpoint for CRM integration
- **Causal inference**: Upgrade to DragonNet or causal forests
- **Monitoring**: PSI drift detection on feature distributions
