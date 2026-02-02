### Title

### Team

### Introduction

### Literature Review

### Data and Methods

#### Data

#### Methods

## Feature Engineering
All features use only data available up to week *t*.

* **Current level:** Cases and deaths per 100k, plus 2–4 week rolling averages to reduce reporting noise.
* **Momentum:** Week-over-week and multi-week growth rates to capture acceleration in spread.
* **Recent history:** Lagged cases and deaths from the past 1–8 weeks to model temporal dependence.
* **Abnormality:** Rolling volatility and z-scores to flag spikes relative to a country’s recent baseline.
* **Seasonality:** Week-of-year encoded cyclically (sin/cos) to account for recurring patterns.
* **Optional enrichment:** Includes GHO indicators such as healthcare capacity proxies.

## Modeling Strategy

**Baselines**
* Rule-based alerts using growth and deviation thresholds.

**Forecasting Models**
* ARIMA / SARIMA
* Prophet

**Surge Risk Classification (Primary Task)**
* **Logistic Regression:** Interpretable baseline.
* **Random Forest:** Robust nonlinear baseline.
* **Gradient Boosting:** Primary candidate due to strong tabular performance.

## Evaluation
* **Forecasting:** Mean Absolute Error (MAE).
* **Surge detection:** Precision, Recall, F1, PR-AUC.
* **Validation:** Rolling / expanding window cross-validation.
* **Thresholds:** Tuned based on stakeholder cost (prioritizing recall with persistence rules).


### Project Plan

| Period | Activity | Milestone |
| :--- | :--- | :--- |
| **Weeks 1–2** | Data ingestion, EDA, baseline metrics | Cleaned COVID-19 dataset, feature definitions |
| **Weeks 3–4** | Feature engineering, rule-based alerts | Baseline early warning signals |
| **Weeks 5–6** | Forecasting and classification models | Initial model comparisons |
| **Weeks 7–8** | Validation, threshold tuning | Final risk classification framework |
| **Weeks 9–10** | Phase 2 planning (non-COVID data) | Expanded data roadmap |


### Risks

