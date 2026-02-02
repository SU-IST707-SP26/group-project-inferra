### Title
Early Warning System for Pandemic Risk Using WHO Surveillance Data

### Team

Roshni Ramesh More (GitHub: [ RoshniRMore](https://github.com/RoshniRMore))

Jayesh Vinod Sawarkar (GitHub: [ jayeshsawarkar](https://github.com/jayeshsawarkar)) — Point of Contact (POC)

Insha Maniyar (GitHub id:[ inshamaniyar77](https://github.com/inshamaniyar77))

### Introduction

We aim to build an early warning system that uses routinely reported public health data to identify whether disease spread is accelerating in a way that may signal a future pandemic. Instead of reacting after outbreaks peak, this system focuses on detecting early signs of rapid growth.
Most existing systems emphasize retrospective reporting. Our approach reframes surveillance data as a forward-looking risk prediction problem, using growth patterns, trends, and anomalies to estimate short-term outbreak risk before cases surge.
International organizations, governments, and public health agencies rely on early signals to allocate resources, issue advisories, and prepare healthcare systems. Earlier detection can reduce preventable illness, deaths, and economic disruption.

### Literature Review

Current Practice and Limitations
Traditional disease surveillance systems rely on reported case counts and historical summaries, which often identify outbreaks only after sustained transmission has occurred. Recent research highlights the importance of growth-based indicators and anomaly detection for outbreak detection, particularly when reporting frequency varies across countries. However, many systems remain descriptive rather than predictive.

Stakeholders and Their Needs

    ____________________________________________________________________________________________________________
              Stakeholder                   |      Why They Care          |           Key Needs
    _____________________________________________________________________________________________________________
      WHO emergency response teams          |  Global outbreak monitoring | Early detection of abnormal growth
      National public health agencies       |   Domestic preparedness     | Country-level risk signals
      Policymakers                          |      Resource allocation    | Clear, interpretable risk categories 
      Researchers                           |   Epidemiological insight   | Reproducible trend and surge analysis
____________________________________________________________________________________________________________________


### Data and Methods

### Data

#### Phase 1: COVID-19 Surveillance Data (Current Focus)
**Source:** WHO COVID-19 Dashboard  
🔗 https://data.who.int/dashboards/covid19/data  

**Dataset:** WHO COVID-19 Global Daily Data (CSV)  
**Unit of analysis:** Country × Week (after aggregation)

**Raw Data (Daily Level):**
- **Rows:** ~1.2 million records (country × date)
- **Columns:** 8
- **Time span:** January 2020 – January 4, 2026
- **Geographic coverage:** ~200 countries and territories

**Raw columns:**
- `Date_reported`
- `Country_code`
- `Country`
- `WHO_region`
- `New_cases`
- `Cumulative_cases`
- `New_deaths`
- `Cumulative_deaths`

**Unit of analysis after preprocessing:** Country × Week

**Strengths:**
- Globally standardized
- Longitudinal coverage
- Suitable for weekly modeling

**Limitations:**
- Reporting delays and revisions
- Under-reporting in some regions
- Changes in testing and reporting policies over time

---

#### Phase 2: Country-Specific Reporting for Other Diseases (Planned / In Progress)
**Source:** WHO Global Health Observatory (GHO)  
🔗 https://www.who.int/data/gho  

This phase will incorporate communicable disease indicators, mortality metrics, and health system proxies to generalize the early warning framework beyond COVID-19.

---

### Methods

#### Feature Engineering
All features use only data available up to week *t*.

* **Current level:** Cases and deaths per 100k, plus 2–4 week rolling averages to reduce reporting noise.
* **Momentum:** Week-over-week and multi-week growth rates to capture acceleration in spread.
* **Recent history:** Lagged cases and deaths from the past 1–8 weeks to model temporal dependence.
* **Abnormality:** Rolling volatility and z-scores to flag spikes relative to a country’s recent baseline.
* **Seasonality:** Week-of-year encoded cyclically (sin/cos) to account for recurring patterns.
* **Optional enrichment:** Includes GHO indicators such as healthcare capacity proxies.

#### Modeling Strategy

**Baselines**
* Rule-based alerts using growth and deviation thresholds.

**Forecasting Models**
* ARIMA / SARIMA
* Prophet

**Surge Risk Classification (Primary Task)**
* **Logistic Regression:** Interpretable baseline.
* **Random Forest:** Robust nonlinear baseline.
* **Gradient Boosting:** Primary candidate due to strong tabular performance.

#### Evaluation
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

| Risk | Mitigation |
|-----|-----------|
| Reporting bias | Growth-based features and normalization |
| Inconsistent reporting | Weekly smoothing and persistence rules |
| Model overfitting | Time-aware cross-validation |
| Ambiguous outbreak definitions | Transparent risk categories |

These challenges strengthen the project by encouraging cautious, ethical interpretation.

### One-Sentence Summary
This project transforms WHO surveillance data into a predictive early warning system that identifies pandemic outbreak risk before cases surge.
