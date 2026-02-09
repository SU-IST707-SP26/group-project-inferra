### Title

### Team

### Introduction

### Literature Review

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
## A: Australia (NNDSS Fortnightly Reports)

### Dataset Overview  
**Dataset Name:** National Notifiable Diseases Surveillance System (NNDSS) – Fortnightly Reports (Merged Dataset)  
**Source:** Australian Government Department of Health and Aged Care  
**Data Link:** NNDSS Public Reports Collection :contentReference[oaicite:0]{index=0}  
**Merged File Used:** `merged_nndss_all_diseases.xlsx`

### Dataset Structure  
- **Rows:** 76 diseases  
- **Columns:** 50 total  
  - 1 column: `Disease_Name`  
  - 49 columns: Fortnightly reporting periods  
- **Time Period Covered:** December 2023 – November 2025  
- **Temporal Resolution:** Fortnightly (bi-weekly aggregated case counts)

**Column Format:**  
- `Disease_Name` (categorical)  
- `FN##_YEAR` (e.g., FN01_2024, FN02_2024, … FN25_2025)

**Feature Types:**  
- Disease names: Categorical (string)  
- Case counts: Numerical (integer)

### Diseases Covered  
The dataset includes 76 nationally notifiable diseases across Australia, including:

- **Bloodborne:** Hepatitis B, Hepatitis C, Hepatitis D, HIV  
- **Gastrointestinal:** Campylobacteriosis, Salmonellosis, Listeriosis, Cryptosporidiosis  
- **Vaccine-preventable:** Measles, Mumps, Pertussis, Influenza  
- **Vector-borne:** Dengue, Ross River virus, Barmah Forest virus  
- **Respiratory:** COVID-19, Influenza, Legionellosis  
- **Sexually transmitted infections:** Chlamydia, Gonorrhoea, Syphilis  

### Data Provenance & Reliability  

- **Official Government Source:**  
  - Managed by the Australian Government Department of Health and Aged Care  
  - Part of the National Notifiable Diseases Surveillance System (NNDSS)  
  - Legally mandated under the National Health Security Act 2007  

- **Collection Process:**  
  1. Medical practitioners and laboratories report cases to state/territory health authorities  
  2. States and territories validate and submit data to the national NNDSS system  
  3. The Department of Health aggregates and publishes fortnightly reports  
  4. Data is de-identified to protect patient privacy  

- **Reliability Indicators:**  
  - Authoritative national surveillance system  
  - Standardized disease definitions (ICD-based classifications)  
  - Multi-level quality control and validation  
  - Regular, consistent reporting schedule  
  - Transparent metadata and documentation  

### Metadata Availability  
Each original NNDSS fortnightly report includes:  
- Reporting period dates  
- State/territory-level breakdowns  
- Disease definitions and classifications  
- Data collection methodology  
- Interpretation notes and caveats  
- Historical comparison metrics (e.g., YTD, rolling averages)

### Data Completeness  

- **2024 Coverage:** 25 of 26 fortnights (~96% complete)  
- **2025 Coverage:** 24 of 26 fortnights (~92% complete)  
- **Known Missing Fortnights:** FN26_2024, FN25_2025, FN26_2025  
- **Handling Missing Data:** Missing values may appear as 0 or can be treated as NA depending on analysis approach  

### Geographic Coverage  

- Nationwide aggregated data across all Australian states and territories  
- Original source files include state-level breakdowns (ACT, NSW, NT, QLD, SA, TAS, VIC, WA)  

---

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

| Risk | Mitigation |
|-----|-----------|
| Reporting bias | Growth-based features and normalization |
| Inconsistent reporting | Weekly smoothing and persistence rules |
| Model overfitting | Time-aware cross-validation |
| Ambiguous outbreak definitions | Transparent risk categories |

These challenges strengthen the project by encouraging cautious, ethical interpretation.

### One-Sentence Summary
This project transforms WHO surveillance data into a predictive early warning system that identifies pandemic outbreak risk before cases surge.
