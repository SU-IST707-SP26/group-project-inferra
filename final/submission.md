# INFERRA — Infectious Disease Surge Early Warning System

## Team

- **Roshni Ramesh More** (GitHub: [RoshniRMore](https://github.com/RoshniRMore))
- **Jayesh Vinod Sawarkar** (GitHub: [jayeshsawarkar](https://github.com/jayeshsawarkar)) — Point of Contact
- **Insha Maniyar** (GitHub: [inshamaniyar77](https://github.com/inshamaniyar77))

## Introduction

Australian public health agencies — including the Australian Institute of Health and Welfare (AIHW) and state health departments — depend on timely surveillance to allocate resources, alert hospitals, and issue public health advisories when infectious diseases begin to surge. Current systems are largely retrospective: they confirm outbreaks after sustained transmission, when intervention windows are narrowing.

INFERRA addresses this gap by building a machine learning-based early warning system trained on WHO COVID-19 global surveillance data. The system detects surge onset using estimated reproduction numbers (Rt) and predicts future case intensity at 2, 4, 6, and 8-week horizons. We evaluate two core questions from our stakeholder's perspective: (1) can a globally-trained model detect surges in a country it has never seen (geographic generalization to Australia), and (2) can patterns learned from one disease transfer to another (cross-disease generalization from COVID-19 to RSV).

Our approach uses epidemiologically grounded features — Rt estimates, growth rates, rolling averages, anomaly z-scores, and seasonal encodings — rather than raw case counts alone. This design choice reflects the professor's feedback to "explicitly model infectiousness (R₀)" and the stakeholder need for interpretable signals that public health professionals can act on.

## Literature Review

<!-- INSHA: Write this section. Cover:
   - Traditional surveillance limitations (retrospective, reporting delays)
   - Growth-based indicators and anomaly detection in outbreak surveillance
   - SIR/SIS compartmental models and R0 estimation methods
   - Prior ML approaches to outbreak prediction
   - Cross-disease transferability of epidemic features
   - Cite 5-8 actual papers/textbooks
   - Justify why we chose gradient boosting + Rt features over pure time series (ARIMA/Prophet)
-->

*[To be completed by Insha]*

## Data and Methods

### Data

<!-- ROSHNI: Write this section. Include:
   - WHO COVID-19 dataset description (rows, columns, time span, countries)
   - Australia NNDSS dataset description (76 diseases, fortnightly, provenance)
   - New Zealand COVID dataset (316 weeks from WHO, same pipeline as Australia)
   - Population data source (World Bank)
   - 2-5 key visualizations:
     1. Global COVID training data overview (239 countries)
     2. Australia COVID timeline with labeled surge periods
     3. Australia RSV fortnightly cases with surge labels
     4. Surge % distribution across WHO regions
     5. Missing data pattern visualization
   - Data quality discussion: reporting delays, under-reporting, policy changes
-->

*[To be completed by Roshni]*

### Methods

<!-- ROSHNI + JAYESH: Write this section together. Cover:

   Feature Engineering:
   - cases_per_100k normalization (why: comparing across countries)
   - Lag features (lag_1..6): captures recent trajectory
   - Rolling means (rm2, rm4, rm8): smooths weekly noise
   - Growth rates (growth_1w, growth_4w): momentum signals
   - Z-score (z12): anomaly detection against 12-week baseline
   - Seasonal encoding (sin/cos of week number): captures annual patterns
   - Rt estimation: log-linear slope method, 4-week window, 5.5-day serial interval

   Surge Labeling:
   - Definition: Rt > 1.2 for 3+ consecutive weeks
   - Why 1.2: represents accelerating transmission above replacement
   - Why 3 weeks: filters out noise from single-week spikes

   Modeling:
   - Classification (surge detection): LogReg, RF, GB, XGBoost
   - Regression (surge size): RF, GB, XGB regressors on y_2/4/6/8
   - Training: 239 countries (Australia held out), SMOTE for class imbalance
   - No CV needed: test sets are entirely unseen geography/disease

   Evaluation:
   - Classification: ROC-AUC (primary), F1, Precision, Recall
   - Regression: MAE, RMSE
   - Horizon analysis: performance at 1, 2, 3, 4, 6, 8 weeks ahead

   Explain WHY each choice was made, not just what.
-->

*[To be completed by Roshni and Jayesh]*

### Supporting Files

| Notebook | Purpose |
|----------|---------|
| `work/data_cleaning_who_covid.ipynb` | Cleans raw WHO COVID-19 CSV: handles NaN, negatives, type conversion, temporal gaps |
| `work/data_biweekly_conversion.ipynb` | Converts weekly WHO data to biweekly for NNDSS alignment |
| `work/data_cleaning_aus_nndss.ipynb` | Processes Australian NNDSS fortnightly Excel reports |
| `work/data_extract_aus_biweekly.ipynb` | Extracts RSV and Measles subsets from merged NNDSS |
| `work/modeling_v1_aus_only.ipynb` | Initial modeling: Australia-only, 80/20 split, 4 classifiers |
| `work/modeling_v2_global_with_regression.ipynb` | Final modeling: global training, Aus/RSV test sets, regression + classification, horizon analysis |
| (NZ extracted from WHO cleaned CSV) | No separate notebook needed — same pipeline as Australia |
| `inferra_dashboard/app.py` | Streamlit dashboard for stakeholder use |

## Results

### Classification: Surge Detection on Australia COVID

The model was trained on 239 countries (74,090 weekly observations) and tested on held-out Australia (310 weeks, 27 surge weeks). Current-period detection achieved near-perfect scores across all models:

| Model | ROC-AUC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| Logistic Regression | 0.995 | 0.964 | 0.931 | 1.000 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 |
| Gradient Boosting | 1.000 | 0.981 | 1.000 | 0.963 |
| XGBoost | 1.000 | 0.941 | 1.000 | 0.889 |

**Important caveat:** AUC ~1.0 at horizon=0 reflects autocorrelation in epidemic data — features like lag_1 and Rt are computed from recent cases, and the surge label is also derived from Rt. This is expected and not operationally useful. The horizon analysis below provides the true measure of early warning capability.

### Horizon Analysis: Timing vs Accuracy Tradeoff

This is the core experiment requested by the professor. As the forecast horizon increases, classification AUC degrades — confirming that earlier predictions are harder but still actionable at 1-2 weeks:

Classification (ROC-AUC) and Regression (MAE in cases_per_100k) both show the expected degradation with horizon, demonstrating the fundamental tradeoff between forecast lead time and prediction accuracy. Random Forest Regressor achieved the best results at all horizons (2-week MAE = 46.41, 4-week MAE = 94.99).

### Cross-Disease Generalization: COVID → RSV

The COVID-trained model was tested on Australia RSV data (43 fortnights, 14 surge periods). Logistic Regression achieved AUC = 0.923, meaning it correctly ranks surge periods above normal periods. However, F1 = 0.000 because the model's probability calibration does not transfer across diseases — all RSV predictions fell below the 0.5 threshold.

After threshold recalibration (optimal threshold search), F1 improved from 0.000 to 0.848, demonstrating that the model captures genuine surge signal across diseases but requires domain-specific calibration for deployment.

### State Propagation Analysis

Cross-correlation analysis of state-level RSV Rt values revealed that NSW leads other states by 2-8 weeks. Key findings:
- NSW → VIC: peak correlation r = 0.835 at 4-week lag
- NSW → SA: peak correlation r = 0.738 at 8-week lag
- Granger causality confirms NSW significantly predicts VIC (p = 0.002), SA (p = 0.001), and WA (p = 0.007)

This establishes NSW as Australia's "canary state" for RSV surveillance — an operationally useful finding for our stakeholder.

### Feature Importance

The top 3 features driving surge prediction are Rt (45.8%), growth_4w (24.5%), and z-score (11.0%). This confirms the professor's recommendation to explicitly model the reproduction number — Rt alone accounts for nearly half of the model's predictive power.

### Prototype Dashboard

A Streamlit dashboard was built for stakeholder use, featuring four pages: situation overview with live Rt metrics and alert banners, early warning signal tracking, state surge propagation maps, and model documentation. The dashboard can be launched via `streamlit run inferra_dashboard/app.py`. A screenshot is available in the supporting notebooks.

## Discussion

The core finding of this project is that surveillance-based features — particularly the estimated reproduction number (Rt), short-term growth rates, and anomaly z-scores — contain sufficient signal to provide meaningful early warning of disease surges. From our stakeholder's perspective (Australian public health agencies), the key insight is the timing-accuracy tradeoff: predictions 1-2 weeks ahead are substantially more reliable than predictions 6-8 weeks ahead. This directly informs how agencies should use such a system — as a near-term alert mechanism rather than a long-range forecasting tool.

The geographic generalization result (training globally, testing on unseen Australia) demonstrates that surge dynamics share common patterns across countries, which is encouraging for deploying such systems in regions with limited local training data. The cross-disease result (COVID-trained model detecting RSV surges) is more nuanced, while AUC scores suggest the model correctly ranks surge periods above normal periods, the probability calibration does not transfer directly, meaning a threshold recalibration step would be needed in production.

These results connect to our stakeholder's needs in concrete ways. An agency monitoring Australia's respiratory disease landscape could use this system to receive early alerts when Rt and growth indicators signal an impending surge, giving them a 1-2 week window to pre-position resources, issue advisories, and prepare healthcare capacity.

## Limitations

<!-- INSHA: Write this section. Cover:
   - Reporting delays and under-reporting in WHO data
   - No subnational/regional granularity (professor asked about regional-level surges)
   - Population normalization incomplete for global training (most countries lack pop data)
   - Rt estimation is a proxy, not true epidemiological R0
   - Cross-disease threshold calibration not solved
   - Limited to respiratory diseases tested so far
   - Only tested on two geographic regions (Australia, New Zealand)
   - No mobility or population density features incorporated
-->

*[To be completed by Insha]*

## Future Work

<!-- INSHA: Write this section. Cover:
   - Add subnational/regional data for "jump" pattern analysis
   - Incorporate mobility and population density features
   - Test on additional diseases (Influenza, Dengue)
   - Add NZ as additional geographic test
   - Validate on additional countries (Singapore, South Korea, Japan)
   - Build automated retraining pipeline
   - Threshold calibration for cross-disease deployment
   - Integration with real-time surveillance feeds
-->

*[To be completed by Insha]*
