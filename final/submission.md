# INFERRA — Infectious Disease Surge Early Warning System

## Team

- **Roshni Ramesh More** (GitHub: [RoshniRMore](https://github.com/RoshniRMore))
- **Jayesh Vinod Sawarkar** (GitHub: [jayeshsawarkar](https://github.com/jayeshsawarkar)) — Point of Contact
- **Insha Maniyar** (GitHub: [inshamaniyar77](https://github.com/inshamaniyar77))

## Introduction

Infectious disease surges do not announce themselves. By the time hospitals report overcrowding and health agencies confirm an outbreak, the window for early intervention has already closed. INFERRA is built for the people who need to act before that happens, specifically, surge preparedness officers within Australia's Communicable Disease Network (CDNA) and equivalent New Zealand health departments, who are responsible for pre-positioning antivirals, activating hospital surge protocols, and issuing public advisories. Their core need is simple: a reliable signal 1 to 2 weeks before a surge peaks, early enough to act but grounded enough to trust.

To address this, we trained a machine learning early warning system on WHO COVID-19 surveillance data from 238 countries, then tested it on data it had never seen. Australia and New Zealand were fully held out from training. We asked three questions: can the model detect surges in a country it was never trained on, can it transfer from COVID-19 to a completely different disease (RSV), and can it do both simultaneously. We engineered 16 epidemiological features from raw case counts, most importantly the time-varying reproduction number Rₜ, which alone accounts for 42% of the model's predictive power, and evaluated four classifiers alongside regression models predicting surge size at 2, 4, 6, and 8-week horizons.

The results are encouraging. Geographic generalization to both Australia and New Zealand COVID-19 achieves AUC above 0.99. Cross-disease transfer to Australian RSV achieves AUC = 0.920, and the model provides actionable early warning up to 4 weeks ahead before degrading to random. A Granger causality analysis of state-level NNDSS data reveals that NSW RSV surges predict Victorian surges 4 weeks later (p = 0.002), a structural early warning signal that works independently of the model. All findings are operationalized in a live Streamlit dashboard designed for direct stakeholder use.


## Literature Review

Early warning systems for infectious diseases have evolved considerably over the past two decades, moving from simple threshold-based alerts to multi-source data integration using machine learning and artificial intelligence. A 2024 review of global infectious disease early warning models [Hu et al. (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11731462/) identifies the integration of diverse multi-source data with AI techniques as the defining trend in the field, noting that the COVID-19 pandemic substantially accelerated the development and deployment of real-time surveillance frameworks worldwide. Despite this progress, most deployed systems remain disease-specific and country-specific, trained and validated on a single pathogen in a single geographic setting. INFERRA directly addresses this gap by asking whether a model trained globally on COVID-19 can generalize to different diseases and geographies without retraining.

The central epidemiological feature driving our system is the time-varying reproduction number Rₜ-the average number of secondary infections caused by one case at time t. Unlike R₀, which is a fixed biological constant describing transmission in a fully susceptible population, Rₜ varies continuously as population immunity, behaviour, and public health interventions evolve. We estimate Rₜ using the exponential growth rate approach described by [Wallinga and Lipsitch (2007)](https://doi.org/10.1098/rspb.2006.3754), which converts the slope of log-linear case growth into a reproduction number using the serial interval distribution. This method is well-suited to our setting because it requires only a case count time series and can be computed in near real-time without individual-level contact tracing data. The operational interpretation is straightforward, Rₜ > 1 means the epidemic is growing, making it a natural leading indicator of surge onset. This is confirmed in our results, where Rₜ accounts for 42% of feature importance across all models.

The question of cross-disease generalization in epidemic forecasting is relatively underexplored in the literature. Most prior transfer learning work in epidemiology focuses on geographic transfer, applying models trained in one country to another for the same disease, such as influenza forecasting systems trained on US data applied to European settings. Our approach is more ambitious: we transfer from COVID-19 to RSV, two diseases with meaningfully different natural histories, age profiles, and seasonal dynamics. The underlying hypothesis is that the mathematical structure of surge dynamics, exponential growth, Rₜ crossing 1, anomalous z-score elevation above a rolling baseline, is disease-agnostic. The cross-disease AUC of 0.920 for Australian RSV supports this hypothesis. The finding that simpler models generalize better across diseases than tuned ensembles is consistent with the broader machine learning literature on the bias-variance tradeoff in transfer settings, models that memorize training distribution-specific patterns lose the structural generality needed to transfer.

Class imbalance is a known challenge in rare event detection, including epidemic surge identification. In our dataset, surge weeks represent only approximately 7% of all observations. We address this using SMOTE (Synthetic Minority Oversampling Technique), which generates synthetic minority class samples in feature space rather than simply reweighting existing observations. This is preferred over class weighting alone when the minority class boundary is gradual rather than sharp, which is typical in epidemic data where the transition from normal to surge unfolds over several weeks. For the geographic propagation analysis, we apply Granger causality, a time-series statistical test that determines whether past values of one variable improve predictions of another beyond what that variable's own history provides. The finding that NSW Granger-causes VIC RSV surges with a 4-week lead (p = 0.002) is consistent with documented mobility patterns between Sydney and Melbourne and provides an operationally useful independent early warning signal for state-level health departments that does not rely on the machine learning model at all.

## Data and Methods

### Data

INFERRA draws on four publicly available datasets, each serving a distinct role in the pipeline.

WHO COVID-19 Global Surveillance Data

The backbone of our training data is the [WHO COVID-19 Global Surveillance Data](https://data.who.int/dashboards/covid19/data?n=o), which tracks weekly new cases across 240 countries from January 2020 to January 2026. After removing negative counts, filling temporal gaps, and normalizing to cases per 100,000 population, the cleaned dataset contains 75,840 rows. Australia and New Zealand are excluded entirely from training, they appear nowhere in the model's learning process and are used only as held-out test sets. Surge weeks account for approximately 7% of all observations, reflecting how rare sustained outbreak periods are relative to baseline transmission.

Australian NNDSS - RSV Surveillance

The [Australian National Notifiable Diseases Surveillance System (NNDSS)](https://www.cdc.gov.au/resources/publications/national-notifiable-diseases-surveillance-system-nndss-fortnightly-reports-10-23-november-2025) provides fortnightly RSV case counts broken down by state; NSW, VIC, QLD, WA, and SA, covering 2017 to 2024. With 8 years of history and 43 usable fortnights after feature engineering warmup, this is our richest test dataset. It serves two purposes: Test Set 2 for cross-disease generalization, and the basis for the state-level Granger causality analysis that establishes NSW as an early warning signal for Victoria.

PHF Science New Zealand Virology Reports

[PHF Science NZ Digital Library](https://www.phfscience.nz/digital-library/?researchType%5B0%5D=reportItem) publishes weekly national virology surveillance reports covering RSV, Influenza A/B, SARS-CoV-2, rhinovirus, hMPV, and adenovirus. Since these reports are published as PDFs, we built a custom scraper (NZ_Airborne_Disease_Scraper) to extract and consolidate case counts into a structured weekly time series. RSV data spans 112 weeks from April 2023 to December 2025. One important limitation is visible in the data immediately, 76.6% of weeks are labelled as surge, reflecting the unusually large RSV rebound that followed the lifting of COVID restrictions. This short and surge-heavy baseline is a known limitation of the NZ RSV test set and is discussed further in the Limitations section.

World Bank Population Data

[World Bank population estimates](https://data.worldbank.org/indicator/SP.POP.TOTL) provide the 2023 population for each country, used to convert raw case counts into cases per 100,000. This normalization removes country-size bias and makes Australia (26,473,055) and New Zealand (5,223,100) directly comparable. Where country names differed between the WHO and World Bank datasets, a manual mapping dictionary resolved mismatches; the 46 small territories with no population match were assigned the global median population.

Visualizations of the WHO COVID global case distribution, AUS RSV seasonal patterns, NZ RSV baseline coverage, and surge class imbalance across all four test sets are available in [modeling_v2_global_with_regression.ipynb](../work/modeling_v2_global_with_regression.ipynb).

### Methods

#### Feature Engineering

One of the core design decisions in INFERRA was to avoid using raw case counts as model inputs. Case counts tell you how many people are sick today, but they don't tell you whether things are getting better or worse, how fast transmission is accelerating, or whether this week is unusually bad relative to recent history. To capture these dynamics, we engineer 16 features from each country's weekly time series, all computed using only past data so the model never has access to future information.
The first step is normalization, converting raw cases to cases per 100,000 population. This allows direct comparison between a country of 5 million (New Zealand) and a country of 26 million (Australia) without the larger country always appearing to have more severe outbreaks simply because it has more people.

From the normalized series we compute:

Lag features (lag_1 through lag_6): what were case counts 1 to 6 weeks ago, capturing the recent trajectory the epidemic is on
Rolling averages (2, 4, 8-week means): smoothed trends that reduce the impact of single noisy weeks
Rolling volatility (4-week standard deviation): how stable or erratic the current trend is
Growth rates (week-on-week and 4-week): how fast cases are accelerating or decelerating
Z-score anomaly: how unusual this week is compared to the past 12 weeks, a z-score above 1.5 signals something genuinely abnormal
Seasonal encoding (sine and cosine of epi-week): captures annual respiratory disease seasonality without overfitting to specific calendar months
Rₜ (time-varying reproduction number): the single most important feature, accounting for 42% of model decisions. Computed using the Wallinga & Lipsitch (2007) exponential growth rate method with a 5-week sliding window and 5.5-day serial interval. Rₜ > 1 means each infected person is spreading to more than one other person, the epidemic is growing.

The first 6 rows of each series are dropped after feature construction to remove NaN values from lag warmup.

#### Surge Labelling

Before training, we need to tell the model which weeks were surge weeks and which were normal. We use two methods depending on data length. For COVID-19 series (100+ weeks): Rₜ > 1.2 for 3 or more consecutive weeks. For RSV and shorter series: z-score > 1.5 for 2 or more consecutive weeks. Both methods require sustained elevation, a single noisy week above threshold is not counted as a surge. The Rₜ threshold of 1.2 rather than the biological boundary of 1.0 was chosen deliberately to reduce false positives during periods of marginal transmission growth.

#### Handling Class Imbalance

Surge weeks are rare, only about 7% of all training observations are labelled as surges. If left unaddressed, a model could achieve 93% accuracy by simply predicting "not surge" every single week, which is completely useless for early warning. We apply SMOTE (Synthetic Minority Oversampling Technique) to the training data, which generates synthetic surge examples in feature space rather than just copying existing ones. Critically, SMOTE is applied only to the training set, the held-out test sets remain untouched and reflect real-world class imbalance.

#### Classification Models

We train four classifiers on the global COVID training data and evaluate all of them on the held-out test sets without any retraining:

Logistic Regression- the simplest model, L2 regularized. Turned out to be the most transferable across diseases
Random Forest- 200 decision trees, tuned with RandomizedSearchCV (30 iterations, 5-fold stratified cross-validation)
Gradient Boosting- 200 estimators, learning rate 0.05, max depth 4
XGBoost- regularized gradient boosting, also tuned with RandomizedSearchCV

Hyperparameter tuning runs 300 total model fits per tuned model (30 random combinations × 5 folds). The primary metric throughout is ROC-AUC, not accuracy- because accuracy is misleading when one class represents only 7% of the data.

#### Threshold Recalibration

When we applied the COVID-trained model to RSV data, F1 score came back as zero. The model was ranking surge weeks correctly (AUC was fine) but it was never actually predicting surge, because every RSV probability fell below the default threshold of 0.5, which had been calibrated on COVID data. We fixed this by finding the threshold that maximizes F1 for each disease using precision-recall curve optimization. This is a realistic deployment approach, it requires only a small amount of disease-specific validation data and no retraining of the model itself.

#### Regression Models

Beyond detecting whether a surge will happen, we also predict how big it will be. Five regression models predict future cases per 100,000 at 2, 4, 6, and 8-week horizons: Linear Regression, Ridge Regression, Random Forest Regressor, Gradient Boosting Regressor, and XGBoost Regressor. Random Forest Regressor performed best across all horizons. Results are reported in cases per 100,000 to maintain real-world interpretability.

#### Horizon Analysis

The most important experiment in the project. We shift the surge label forward by 1, 2, 3, 4, 6, and 8 weeks and measure how well the model predicts at each horizon. This separates genuine predictive capability from autocorrelation, at h=0, the model is essentially confirming what it already knows from this week's features. The operationally meaningful number is AUC at 4 weeks ahead, because that is the horizon at which a health agency can realistically pre-position resources and issue advisories before a surge peaks.

#### Geographic Propagation Analysis

We compute state-level Rₜ separately for NSW, VIC, QLD, WA, and SA using the same pipeline applied to NNDSS state-level RSV counts. Cross-correlation analysis identifies how many weeks ahead each state pair is correlated. Granger causality tests confirm whether past NSW Rₜ statistically predicts future VIC Rₜ beyond what VIC's own history provides. This analysis is Australia only, New Zealand's PHF Science data is national only with no regional breakdown available.

### Supporting Files

| Notebook | Purpose |
|----------|---------|
| `work/data_cleaning_who_covid.ipynb` | Cleans raw WHO COVID-19 CSV: handles NaN, negatives, type conversion, temporal gaps |
| `work/data_biweekly_conversion.ipynb` | Converts weekly WHO data to biweekly for NNDSS alignment |
| `work/data_cleaning_aus_nndss.ipynb` | Processes Australian NNDSS fortnightly Excel reports |
| `work/data_extract_aus_biweekly.ipynb` | Extracts RSV and Measles subsets from merged NNDSS |
| `work/modeling_v1_aus_only.ipynb` | Initial modeling: Australia-only, 80/20 split, 4 classifiers |
| `work/modeling_v2_global_with_regression.ipynb` | Final modeling: global training, Aus/RSV test sets, regression + classification, horizon analysis |
| `work/nz_airborne_disease_scraper.ipynb` | Scrapes PHF Science PDF reports, extracts and cleans NZ virology data, converts epi-weeks to dates, aggregates influenza subtypes into weekly time series |
| `inferra_dashboard/app.py` | Streamlit dashboard for stakeholder use |

## Results

### Classification: Surge Detection- Australia COVID (Test 1)

The model was trained on 238 countries (75,840 weekly observations) and tested on held-out Australia (310 weeks, 27 surge weeks). All four models achieved strong current-period detection:

| Model | ROC-AUC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| Logistic Regression | 0.995 | 0.964 | 0.931 | 1.000 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 |
| Gradient Boosting | 1.000 | 0.981 | 1.000 | 0.963 |
| XGBoost | 1.000 | 0.962 | 1.000 | 0.927 |

**Important caveat:** AUC ≈ 1.0 at horizon=0 reflects autocorrelation, features like lag_1 and Rₜ are derived from recent cases, and the surge label is also Rₜ-derived. This is expected in epidemic data and not operationally meaningful. The horizon analysis below is the true measure of early warning capability.

### Horizon Analysis: How Early Can We Warn? (AUS COVID)

This is the core experiment. We shift the surge label forward by 1 to 8 weeks and evaluate how well the model predicts at each horizon. Logistic Regression, the best-generalizing model, achieves the following AUC scores:

|   Horizon           |         AUC      |
|---------------------|------------------|
|1 week ahead         |        0.976     |
|2 weeks ahead        |        0.907     |
|3 weeks ahead        |        0.809     |
|4 weeks ahead        |        0.710     |
|6 weeks ahead        |        0.579     |
|8 weeks ahead        |        0.446     |

The model provides actionable early warning from 1 to 4 weeks ahead, AUC stays above 0.7 across this window, which is our defined threshold for reliable prediction. Beyond 4 weeks, stochastic events dominate and AUC degrades toward random. At 8 weeks ahead the model performs below 0.5, meaning prediction is worse than random at that horizon. This degradation is not a modeling failure, it reflects the fundamental limit of epidemic forecasting beyond the serial interval horizon.

The regression models predict surge size at the same horizons. Random Forest Regressor performed best across all horizons, achieving MAE of 46.41 cases per 100,000 at 2 weeks ahead and 94.99 at 4 weeks ahead, results reported in original units to maintain interpretability for stakeholders.

### Cross-Disease Generalization: COVID → RSV (Test 2)

The COVID-trained model was applied without retraining to Australian RSV data (43 fortnights, 14 surge periods). Logistic Regression achieved AUC = 0.920, meaning it correctly ranks surge periods above normal periods 92% of the time despite never having seen RSV data during training.

However, F1 = 0.000 before recalibration. The model's probability outputs are calibrated for COVID density, all RSV predictions fell below the default 0.5 threshold, so the model never predicted surge. After threshold recalibration using precision-recall curve optimization, F1 improved from 0.000 to 0.848, demonstrating that the model captures genuine surge signal across diseases but requires a small domain-specific calibration step for deployment.

A bootstrap validation with 1,000 resamples confirms the RSV result is stable: AUC = 0.920, 95% CI [0.809, 0.997].

An important finding emerged here: tuned Random Forest and XGBoost show substantially lower RSV AUC (0.536 and 0.592 respectively) compared to untuned Logistic Regression (0.920). Deeper, more heavily tuned models overfit COVID-specific patterns during training, losing the structural generality needed to transfer to a different disease. This is a direct illustration of the bias-variance tradeoff across domains, simpler models generalize better.

### Geographic Generalization: New Zealand COVID (Test 3)

New Zealand COVID-19 data was held out from training entirely and used as a second geographic test. The model was applied without retraining to 228 usable weeks (19 surge weeks, 8.3% surge rate):
 
|         Model         |     ROC-AUC      |    F1    |     Precision     |    Recall   | 
|-----------------------|------------------|----------|-------------------|-------------|
|Logistic Regression    |     0.981        |  0.792   |     0.655         |     1.000   |
|Random Forest          |     0.993        |  0.857   |     0.783         |     0.947   |
|Gradient Boosting      |     0.993        |  0.857   |     0.783         |     0.947   |
|XGBoost                |     0.993        |  0.844   |     0.731         |     1.000   |

Gradient Boosting achieved the best AUC at 0.993, nearly identical to the Australia COVID result, confirming that surge dynamics learned from global COVID data transfer cleanly to a new country the model has never seen.

### Cross-Disease + Cross-Country- New Zealand RSV (Test 4)
 
The most challenging evaluation: a COVID-trained model applied to RSV data from a country it never saw. Results on 112 weeks (76.6% surge weeks):

|         Model         |     ROC-AUC      | F1 (default)  |  F1 (recalibrated) |  
|-----------------------|------------------|---------------|--------------------|
|Logistic Regression    |      0.567       |     0.024     |      0.911         |   
|Random Forest          |      0.429       |     0.024     |       --           |   
|Gradient Boosting      |      0.368       |     0.024     |       --           |   
|XGBoost                |      0.295       |     0.024     |       --           |   

AUC of 0.567 for Logistic Regression is weak. This is not a modeling failure, it is a data availability limitation. With 76.6% of weeks labelled as surge, the model has almost no normal weeks to contrast against and cannot learn meaningful boundaries. For context, Australian RSV with an 8-year baseline and more balanced class distribution achieves AUC = 0.920 using the identical model and pipeline. After threshold recalibration, F1 reaches 0.911 with perfect recall (1.000), meaning the model catches every real surge, at the cost of some false positives.

### Geographic Propagation: NSW as Australia's Canary State

State-level Rₜ analysis of NNDSS data reveals that RSV surges propagate geographically across Australia with measurable lead times:
|  State Pair   |  Peak Correlation  |    Lag    | 
|---------------|--------------------|-----------|
| NSW → VIC     |    r = 0.835       |  4 weeks  |    
| NSW → SA      |    r = 0.738       |  8 weeks  | 

Granger causality confirms these relationships are statistically significant:

NSW → VIC: p = 0.002
NSW → SA: p = 0.001
NSW → WA: p = 0.007

NSW consistently leads other states by 2 to 8 weeks, establishing it as Australia's early warning state for RSV. A health officer in Victoria watching NSW Rₜ today has a structural 4-week warning signal, independent of the machine learning model entirely.

### Feature Importance
The three most important features driving surge prediction are Rₜ (42%), 4-week growth rate (24.5%), and z-score anomaly (11%). This confirms that epidemiologically grounded features, particularly the reproduction number, are central to the model's predictive power, not just statistical artifacts of the lag structure.

An ablation study confirms Rₜ contributes genuine signal: removing it reduces AUC from 1.000 to 0.999, a small but meaningful drop showing that lags and z-score carry independent signal and the model is not simply a threshold detector on Rₜ alone.

### Dashboard
A live Streamlit dashboard operationalizes all findings for stakeholder use. It features pages: a situation overview with live Rₜ metrics and surge alert banners for 240 countries; an early warning signal page showing 1 to 4 week surge probability for Australia and New Zealand; and a plain-English traffic light advisory (🟢 Normal / 🟡 Watch / 🔴 Surge) designed for direct use by CDNA surge officers. The live dashboard is accessible at [https://inferra-ihhg8qzykkuu2hojnkslmz.streamlit.app/](https://inferra-ihhg8qzykkuu2hojnkslmz.streamlit.app/).

## Discussion

INFERRA set out to answer a simple but ambitious question: can a model trained on one disease in one part of the world warn us about a completely different disease somewhere else? The answer, with important nuance, is yes. Logistic Regression, the best-generalizing model across all four test sets, achieves AUC = 0.976 at 1 week ahead and 0.710 at 4 weeks ahead, staying above the 0.7 actionable threshold across the full 1 to 4 week window. For a CDNA surge officer, that window is exactly what is needed to pre-position antiviral stockpiles, put hospital surge plans on standby, and prepare public communications before beds start filling. Beyond 4 weeks the model loses confidence, not because of any flaw in the approach, but because epidemics are genuinely unpredictable that far out. No model can read a superspreader event or a new variant from a case count time series months in advance.

The cross-disease result is the heart of the project. Logistic Regression achieves AUC = 0.920 on Australian RSV despite never having seen a single RSV data point during training, confirming that surge dynamics, exponential growth, Rₜ rising above 1, anomalous z-score elevation, follow a common mathematical pattern regardless of the specific virus. One important caveat is that probability calibration does not transfer automatically. The model ranked surge weeks correctly but never predicted surge at the default 0.5 threshold because RSV produces systematically lower probabilities than COVID. A threshold recalibration using PR curve optimization fixed this, bringing AUS RSV F1 from 0 to 0.848 and NZ RSV F1 to 0.911, a one-time lightweight step that requires no retraining. Unexpectedly, simpler models generalized better: Random Forest and XGBoost after 300 model fits of tuning achieved RSV AUC of only 0.536 and 0.592, far below untuned Logistic Regression at 0.920. This is the bias-variance tradeoff across domains, models optimized for COVID patterns lose the structural generality needed to transfer to a different disease.

The Granger causality analysis adds something the machine learning model alone cannot provide, a structural early warning signal that requires no model inference at all. NSW RSV surges predict Victorian surges 4 weeks later (p = 0.002), SA surges with p = 0.001, and WA surges with p = 0.007. A health officer in Melbourne watching NSW Rₜ today has a free 4-week warning built into the geographic structure of how RSV moves across Australia. New Zealand RSV is the weakest result at AUC = 0.567, but the cause is not model failure, with 76.6% of weeks labelled as surge due to the post-COVID rebound, the model has almost no normal baseline to learn from. The identical pipeline on Australian RSV with 8 years of balanced data achieves 0.920. This is a data availability constraint, not a generalization failure.

Taken together, the results connect directly to what our stakeholder needs. A CDNA surge officer monitoring Australia and New Zealand's respiratory disease landscape can use INFERRA to receive a reliable early alert 1 to 4 weeks before a surge peaks, with the NSW Granger signal providing an additional structural layer of warning for Victorian health officers specifically. The system requires no retraining for new diseases, only a lightweight threshold adjustment, making it practical to deploy and maintain in a real public health setting.

## Limitations

No model is perfect, and INFERRA is no exception. The most significant limitation is the NZ RSV baseline problem. PHF Science virology reports only became available from 2022, right after COVID restrictions lifted, during an unusually large RSV rebound across the Southern Hemisphere. With 76.6% of weeks labelled as surge, the model has almost no normal weeks to contrast against. This is not something we can fix by changing the model or the features. It is simply a data availability problem that will resolve itself as more years of NZ virology data accumulate beyond the rebound period.

Our Rₜ estimates are a proxy, not a direct epidemiological measurement. We compute Rₜ from the slope of case counts using a fixed serial interval of 5.5 days. In reality, serial intervals vary by variant, testing rates affect reported case counts, and reporting delays mean the most recent weeks are often incomplete. The Rₜ values we produce are consistent and comparable across countries and time periods, but they should not be interpreted as precise real-world measurements, they are a useful signal, not a ground truth.

The system has only been tested on two countries; Australia and New Zealand, both high-income Southern Hemisphere countries with strong surveillance infrastructure and consistent reporting. Whether the model generalizes to lower-income countries with weaker surveillance, irregular reporting, or different disease profiles is genuinely unknown. The geographic propagation analysis is even more limited, it only exists for Australia because NNDSS breaks down by state. New Zealand has no regional breakdown available in PHF Science data, so no equivalent sub-national analysis was possible for NZ.

Finally, while threshold recalibration successfully recovered F1 scores for RSV, it does require a small amount of disease-specific validation data to work. The system is not fully plug-and-play for a brand new disease with zero prior data. And across all test sets, the model does not incorporate any mobility data, population density, or healthcare capacity features, all of which a real-world surge officer would consider alongside case counts. These are meaningful gaps between what INFERRA currently does and what a fully operational early warning system would need.

## Future Work

The most natural next step is extending the test set to non-respiratory diseases. Dengue and measles are the strongest candidates, both have well-documented surge dynamics, global WHO surveillance data, and fundamentally different transmission characteristics from respiratory viruses. Testing whether a COVID-trained model can detect a dengue surge in a tropical country would be the ultimate stress test of cross-disease generalization, and would tell us whether the transferable signal we found is specific to respiratory viruses or genuinely disease-agnostic.

The NZ RSV result will improve on its own over time. As PHF Science accumulates more years of virology data beyond the 2022 post-COVID rebound, the surge-to-normal ratio will balance out and the test set will become more representative. Re-evaluating Test 4 in a year or two, with a fuller baseline, would give a much more honest picture of cross-country RSV generalization than the current data allows.

On the modeling side, incorporating mobility data is the most promising avenue for improvement. The Granger finding that NSW leads Victoria by 4 weeks is consistent with travel patterns between Sydney and Melbourne, but the model itself does not use any mobility information at all. Explicitly including airline passenger volumes or internal travel flows could sharpen the geographic propagation signal and potentially extend it to New Zealand, where no state-level breakdown currently exists.

Finally, a third country would meaningfully strengthen the cross-country generalization claim. Singapore was originally in scope for this project before being dropped per professor feedback. Adding it back, or choosing another country with a different surveillance system, disease profile, and hemisphere, would test whether the two-country result holds more broadly and move INFERRA closer to a genuinely global early warning system.
