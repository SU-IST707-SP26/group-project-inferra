This is an ambitious project with several impressive pieces: the held-out experimental design, the horizon analysis, the custom PDF scraper for NZ virology data, the Streamlit dashboard, and especially the Granger causality analysis. The volume of work for a three-person team was significant and the writeup well-organized. So, while I want to flag the strengths, but there's also a core a methodological limitation that I think is important for you to understand about what your system is actually doing. 

**The limitation**: What your system is essentially doing is computing the time-varying reproduction number R_t and detecting when it crosses a threshold. That's a standard epidemiological surveillance technique — the Wallinga–Lipsitch (2007) method you cite has been in use for nearly two decades. Your ML classifier sits on top of this and learns to recognize "R_t > 1.2 sustained" from features that are mostly R_t itself plus near-equivalent transformations of the same underlying case-count window.

This matters for how your results should be interpreted:

- **"Cross-disease generalization"** is largely the observation that R_t is a disease-agnostic statistic. Any infectious disease that grows exponentially in early phases will have a meaningful R_t, and R_t > threshold will detect surges. That's true by construction, not learned by the model.
- **"Geographic generalization"** is similarly a feature of R_t rather than a feature of the model — R_t computed on Australian case counts has the same units and meaning as R_t computed on French case counts.
- **Horizon analysis** is essentially measuring how predictable R_t is at lead h given recent R_t history. Since R_t is a 5-week smoothed quantity computed from lagged cases, the very short horizons (h=1, h=2) are partly measuring autocorrelation in your engineered feature rather than genuine forward-looking skill. The honest forecasting horizon is probably h=3 or h=4, where the prediction window exceeds the smoothing window.

In other words: your project is essentially doing well-calibrated early surge detection that drops out of the temporal dynamics of SIR-class epidemics. The classifier wrapper is doing very little work beyond what an epidemiologist with the Wallinga–Lipsitch method and a threshold rule would produce. This is not a critique of the engineering — the engineering is good — but it's a critique of the framing, which positions the work as a generalizable ML model when it's better understood as a calibrated wrapper around a known epidemiological statistic.

## On the R_t ablation

Your writeup includes an ablation: "removing R_t reduces AUC from 1.000 to 0.999, a small but meaningful drop showing that lags and z-score carry independent signal." I want to flag this gently because I think it reflects a misunderstanding of where R_t came from in your pipeline.

You engineered Rₜ from the case-count time series via the Wallinga–Lipsitch method. I misunderstood this when we initially spoke - I thought you were this was in the original data. R_t is a deterministic function of the lags — specifically, a scaled exponentiation of the smoothed slope of log-cases over a 5-week window. The other engineered features (week-on-week growth, 4-week growth, z-score) are all functions of the same recent case-count trajectory. They are not independent of R_t, just different encodings of it.

So when you drop R_t and AUC barely moves, that's not because "lags and z-score carry independent signal." It's because the lags and growth features can recover R_t approximately, and the model simply re-learns the same threshold over those features. The ablation as conducted is testing a tautology.

A more informative ablation would have been to drop *all* of {R_t, growth rates, lags} and train on something orthogonal (z-score alone, or seasonal features alone). Or — more interestingly — train on raw case counts only and see whether the model rediscovers SIR-like structure on its own. That would test whether the engineered features carry value beyond the raw inputs they're computed from.

## Other things you could have done...

The genuine ML opportunity in this problem space is in features that are not reducible to recent case counts. Mobility data, contact-network signals, vaccination rates, weather, search trends, social media activity — these are the levers where a learned model can add information beyond what classical SIR-derived statistics already provide. You acknowledge this, and I agree. 

However, acknowledging the challenge in accessing this kind of data, ine direction I had hoped you would explore is whether cross-geographic comparisons could substitute for the missing mobility data. The Granger finding (NSW -> VIC with 4-week lead, p = 0.002) is gesturing in this direction — it implicitly captures a mobility relationship without measuring mobility directly. But it stops at being a univariate statistical test. The next step would have been to operationalize that relationship: learn a network of inter-regional influence weights, treat those as predictive features, and use them to forecast a trailing region's surge from leading regions' Rₜ. That would have been a genuine cross-geographic ML contribution. With NNDSS state-level data this was plausibly within reach. It's reasonable that you may not have had the right scale of data to pull this off across countries, but at the within-Australia state level, this was a real opportunity.

## Strengths

- **Held-out experimental design.** Australia and New Zealand fully excluded from training, used only for evaluation. 
- **Horizon analysis is the right experiment.**  Sliding the label forward and watching AUC degrade is exactly the correct way to separate forecast skill from autocorrelation. 
- **You explicitly acknowledge the h=0 autocorrelation problem.** "AUC ≈ 1.0 reflects autocorrelation… not operationally meaningful." Flagging this is good methodology.
- **"Simpler models generalize better" is a real ML finding.** Tuned RF/XGBoost at AUC 0.536/0.592 vs untuned LR at 0.920 on RSV is a clean bias-variance-across-domains illustration, correctly identified.
- **Threshold recalibration** Recognizing that probability calibration is COVID-specific and needs re-thresholding for RSV is good thinking.
- **Bootstrap CI on the AUS RSV result** (AUC 0.920, [0.809, 0.997]) demonstrates statistical literacy.
- **Granger causality analysis is the strongest empirical contribution.** NSW -> VIC RSV with 4-week lead is genuinely useful for operational planning and is model-independent.
- **Custom PDF scraper for NZ virology data** is non-trivial data engineering.
- **Streamlit dashboard and traffic-light advisory** are nice operational touches that take the work toward something a stakeholder could actually use.

## Weaknesses

- **The R_t ablation tests a tautology** (see discussion above). The conclusion "R_t contributes independent signal" doesn't follow from what you measured.
- **The "cross-disease generalization" claim oversells what the experiment shows.** What's transferring is the feature engineering (R_t_ as a disease-agnostic statistic), not learned model knowledge. The right framing is "epidemiologically-derived features produce disease-agnostic representations, and a thin classifier on top of them transfers as expected."
- **Threshold recalibration is doing more work than the writeup acknowledges.** F1 going from 0.000 to 0.848 isn't a "lightweight calibration step" — it's the difference between an unusable model and a usable one. And it requires labeled disease-specific data, so the system isn't really zero-shot cross-disease.
- **NZ RSV: AUC of 0.567 is barely above random, and the recalibrated F1 of 0.911 is mostly the 76.6% base rate doing the work.** The honest summary is "the cross-disease + cross-country combination is significantly harder, and we don't really have enough baseline data to assess generalization here" — not "data limitation, not a modeling failure." AUC is the right metric to anchor on; F1 with a high positive class prevalence is misleading.
- **AUC < 0.5 at h=8 needs investigation** Anti-correlation with truth at long horizons suggests systematic miscalibration — possibly the model treating "high recent activity" as a positive surge signal when at long horizons this often inverts (peaks followed by lulls). Or perhaps the inclusion of cyclical features to capture seasonality. Worth diagnosing.
- **No baseline comparison.** "Predict surge if R_t > 1.2 right now" would have been the natural baseline. My strong suspicion: the ROC curves are very close to your LR model. Running this comparison would have honestly framed how much the ML adds over the simple rule.
- **Inconsistent surge labels across diseases.** COVID uses R_t > 1.2 sustained; RSV uses z-score > 1.5 sustained. The model trained on one label definition is evaluated on another. The "transfer" is partly the model adapting to a different operational definition, partly disease-agnostic features.
- **The held-out Australia isn't truly epidemiologically isolated.** Your training set includes the UK, Canada, France, Italy, etc. — many countries with COVID dynamics very similar to Australia's. "Generalizing to Australia" is partly the features being universal across similar surveillance systems rather than the model learning country-agnostic patterns from genuinely different data.

## Closing thought

You did a nice job here, and this is really material we didn't cover in class. The methodological observations above aren't "you should have known better" — much of this would only become apparent to someone who's worked in computational epidemiology. Instead, they're intended as "here's what your system actually demonstrates vs. what the writeup claims it demonstrates," and they're worth understanding because they reshape what the next iteration would need to do to add genuine ML value beyond a sophisticated wrapper around a known epidemiological statistic. Strong work overall.

**Score: 28/30**


---

## Final Project Grade
| Assessment Item | Roshni Ramesh More | Jayesh Vinod Sawarkar | Insha Maniyar |
|---|---|---|---|
| **Proposal (5 pts)** | 5 | 5 | 5 |
| **Midterm Report (10 pts)** | 10 | 10 | 10 |
| **Final Presentation (5 pts)** | 5 | 5 | 5 |
| **Final Report (30 pts)** | 28 | 28 | 28 |
| **Weekly Updates (30 pts)** | 30 | 30 | 28 |
| **Total (80 pts)** | **78** | **78** | **76** |
