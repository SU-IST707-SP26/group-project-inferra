# VISION.md

## Current Vision

### Project: INFERRA — Infectious Disease Surge Early Warning System

**Primary Stakeholder:** Australian public health agencies (specifically the Australian Institute of Health and Welfare and state health departments)

**Problem Statement:**
Australian public health agencies currently rely on retrospective surveillance reports that identify outbreaks only after sustained transmission has occurred. When a disease begins surging — whether COVID-19, RSV, or influenza — agencies need early warning to position resources, alert hospitals, and issue public advisories. The critical question is: given an uptick in cases, how early can we predict the surge, and how does prediction accuracy degrade with forecast horizon?

**Solution:**
We build an ML-based early warning system that uses routinely reported surveillance data (WHO COVID-19 weekly reports and Australia's NNDSS fortnightly reports) to:
1. **Classify** whether a disease surge is imminent (surge detection using Rt thresholds)
2. **Predict** future case intensity (regression on cases_per_100k at 2, 4, 6, 8 weeks ahead)
3. **Generalize** across diseases — a model trained on COVID can detect RSV surges

The system uses epidemiologically grounded features: estimated reproduction number (Rt), growth rates, rolling averages, z-scores, lag history, and seasonal encoding. Models are trained globally on 239 countries and tested on held-out Australia (geographic generalization) and on RSV (cross-disease generalization).

**Key Deliverables:**
- Feature-engineered datasets for COVID (WHO) and respiratory diseases (NNDSS)
- Classification and regression models with horizon analysis
- Streamlit dashboard for public health agency use
- Singapore data for external validation

---

## Version History

### Version 1.0 — Initial Vision (2026-02-09)
**Project:** Early Warning System for Pandemic Risk Using WHO Surveillance Data
**Stakeholders:** WHO, national agencies, policymakers, researchers (broad)
**Scope:** Global COVID-19 surveillance across all countries

**Change Reason (2026-03-15):** Based on professor feedback:
1. Narrowed from 4 stakeholders to 1 (Australian public health agencies)
2. Focused on Australia/NZ region instead of global scope
3. Added Rt modeling and cross-disease generalization per professor's suggestion
4. Added regression targets (y_2, y_4, y_6, y_8) for surge size prediction
