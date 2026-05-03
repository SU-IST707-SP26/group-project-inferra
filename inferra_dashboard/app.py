"""
Inferra — Infectious Disease Surge Early Warning System
4 pages:
  Page 1 — Technical Summary (professor / technical)
  Page 2 — Disease Surveillance (COVID dropdown + AUS RSV)
  Page 3 — New Zealand (Tests 3 & 4)
  Page 4 — Public Advisory (plain English, traffic light)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Inferra — Disease Surge Early Warning",
    page_icon="\U0001f9a0",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#003366}
[data-testid="stSidebar"] *{color:#fff !important}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,0.2)}
.who-bar{background:#003366;color:#fff;padding:.75rem 1.5rem;border-radius:8px;margin-bottom:1rem}
.who-bar h1{color:#fff;margin:0;font-size:20px}
.who-bar p{color:rgba(255,255,255,.75);margin:3px 0 0;font-size:13px}
.kpi{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:.75rem 1rem;text-align:center}
.kpi-label{font-size:11px;color:#666;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.kpi-value{font-size:24px;font-weight:700;color:#003366}
.kpi-sub{font-size:11px;color:#999;margin-top:2px}
.sec{font-size:12px;font-weight:700;color:#003366;text-transform:uppercase;letter-spacing:.06em;
     border-bottom:2px solid #003366;padding-bottom:3px;margin:1.2rem 0 .6rem}
.box-blue{background:#e3f2fd;border-radius:6px;padding:10px 14px;font-size:13px;color:#0d47a1;margin:.4rem 0}
.box-green{background:#e8f5e9;border-left:4px solid #2e7d32;border-radius:4px;padding:10px 14px;font-size:13px;color:#1b5e20}
.box-amber{background:#fff8e1;border-left:4px solid #f57f17;border-radius:4px;padding:10px 14px;font-size:13px;color:#e65100}
.box-red{background:#ffebee;border-left:4px solid #c62828;border-radius:4px;padding:10px 14px;font-size:13px;color:#b71c1c}
</style>
""", unsafe_allow_html=True)

DATA      = Path(__file__).parent / "data"
REPO_DATA = Path(__file__).parent.parent / "Data"

def find_file(filename):
    if (DATA / filename).exists():
        return DATA / filename
    if (REPO_DATA / filename).exists():
        return REPO_DATA / filename
    raise FileNotFoundError(f"{filename} not found in {DATA} or {REPO_DATA}")

@st.cache_data
def load_base():
    rsv_nat  = pd.read_csv(DATA/"rsv_national.csv",    parse_dates=["Date"])
    rsv_st   = pd.read_csv(DATA/"rsv_state.csv",       parse_dates=["Date"])
    rsv_pred = pd.read_csv(DATA/"rsv_predictions.csv", parse_dates=["Date"])
    covid    = pd.read_csv(DATA/"covid_national.csv",  parse_dates=["Date"])
    horizon  = pd.read_csv(DATA/"horizon_results.csv")
    feat_imp = pd.read_csv(DATA/"feature_importance.csv")
    return rsv_nat, rsv_st, rsv_pred, covid, horizon, feat_imp

@st.cache_data
def load_who():
    who = pd.read_csv(find_file("WHO_COVID19_cleaned.csv"), parse_dates=["Date_reported"])
    pop = pd.read_csv(find_file("population.csv"))
    pop_latest = pop.sort_values("Year").groupby("Country Name")["Value"].last().reset_index()
    pop_latest.columns = ["Country","Population"]
    return who, pop_latest

@st.cache_data
def load_nz():
    nz_covid = pd.read_csv(DATA/"nz_covid_national.csv", parse_dates=["Date"])
    nz_rsv   = pd.read_csv(DATA/"nz_rsv_national.csv",   parse_dates=["Date"])
    return nz_covid, nz_rsv

@st.cache_resource
def load_model():
    with open(DATA/"best_model.pkl","rb") as f:
        return pickle.load(f)

rsv_nat, rsv_st, rsv_pred, covid, horizon, feat_imp = load_base()
bundle = load_model()
MODEL      = bundle["model"]
FEATS      = bundle["feature_cols"]
MODEL_NAME = bundle.get("model_name", "Random Forest")
THEME  = dict(plot_bgcolor="white", paper_bgcolor="white",
              font=dict(family="Arial", size=12, color="#333"),
              margin=dict(l=45, r=25, t=45, b=45))
STATE_COLORS = {"NSW":"#1f77b4","VIC":"#d62728","QLD":"#ff7f0e","WA":"#2ca02c","SA":"#9467bd"}

def trim_trailing_zeros(df, case_col="Cases"):
    last_nonzero = df[df[case_col] > 0].index.max()
    if pd.isna(last_nonzero):
        return df
    return df.loc[:last_nonzero]

@st.cache_data(show_spinner=False)
def build_features(dates, cases, population):
    df = pd.DataFrame({"Date":pd.to_datetime(dates),"Cases":cases}).sort_values("Date").reset_index(drop=True)
    df["Cases"] = df["Cases"].clip(lower=0)
    df["cpp"] = (df["Cases"]/population)*100000 if population else df["Cases"]
    f = df["cpp"]
    for lag in [1,2,3,4,5,6]: df[f"lag_{lag}"] = f.shift(lag)
    df["roll2_mean"]=f.rolling(2).mean(); df["roll4_mean"]=f.rolling(4).mean()
    df["roll4_std"]=f.rolling(4).std().replace(0,np.nan); df["roll8_mean"]=f.rolling(8).mean()
    df["roll12_mean"]=f.rolling(12).mean(); df["growth_1w"]=f/f.shift(1).replace(0,np.nan)
    df["growth_4w"]=f/f.shift(4).replace(0,np.nan)
    df["zscore"]=(f-df["roll12_mean"])/df["roll4_std"]
    wn=df["Date"].dt.isocalendar().week.astype(int)
    df["week_sin"]=np.sin(2*np.pi*wn/52); df["week_cos"]=np.cos(2*np.pi*wn/52)
    fc=f.clip(lower=0.001); rt=[np.nan]*4
    for i in range(4,len(df)):
        wc=fc.iloc[i-4:i+1].values; slope=np.polyfit(np.arange(5),np.log(wc+1),1)[0]
        pd_days=(df["Date"].iloc[i]-df["Date"].iloc[i-4]).days/4
        r_day=slope/max(pd_days,1); rt.append(np.clip(np.exp(r_day*5.5),0.1,10.0))
    df["Rt"]=rt
    return df.dropna(subset=["lag_6"]).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def predict_surge(df, h):
    rows=[]
    for i in range(len(df)):
        fi=i+h
        if fi>=len(df): rows.append(np.nan); continue
        row=df.iloc[i][FEATS]
        if row.isnull().any(): rows.append(np.nan); continue
        rows.append(MODEL.predict_proba(row.values.reshape(1,-1))[0,1])
    return rows

def shade_surges(fig, df, color="rgba(220,50,50,0.12)"):
    """Add vertical surge shading bands to a plotly figure."""
    in_s = False
    s0 = None
    for _, row in df.iterrows():
        if row["surge"] == 1 and not in_s:
            s0 = row["Date"]; in_s = True
        elif row["surge"] == 0 and in_s:
            fig.add_vrect(x0=s0, x1=row["Date"], fillcolor=color, line_width=0)
            in_s = False
    if in_s and s0 is not None:
        fig.add_vrect(x0=s0, x1=df["Date"].iloc[-1], fillcolor=color, line_width=0)
    return fig

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### \U0001f9a0 Inferra")
    st.markdown("*Disease Surge Early Warning*")
    st.markdown("---")
    page = st.radio("Navigate",[
        "\U0001f4cb  Technical Summary",
        "\U0001f30d  Disease Surveillance",
        "\U0001f3e5  Public Advisory",
    ],label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Data sources**")
    st.markdown("\u2022 WHO COVID-19 (238-country training)\n\u2022 NNDSS fortnightly reports\n\u2022 PHF Science NZ virology\n\u2022 ABS / Stats NZ populations")
    st.markdown("---")
    st.caption("Inferra v1.0")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TECHNICAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
if "Summary" in page:
    st.markdown('''<div class="who-bar"><h1>\U0001f9a0 Inferra \u2014 Project Summary</h1>
        <p>Respiratory disease surveillance · Australia & New Zealand · Updated fortnightly</p>
    </div>''', unsafe_allow_html=True)

    st.markdown('<div class="sec">What we built</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.markdown("**Problem**")
    c1.markdown("Public health agencies need 2\u20134 weeks advance warning before a respiratory disease surge peaks \u2014 to issue communications, pre-position hospital capacity, and coordinate inter-state resources.")
    c2.markdown("**Solution**")
    c2.markdown("""A machine learning early warning system that:
- Detects whether a surge is approaching (classification)
- Estimates surge magnitude (regression)
- Generalises across geographies AND diseases""")
    c3.markdown("**Novel finding**")
    c3.markdown("NSW Granger-causes RSV surges in VIC (p=0.002, 4-week lead). Monitoring NSW gives VIC health agencies an extra 4 weeks of warning beyond the ML model alone.")

    st.markdown('<div class="sec">Training pipeline</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    for col,label,val,sub in [
        (k1,"Training countries","238","AUS + NZ held out — no leakage"),
        (k2,"Training rows","74,090","after feature engineering"),
        (k3,"Features","16","Rt + lags + rolling + z-score"),
        (k4,"Test sets","4","2 geographic + 2 cross-disease"),
    ]:
        col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div></div>',unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="sec">Key results</div>', unsafe_allow_html=True)
    left,right = st.columns(2)
    with left:
        fs=feat_imp.sort_values("Importance",ascending=True)
        colors=["#c62828" if f=="Rt" else "#003366" if i>0.05 else "#90a4ae" for f,i in zip(fs["Feature"],fs["Importance"])]
        fig1=go.Figure(go.Bar(x=fs["Importance"],y=fs["Feature"],orientation="h",marker_color=colors,hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig1.add_vline(x=feat_imp["Importance"].mean(),line_dash="dash",line_color="#f57f17",annotation_text="mean")
        fig1.update_layout(**{**THEME,"margin":dict(l=45,r=25,t=60,b=45)},height=380,
            title=dict(text="Feature importance \u2014 R\u209c drives 46% of predictions",font=dict(size=13)),
            xaxis=dict(title="Importance",gridcolor="#f0f0f0"),yaxis=dict(gridcolor="#f0f0f0"))
        st.plotly_chart(fig1,use_container_width=True)
    with right:
        fig2=go.Figure()
        fig2.add_hrect(y0=0.7,y1=1.0,fillcolor="rgba(46,125,50,0.06)",line_width=0,
            annotation_text="Actionable zone (AUC > 0.7)",annotation_position="top left",annotation_font_size=10)
        for mn,color in [("Logistic Regression","#003366"),("Random Forest","#c62828")]:
            dh=horizon[horizon["Model"]==mn].sort_values("Horizon_weeks")
            fig2.add_trace(go.Scatter(x=dh["Horizon_weeks"],y=dh["AUC"],name=mn,
                line=dict(color=color,width=2.5),mode="lines+markers",marker=dict(size=8)))
        fig2.update_layout(**{**THEME,"margin":dict(l=45,r=25,t=60,b=45)},height=380,
            title=dict(text="Early warning AUC vs forecast horizon",font=dict(size=13)),
            xaxis=dict(title="Weeks ahead",gridcolor="#f0f0f0",tickvals=[1,2,3,4,6,8],ticktext=["1w","2w","3w","4w","6w","8w"]),
            yaxis=dict(title="ROC-AUC",gridcolor="#f0f0f0",range=[0.4,1.05]),legend=dict(orientation="h",y=-0.2))
        st.plotly_chart(fig2,use_container_width=True)

    st.markdown('<div class="sec">Model performance — all four tests</div>', unsafe_allow_html=True)
    r1,r2 = st.columns(2)
    with r1:
        st.caption("**Test 1 \u2014 Geographic generalisation (Australia COVID)**")
        st.dataframe(pd.DataFrame({
            "Model":["Random Forest","Gradient Boosting","Logistic Regression","XGBoost"],
            "AUC (h=0)":[1.000,1.000,0.995,1.000],"AUC (4w)":[0.654,0.655,0.711,0.648],"F1":[1.000,0.981,0.964,0.941],
        }),use_container_width=True,hide_index=True)
        st.caption("AUC at h=0 reflects autocorrelation. The 4-week horizon AUC is the operationally meaningful number.")
        st.markdown("")
        st.caption("**Test 3 \u2014 Geographic generalisation (New Zealand COVID — proper holdout)**")
        st.dataframe(pd.DataFrame({
            "Model":["Random Forest","Gradient Boosting","Logistic Regression","XGBoost"],
            "AUC":[0.971,0.960,0.952,0.963],
            "F1":[0.767,0.741,0.698,0.752],
            "Precision":[0.911,0.882,0.841,0.897],
            "Recall":[0.662,0.634,0.588,0.640],
        }),use_container_width=True,hide_index=True)
        st.caption("NZ excluded from training — legitimate geographic holdout. No data leakage.")
    with r2:
        st.caption("**Test 2 \u2014 Cross-disease generalisation (Australia RSV)**")
        st.dataframe(pd.DataFrame({
            "Model":["Logistic Regression","Gradient Boosting","Random Forest","XGBoost"],
            "AUC":[0.923,0.881,0.549,0.518],
            "F1 (default)":[0.000,0.000,0.000,0.000],
            "F1 (Fix A)":[0.848,0.612,0.400,0.320],
        }),use_container_width=True,hide_index=True)
        st.caption("F1=0 before recalibration \u2014 model ranks correctly (AUC=0.92) but uses COVID probability scale.")
        st.markdown("")
        st.caption("**Test 4 \u2014 Cross-disease generalisation (New Zealand RSV)**")
        st.dataframe(pd.DataFrame({
            "Model":["Logistic Regression","Gradient Boosting","Random Forest","XGBoost"],
            "AUC":[0.545,0.532,0.521,0.518],
            "F1 (optimal)":[0.868,0.821,0.794,0.782],
        }),use_container_width=True,hide_index=True)
        st.caption("AUC near 0.5 = data-length issue (NZ RSV 2022–2025 only). AUS RSV (8yr) = 0.923 same model.")

    st.markdown('<div class="sec">Three headline findings</div>', unsafe_allow_html=True)
    f1c,f2c,f3c = st.columns(3)
    f1c.markdown('<div class="box-blue"><strong>4-week early warning</strong><br>AUC=0.711 at 4-week horizon. Correctly identifies upcoming surges 4 weeks in advance \u2014 the actionable window for health agencies to prepare.</div>',unsafe_allow_html=True)
    f2c.markdown('<div class="box-blue"><strong>Cross-disease + geographic transfer</strong><br>Trained on COVID (238 countries). AUS RSV AUC=0.923. NZ COVID AUC=0.971. The model learns epidemic shape, not disease or country-specific patterns.</div>',unsafe_allow_html=True)
    f3c.markdown('<div class="box-blue"><strong>NSW is the canary</strong><br>NSW Granger-causes VIC (p=0.002, 4w lead) and SA (p=0.001). Monitoring NSW R\u209c gives lagging states extra warning beyond the ML model alone.</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DISEASE SURVEILLANCE
# ══════════════════════════════════════════════════════════════════════════════
elif "Surveillance" in page:
    st.markdown('''<div class="who-bar"><h1>\U0001f9a0 Inferra \u2014 Disease Surge Forecast</h1>
        <p>Powered by machine learning trained on 238 countries \u00b7 2-week and 4-week surge probability</p>
    </div>''', unsafe_allow_html=True)

    st.markdown('<div style="font-size:22px;font-weight:800;color:#003366;margin:1rem 0 0.3rem">🦠 COVID-19 Surge Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#555;margin-bottom:0.8rem">Select a country to view the current Rₜ, weekly cases, and surge probability forecast.</div>', unsafe_allow_html=True)
    who,pop_df = load_who()
    countries = sorted(who["Country"].unique().tolist())
    selected  = st.selectbox("Select a country",countries,
                              index=countries.index("United States of America") if "United States of America" in countries else 0)
    sub = who[who["Country"]==selected].copy().sort_values("Date_reported")
    pop_row = pop_df[pop_df["Country"]==selected]
    pop_val = float(pop_row["Population"].iloc[0]) if len(pop_row) else None

    with st.spinner(f"Building features for {selected}..."):
        feat_df = build_features(sub["Date_reported"],sub["New_cases"],pop_val)

    if len(feat_df)<10:
        st.warning(f"Not enough data for {selected}.")
    else:
        # ── Forecast horizon selector ─────────────────────────────────────
        # Reviewer feedback: let users pick the forecast horizon via a
        # dropdown and show the rough model accuracy at that horizon.
        horizon_opts = {
            "1 week ahead":  1,
            "2 weeks ahead": 2,
            "3 weeks ahead": 3,
            "4 weeks ahead": 4,
        }
        hcol, _ = st.columns([1, 3])
        with hcol:
            horizon_label = st.selectbox(
                "Forecast horizon",
                list(horizon_opts.keys()),
                index=1,  # default to 2 weeks
                help="Shorter horizons are more accurate; longer horizons give more lead time to act.",
            )
        selected_h = horizon_opts[horizon_label]
        h_word = "week" if selected_h == 1 else "weeks"

        # Look up rough accuracy (test-set AUC) for the loaded model
        # at the selected horizon, from horizon_results.csv.
        h_row = horizon[(horizon["Model"]==MODEL_NAME) & (horizon["Horizon_weeks"]==selected_h)]
        auc_sel = float(h_row["AUC"].iloc[0]) if len(h_row) else 0.5
        acc_pct = auc_sel * 100  # rough accuracy proxy

        # Predict surge probability at the selected horizon
        feat_df[f"prob_{selected_h}w"] = predict_surge(feat_df, selected_h)
        psel_series = feat_df[f"prob_{selected_h}w"].dropna()
        psel = psel_series.iloc[-1] if len(psel_series) else 0
        lc = feat_df.dropna(subset=["Rt"]).iloc[-1]
        rt_c = lc["Rt"]

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="kpi"><div class="kpi-label">Current R\u209c</div><div class="kpi-value" style="color:{"#c62828" if rt_c>1.05 else "#f57f17" if rt_c>0.98 else "#2e7d32"}">{rt_c:.3f}</div><div class="kpi-sub">{"Growing" if rt_c>1.05 else "Stable" if rt_c>0.98 else "Declining"}</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="kpi"><div class="kpi-label">Latest weekly cases</div><div class="kpi-value">{int(lc["Cases"]):,}</div><div class="kpi-sub">{lc["Date"].strftime("%d %b %Y")}</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="kpi"><div class="kpi-label">Surge prob \u2014 {selected_h} {h_word}</div><div class="kpi-value" style="color:{"#c62828" if psel>0.5 else "#f57f17" if psel>0.3 else "#2e7d32"}">{psel*100:.0f}%</div><div class="kpi-sub">Model confidence</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="kpi"><div class="kpi-label">Model accuracy</div><div class="kpi-value">~{acc_pct:.0f}%</div><div class="kpi-sub">{MODEL_NAME} \u00b7 AUC at {selected_h}{h_word[0]}</div></div>', unsafe_allow_html=True)
        st.markdown("")

        hist_clean = trim_trailing_zeros(feat_df[feat_df["Cases"] >= 0]).copy()
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist_clean["Date"], y=hist_clean["Cases"],
            fill="tozeroy", fillcolor="rgba(0,51,102,0.12)",
            line=dict(color="#003366", width=2), name="Weekly cases",
            hovertemplate="%{x|%b %Y}<br>Cases: %{y:,}<extra></extra>"))
        fig_hist.update_layout(**THEME, height=320,
            title=f"COVID-19 \u2014 {selected} \u00b7 Full history",
            xaxis=dict(gridcolor="#f0f0f0", tickformat="%Y",
                       range=[hist_clean["Date"].min(), hist_clean["Date"].max()]),
            yaxis=dict(title="Weekly cases", gridcolor="#f0f0f0", rangemode="nonnegative"),
            showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
        last_date = hist_clean["Date"].max()
        st.caption(f"Data: WHO COVID-19 global surveillance · Last reported: {last_date.strftime('%d %b %Y')}")
        if last_date.year < 2024:
            st.warning(f"⚠ **{selected}** stopped reporting COVID-19 data to WHO after {last_date.strftime('%b %Y')}.")

        st.markdown('<div class="sec">Inferra surge forecast</div>', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)

        def surge_label(prob):
            if prob > 0.6:   return "HIGH", "#c62828", "\u26a0\ufe0f"
            elif prob > 0.3: return "MODERATE", "#f57f17", "\u26a1"
            else:            return "LOW", "#2e7d32", "\u2713"

        lbl_sel, col_sel, ico_sel = surge_label(psel)
        lbl_rt,col_rt,ico_rt = ("GROWING","#c62828","\u25b2") if rt_c>1.05 else \
                                ("STABLE","#f57f17","\u25cf") if rt_c>0.98 else \
                                ("DECLINING","#2e7d32","\u25bc")

        fc1.markdown(f'''<div class="kpi"><div class="kpi-label">Current trend</div>
            <div class="kpi-value" style="color:{col_rt}">{ico_rt} {lbl_rt}</div>
            <div class="kpi-sub">R\u209c = {rt_c:.3f}</div></div>''', unsafe_allow_html=True)
        fc2.markdown(f'''<div class="kpi"><div class="kpi-label">Surge risk \u2014 {selected_h} {h_word}</div>
            <div class="kpi-value" style="color:{col_sel}">{ico_sel} {lbl_sel}</div>
            <div class="kpi-sub">{psel*100:.0f}% model confidence</div></div>''', unsafe_allow_html=True)
        fc3.markdown(f'''<div class="kpi"><div class="kpi-label">Model accuracy</div>
            <div class="kpi-value">~{acc_pct:.0f}%</div>
            <div class="kpi-sub">{MODEL_NAME} \u00b7 test-set AUC at {selected_h}{h_word[0]}</div></div>''', unsafe_allow_html=True)

        interp_color = "box-red" if psel>0.6 else \
                       "box-amber" if psel>0.3 else "box-green"
        interp_msg = (
            f"The model detects a <strong>high probability of a COVID-19 surge in {selected}</strong> "
            f"within the next {selected_h} {h_word}. Public health agencies should review surge preparedness protocols."
        ) if psel>0.6 else (
            f"There is a <strong>moderate surge signal for {selected}</strong> over the next {selected_h} {h_word}. "
            f"R\u209c = {rt_c:.3f} suggests monitoring is warranted but no immediate action required."
        ) if psel>0.3 else (
            f"The model shows <strong>low surge risk for {selected}</strong> over the next {selected_h} {h_word}. "
            f"Current R\u209c = {rt_c:.3f} \u2014 transmission is {lbl_rt.lower()}."
        )
        st.markdown(f'<div class="{interp_color}" style="margin-top:10px;font-size:13px">{interp_msg}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── RSV country dropdown ───────────────────────────────────────────────
    st.markdown('<div style="font-size:22px;font-weight:800;color:#003366;margin:1rem 0 0.3rem">🫁 RSV Surveillance</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#555;margin-bottom:0.8rem">RSV data available for Australia (NNDSS fortnightly reports) and New Zealand (PHF Science virology reports).</div>', unsafe_allow_html=True)
    rsv_country = st.selectbox("Country", ["Australia", "New Zealand"], index=0, key="rsv_country")

    if rsv_country == "Australia":
        rsv_latest=rsv_nat.dropna(subset=["Rt"]).iloc[-1]; rsv_prev=rsv_nat.dropna(subset=["Rt"]).iloc[-2]
        rt_rsv=rsv_latest["Rt"]; rt_trend_r=rt_rsv-rsv_prev["Rt"]; cases_rsv=int(rsv_latest["Cases"])

        ra,rb,rc,rd = st.columns(4)
        ra.markdown(f'<div class="kpi"><div class="kpi-label">RSV R\u209c (national)</div><div class="kpi-value" style="color:{"#c62828" if rt_rsv>1.05 else "#f57f17" if rt_rsv>0.98 else "#2e7d32"}">{rt_rsv:.3f}</div><div class="kpi-sub">{rsv_latest["Date"].strftime("%d %b %Y")}</div></div>',unsafe_allow_html=True)
        rb.markdown(f'<div class="kpi"><div class="kpi-label">R\u209c trend</div><div class="kpi-value" style="color:{"#c62828" if rt_trend_r>0 else "#2e7d32"}">{"▲" if rt_trend_r>0 else "▼"} {abs(rt_trend_r):.3f}</div><div class="kpi-sub">vs previous fortnight</div></div>',unsafe_allow_html=True)
        rc.markdown(f'<div class="kpi"><div class="kpi-label">Cases (last fortnight)</div><div class="kpi-value">{cases_rsv:,}</div><div class="kpi-sub">national total</div></div>',unsafe_allow_html=True)
        rd.markdown(f'<div class="kpi"><div class="kpi-label">Surge fortnights</div><div class="kpi-value">{int(rsv_pred["surge"].sum())}</div><div class="kpi-sub">of {len(rsv_pred)} total</div></div>',unsafe_allow_html=True)
        st.markdown("")

        fig_r = go.Figure()
        merge = rsv_nat.merge(rsv_pred[["Date","surge"]],on="Date",how="left")
        surge_dates = sorted(merge[merge["surge"]==1]["Date"].tolist())
        if surge_dates:
            s=surge_dates[0]
            for i in range(1,len(surge_dates)):
                if (surge_dates[i]-surge_dates[i-1]).days>20:
                    fig_r.add_vrect(x0=s,x1=surge_dates[i-1],fillcolor="rgba(198,40,40,0.08)",line_width=0); s=surge_dates[i]
            fig_r.add_vrect(x0=s,x1=surge_dates[-1],fillcolor="rgba(198,40,40,0.08)",line_width=0,
                annotation_text="confirmed surge",annotation_position="top left",annotation_font_size=10,annotation_font_color="#c62828")
        fig_r.add_trace(go.Scatter(x=rsv_nat["Date"],y=rsv_nat["Cases"],fill="tozeroy",
            fillcolor="rgba(0,51,102,0.07)",line=dict(color="#003366",width=2),name="RSV cases",
            hovertemplate="%{x|%d %b %Y}<br>Cases: %{y:,}<extra></extra>"))
        for sc,color in STATE_COLORS.items():
            col2=f"Rt_{sc}"; valid=rsv_st.dropna(subset=[col2])
            fig_r.add_trace(go.Scatter(x=valid["Date"],y=valid[col2],name=f"{sc} R\u209c",
                line=dict(color=color,width=1.5),yaxis="y2",opacity=0.75,
                hovertemplate=f"{sc} %{{x|%d %b %Y}}<br>R\u209c=%{{y:.3f}}<extra></extra>"))
        fig_r.add_hline(y=1.0,line_dash="dash",line_color="#333",line_width=1,yref="y2",
            annotation_text="R\u209c=1",annotation_font_size=10)
        fig_r.update_layout(**THEME,height=400,
            title="Australia RSV \u2014 national cases + state R\u209c + confirmed surge periods",
            xaxis=dict(gridcolor="#f0f0f0"),
            yaxis=dict(title="RSV cases (fortnightly)",gridcolor="#f0f0f0"),
            yaxis2=dict(title="R\u209c by state",overlaying="y",side="right",range=[0.7,1.4],showgrid=False),
            legend=dict(orientation="h",y=-0.22))
        st.plotly_chart(fig_r,use_container_width=True)
        st.markdown('<div class="box-blue"><strong>How to read:</strong> Blue area = national RSV cases (left axis). Coloured lines = R\u209c per state (right axis) \u2014 when a line crosses 1.0, that state\u2019s epidemic is growing. Red shading = confirmed surge fortnights. NSW R\u209c peaks ~4 weeks before VIC \u2014 watch NSW as an early indicator.</div>',unsafe_allow_html=True)

    else:  # New Zealand
        try:
            _, nz_rsv_surv = load_nz()
            nz_rsv_ok = True
        except FileNotFoundError:
            st.warning("⚠️ NZ RSV data not found. Run Cell 48 + Cell 55 in the notebook first.")
            nz_rsv_ok = False

        if nz_rsv_ok:
            nz_rsv_latest = nz_rsv_surv.dropna(subset=["Rt"]).iloc[-1] if "Rt" in nz_rsv_surv.columns and nz_rsv_surv["Rt"].notna().any() else nz_rsv_surv.iloc[-1]
            nz_cases_rsv  = f"{nz_rsv_latest['Cases']:,.0f}" if "Cases" in nz_rsv_surv.columns else "N/A"
            nz_surge_count = int(nz_rsv_surv["surge"].sum()) if "surge" in nz_rsv_surv.columns else 0

            ra,rb,rc,rd = st.columns(4)
            ra.markdown(f'<div class="kpi"><div class="kpi-label">Data source</div><div class="kpi-value" style="font-size:14px">PHF Science</div><div class="kpi-sub">NZ virology reports</div></div>',unsafe_allow_html=True)
            rb.markdown(f'<div class="kpi"><div class="kpi-label">Date range</div><div class="kpi-value" style="font-size:14px">2022–2025</div><div class="kpi-sub">post-COVID rebound</div></div>',unsafe_allow_html=True)
            rc.markdown(f'<div class="kpi"><div class="kpi-label">Surge weeks</div><div class="kpi-value">{nz_surge_count}</div><div class="kpi-sub">of {len(nz_rsv_surv)} total</div></div>',unsafe_allow_html=True)
            rd.markdown(f'<div class="kpi"><div class="kpi-label">Cross-disease AUC</div><div class="kpi-value">0.545</div><div class="kpi-sub">data-length limited</div></div>',unsafe_allow_html=True)
            st.markdown("")

            fig_nz_rsv = go.Figure()
            fig_nz_rsv = shade_surges(fig_nz_rsv, nz_rsv_surv, "rgba(255,140,0,0.15)")
            fig_nz_rsv.add_trace(go.Scatter(
                x=nz_rsv_surv["Date"], y=nz_rsv_surv["cases_per_100k"],
                name="RSV cases / 100k", line=dict(color="teal", width=2),
                hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} per 100k<extra></extra>"))
            if "surge_prob" in nz_rsv_surv.columns:
                fig_nz_rsv.add_trace(go.Scatter(
                    x=nz_rsv_surv["Date"], y=nz_rsv_surv["surge_prob"],
                    name="P(surge)", yaxis="y2",
                    line=dict(color="darkorange", width=1.5, dash="dot"),
                    hovertemplate="%{x|%d %b %Y}<br>P(surge)=%{y:.2f}<extra></extra>"))
            fig_nz_rsv.update_layout(**THEME, height=400,
                title="New Zealand RSV \u2014 weekly cases & surge probability  (orange shading = surge period)",
                xaxis=dict(gridcolor="#f0f0f0"),
                yaxis=dict(title="Cases per 100k", gridcolor="#f0f0f0"),
                yaxis2=dict(title="P(surge)", overlaying="y", side="right", range=[0,1], showgrid=False),
                legend=dict(orientation="h", y=-0.22))
            st.plotly_chart(fig_nz_rsv, use_container_width=True)
            st.markdown('<div class="box-amber">⚠️ NZ RSV spans only 2022–2025 (post-COVID rebound). 76% of weeks are labelled surge — AUS RSV (8 years) achieves AUC=0.923 with the same model. This is a data-length limitation.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — NEW ZEALAND
# ══════════════════════════════════════════════════════════════════════════════
elif "New Zealand" in page:
    st.markdown('''<div class="who-bar">
        <h1>🇳🇿 New Zealand — Validation Tests 3 &amp; 4</h1>
        <p>Model trained on 238 countries · Australia &amp; NZ both excluded · No data leakage</p>
    </div>''', unsafe_allow_html=True)

    try:
        nz_covid_df, nz_rsv_df = load_nz()
        nz_ok = True
    except FileNotFoundError:
        st.warning("⚠️ NZ data files not found. Run Cells 46, 48 and 55 in `modeling_v2_global_with_regression.ipynb` first.")
        nz_ok = False

    if nz_ok:

        # ── TEST 3: NZ COVID ───────────────────────────────────────────────
        st.markdown('<div class="sec">Test 3 — NZ COVID-19 · Geographic Generalisation</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="box-green">✅ <b>Data source:</b> WHO_COVID19_cleaned.csv · '
            'Country = New Zealand &nbsp;|&nbsp; Model trained on <b>238 countries</b> '
            '— NZ was never seen during training.</div>',
            unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        for col,lbl,val,sub in [
            (c1,"ROC-AUC","0.971","Random Forest"),
            (c2,"F1 Score","0.767","Default threshold"),
            (c3,"Precision","0.911","Low false-alarm rate"),
            (c4,"Recall","0.662","228 test weeks"),
        ]:
            col.markdown(f'<div class="kpi"><div class="kpi-label">{lbl}</div>'
                         f'<div class="kpi-value">{val}</div>'
                         f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)
        st.markdown("")

        # NZ COVID timeline
        fig_nz_c = go.Figure()
        fig_nz_c = shade_surges(fig_nz_c, nz_covid_df, "rgba(220,50,50,0.12)")
        fig_nz_c.add_trace(go.Scatter(
            x=nz_covid_df["Date"], y=nz_covid_df["cases_per_100k"],
            name="Cases / 100k", line=dict(color="#003366", width=1.8),
            hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} per 100k<extra></extra>"))
        if "surge_prob" in nz_covid_df.columns:
            fig_nz_c.add_trace(go.Scatter(
                x=nz_covid_df["Date"], y=nz_covid_df["surge_prob"],
                name="P(surge)", yaxis="y2",
                line=dict(color="firebrick", width=1.5, dash="dot"),
                hovertemplate="%{x|%d %b %Y}<br>P(surge)=%{y:.2f}<extra></extra>"))
        if "Rt" in nz_covid_df.columns:
            fig_nz_c.add_trace(go.Scatter(
                x=nz_covid_df["Date"], y=nz_covid_df["Rt"],
                name="R\u209c", yaxis="y2",
                line=dict(color="darkorange", width=1.2, dash="dot"), opacity=0.7,
                hovertemplate="%{x|%d %b %Y}<br>R\u209c=%{y:.3f}<extra></extra>"))
        fig_nz_c.add_hline(y=1.0, line_dash="dash", line_color="#999", line_width=1,
                            yref="y2", annotation_text="R\u209c=1", annotation_font_size=9)
        fig_nz_c.update_layout(**THEME, height=370,
            title="NZ COVID-19 — Weekly Cases & Surge Probability  (red shading = surge period)",
            yaxis=dict(title="Cases per 100k", gridcolor="#f0f0f0"),
            yaxis2=dict(title="P(surge) / R\u209c", overlaying="y", side="right",
                        range=[0, 2.5], showgrid=False),
            legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_nz_c, use_container_width=True)

        with st.expander("📈 NZ COVID — Horizon Analysis (early warning lead time)"):
            col_h, col_txt = st.columns([2,1])
            with col_h:
                hor_data = pd.DataFrame({
                    "Horizon": [0,1,2,3,4,6,8],
                    "AUC":     [0.971,0.840,0.663,0.524,0.510,0.500,0.412]
                })
                fig_hor = go.Figure()
                fig_hor.add_hrect(y0=0.8, y1=1.05,
                                   fillcolor="rgba(46,125,50,0.06)", line_width=0,
                                   annotation_text="Actionable (≥0.8)",
                                   annotation_position="top left", annotation_font_size=9)
                fig_hor.add_trace(go.Scatter(
                    x=hor_data["Horizon"], y=hor_data["AUC"],
                    mode="lines+markers+text",
                    text=[f"{v:.3f}" for v in hor_data["AUC"]],
                    textposition="top center",
                    line=dict(color="#003366", width=2.5), marker=dict(size=9)))
                fig_hor.add_hline(y=0.5, line_dash="dash", line_color="grey",
                                   annotation_text="Random baseline")
                fig_hor.update_layout(**THEME, height=300,
                    title="AUC vs Forecast Horizon — NZ COVID",
                    xaxis=dict(title="Weeks ahead", tickvals=[0,1,2,3,4,6,8],
                               gridcolor="#f0f0f0"),
                    yaxis=dict(title="ROC-AUC", range=[0.3,1.08], gridcolor="#f0f0f0"),
                    showlegend=False)
                st.plotly_chart(fig_hor, use_container_width=True)
            with col_txt:
                st.markdown("""
**Key findings**

| Horizon | AUC |
|---------|-----|
| 0 wk | 0.971 |
| **1 wk** | **0.840** ✅ |
| 2 wk | 0.663 |
| 3 wk | 0.524 |
| 4+ wk | ~0.5 |

1-week actionable window mirrors Australia — confirms this is a property of epidemic dynamics, not geography.
""")

        st.markdown("---")

        # ── TEST 4: NZ RSV ─────────────────────────────────────────────────
        st.markdown('<div class="sec">Test 4 — NZ RSV · Cross-Disease Generalisation</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="box-blue">📋 <b>Data source:</b> NZ_Airborne_Disease_Data.xlsx · '
            'PHF Science weekly virology reports (2022–2025) &nbsp;|&nbsp; '
            'COVID-trained model applied to RSV</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="box-amber">⚠️ <b>Data limitation:</b> NZ RSV spans only 2022–2025 '
            '(post-COVID rebound). 76% of weeks are labelled surge — minimal non-surge baseline. '
            'AUS RSV (8 years) achieves AUC=0.923 with the same model. '
            'This is a data-length limitation, not a model failure.</div>',
            unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        for col,lbl,val,sub in [
            (c1,"ROC-AUC","0.545","Near-chance ranking"),
            (c2,"F1 (optimal)","0.868","Calibrated threshold"),
            (c3,"Surge weeks","82 / 107","76% positive rate"),
            (c4,"Data span","4 yrs","2022–2025 only"),
        ]:
            col.markdown(f'<div class="kpi"><div class="kpi-label">{lbl}</div>'
                         f'<div class="kpi-value">{val}</div>'
                         f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)
        st.markdown("")

        # NZ RSV timeline
        fig_nz_r = go.Figure()
        fig_nz_r = shade_surges(fig_nz_r, nz_rsv_df, "rgba(255,140,0,0.15)")
        fig_nz_r.add_trace(go.Scatter(
            x=nz_rsv_df["Date"], y=nz_rsv_df["cases_per_100k"],
            name="RSV cases / 100k", line=dict(color="teal", width=2),
            hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} per 100k<extra></extra>"))
        if "surge_prob" in nz_rsv_df.columns:
            fig_nz_r.add_trace(go.Scatter(
                x=nz_rsv_df["Date"], y=nz_rsv_df["surge_prob"],
                name="P(surge) — COVID model", yaxis="y2",
                line=dict(color="darkorange", width=1.5, dash="dot"),
                hovertemplate="%{x|%d %b %Y}<br>P(surge)=%{y:.2f}<extra></extra>"))
        fig_nz_r.update_layout(**THEME, height=350,
            title="NZ RSV — Weekly Cases & Surge Probability  (orange shading = z-score surge)",
            yaxis=dict(title="Cases per 100k", gridcolor="#f0f0f0"),
            yaxis2=dict(title="P(surge)", overlaying="y", side="right",
                        range=[0,1], showgrid=False),
            legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_nz_r, use_container_width=True)

        st.markdown("---")

        # ── ALL TESTS SUMMARY TABLE ────────────────────────────────────────
        st.markdown('<div class="sec">All Tests — Generalisation Summary</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            {"Test":"Test 1","Country":"Australia","Disease":"COVID-19",
             "Type":"Geographic","AUC":"1.000","F1":"1.000",
             "Data source":"WHO COVID file","Status":"✅ Benchmark"},
            {"Test":"Test 2","Country":"Australia","Disease":"RSV",
             "Type":"Cross-disease","AUC":"0.923","F1":"0.848*",
             "Data source":"NNDSS fortnightly reports","Status":"✅ Strong"},
            {"Test":"Test 3","Country":"New Zealand","Disease":"COVID-19",
             "Type":"Geographic","AUC":"0.971","F1":"0.767",
             "Data source":"WHO COVID file","Status":"✅ Confirmed"},
            {"Test":"Test 4","Country":"New Zealand","Disease":"RSV",
             "Type":"Cross-disease","AUC":"0.545","F1":"0.868*",
             "Data source":"NZ_Airborne_Disease_Data.xlsx","Status":"⚠️ Data limited"},
        ]), use_container_width=True, hide_index=True)
        st.caption("* F1 at optimal calibrated threshold  |  "
                   "Training: 238 countries (AUS + NZ excluded — no data leakage)")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PUBLIC ADVISORY
# ══════════════════════════════════════════════════════════════════════════════
elif "Advisory" in page:
    st.markdown('''<div class="who-bar"><h1>\U0001f3e0 Disease Situation \u2014 Australia & Global</h1>
        <p>Current disease situation · RSV and COVID-19 · Australia</p>
    </div>''', unsafe_allow_html=True)

    # ── RSV country dropdown ───────────────────────────────────────────────
    st.markdown('<div style="font-size:22px;font-weight:800;color:#003366;margin:1rem 0 0.3rem">🟦 RSV</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#555;margin-bottom:0.8rem">Select a country to view the current RSV situation and public health advice.</div>', unsafe_allow_html=True)
    rsv_pub_country = st.selectbox("Country", ["Australia", "New Zealand"], index=0, key="rsv_pub_country")

    if rsv_pub_country == "Australia":
        latest=rsv_nat.dropna(subset=["Rt"]).iloc[-1]
        rt_now=latest["Rt"]; cases_now=int(latest["Cases"]); latest_date=latest["Date"].strftime("%d %b %Y")

        if rt_now>1.05:
            color,icon,status,msg,box_class=("#c62828","\u26a0\ufe0f","SURGE RISK",
                "RSV cases are growing across Australia. Each infected person is spreading the virus to more than one other person. People at higher risk \u2014 infants, elderly, and those with respiratory conditions \u2014 should take extra precautions.",
                "box-red")
        elif rt_now>0.98:
            color,icon,status,msg,box_class=("#f57f17","\u26a1","WATCH",
                "RSV activity is stable but could change quickly. Continue normal precautions \u2014 wash hands, stay home if unwell, and keep up with vaccinations.",
                "box-amber")
        else:
            color,icon,status,msg,box_class=("#2e7d32","\u2713","DECLINING",
                "RSV cases are declining across Australia. Transmission is slowing down. Continue routine hygiene practices.",
                "box-green")

        rsv_left, rsv_right = st.columns([1,2])
        with rsv_left:
            st.markdown(f"""<div style="text-align:center;padding:1rem 0">
                <div style="width:110px;height:110px;border-radius:50%;background:{color};
                    display:flex;align-items:center;justify-content:center;font-size:44px;margin:0 auto">{icon}</div>
                <div style="font-size:16px;font-weight:700;color:{color};margin-top:10px">{status}</div>
                <div style="font-size:11px;color:#666;margin-top:3px">as of {latest_date}</div>
                <div style="font-size:12px;color:#333;margin-top:8px"><strong>{cases_now:,}</strong> cases last fortnight</div>
            </div>""", unsafe_allow_html=True)
            state_latest=rsv_st.dropna(subset=["Rt_NSW"]).iloc[-1]
            for sc in ["NSW","VIC","QLD","WA","SA"]:
                rv=state_latest.get(f"Rt_{sc}",None)
                if rv is not None and not np.isnan(float(rv)):
                    rv=float(rv)
                    if rv>1.05: bg,ic,lb,tc="#ffebee","\u26a0","Rising","#c62828"
                    elif rv>0.95: bg,ic,lb,tc="#fff8e1","\u25cf","Stable","#f57f17"
                    else: bg,ic,lb,tc="#e8f5e9","\u2713","Falling","#2e7d32"
                    st.markdown(f'<div style="background:{bg};padding:5px 10px;border-radius:6px;margin-bottom:4px;display:flex;justify-content:space-between;align-items:center"><span style="font-weight:600;font-size:12px">{sc}</span><span style="color:{tc};font-size:12px">{ic} {lb}</span></div>',unsafe_allow_html=True)

        with rsv_right:
            st.markdown(f'<div class="{box_class}" style="font-size:13px;line-height:1.7;margin-bottom:10px">{msg}</div>',unsafe_allow_html=True)
            fig_rsv=go.Figure()
            m2=rsv_nat.merge(rsv_pred[["Date","surge"]],on="Date",how="left")
            sp=m2[m2["surge"]==1]
            fig_rsv.add_trace(go.Scatter(x=rsv_nat["Date"],y=rsv_nat["Cases"],fill="tozeroy",
                fillcolor="rgba(0,51,102,0.10)",line=dict(color="#003366",width=2),name="RSV cases",
                hovertemplate="%{x|%b %Y}<br>%{y:,} cases<extra></extra>"))
            fig_rsv.add_trace(go.Scatter(x=sp["Date"],y=sp["Cases"],mode="markers",
                marker=dict(color="#c62828",size=8),name="Surge period",
                hovertemplate="%{x|%b %Y}<br>\u26a0 Surge<extra></extra>"))
            fig_rsv.update_layout(**THEME,height=260,
                title="RSV cases \u2014 Australia (red dots = surge periods)",
                xaxis=dict(gridcolor="#f0f0f0",tickformat="%b %Y"),
                yaxis=dict(title="Cases",gridcolor="#f0f0f0"),
                legend=dict(orientation="h",y=-0.25))
            st.plotly_chart(fig_rsv,use_container_width=True)

    else:  # New Zealand
        try:
            _, nz_rsv_pub = load_nz()
            nz_pub_ok = True
        except FileNotFoundError:
            st.warning("⚠️ NZ RSV data not found. Run Cell 48 + Cell 55 in the notebook first.")
            nz_pub_ok = False

        if nz_pub_ok:
            nz_latest_row = nz_rsv_pub.iloc[-1]
            nz_surge_now  = int(nz_rsv_pub["surge"].sum()) if "surge" in nz_rsv_pub.columns else 0
            nz_total      = len(nz_rsv_pub)
            nz_surge_pct  = nz_surge_now / nz_total * 100 if nz_total > 0 else 0

            if nz_surge_pct > 60:
                color,icon,status,msg,box_class=("#c62828","\u26a0\ufe0f","SURGE RISK",
                    "RSV activity in New Zealand is elevated. The majority of recent weeks have been classified as surge periods. High-risk individuals should take extra precautions.",
                    "box-red")
            elif nz_surge_pct > 30:
                color,icon,status,msg,box_class=("#f57f17","\u26a1","WATCH",
                    "RSV activity in New Zealand is moderate. Monitor PHF Science reports for updates and continue standard precautions.",
                    "box-amber")
            else:
                color,icon,status,msg,box_class=("#2e7d32","\u2713","DECLINING",
                    "RSV cases in New Zealand are at lower levels. Continue routine hygiene practices.",
                    "box-green")

            rsv_left, rsv_right = st.columns([1,2])
            with rsv_left:
                latest_date_nz = pd.to_datetime(nz_latest_row["Date"]).strftime("%d %b %Y")
                st.markdown(f"""<div style="text-align:center;padding:1rem 0">
                    <div style="width:110px;height:110px;border-radius:50%;background:{color};
                        display:flex;align-items:center;justify-content:center;font-size:44px;margin:0 auto">{icon}</div>
                    <div style="font-size:16px;font-weight:700;color:{color};margin-top:10px">{status}</div>
                    <div style="font-size:11px;color:#666;margin-top:3px">as of {latest_date_nz}</div>
                    <div style="font-size:12px;color:#333;margin-top:8px"><strong>{nz_surge_now}</strong> of {nz_total} weeks were surge</div>
                </div>""", unsafe_allow_html=True)
                st.markdown('<div class="box-amber" style="font-size:11px">⚠️ Data: 2022–2025 only (post-COVID rebound). 76% of weeks classified as surge.</div>', unsafe_allow_html=True)

            with rsv_right:
                st.markdown(f'<div class="{box_class}" style="font-size:13px;line-height:1.7;margin-bottom:10px">{msg}</div>',unsafe_allow_html=True)
                fig_nz_pub = go.Figure()
                fig_nz_pub = shade_surges(fig_nz_pub, nz_rsv_pub, "rgba(255,140,0,0.15)")
                fig_nz_pub.add_trace(go.Scatter(
                    x=nz_rsv_pub["Date"], y=nz_rsv_pub["cases_per_100k"],
                    fill="tozeroy", fillcolor="rgba(0,130,130,0.10)",
                    line=dict(color="teal", width=2), name="RSV cases / 100k",
                    hovertemplate="%{x|%b %Y}<br>%{y:.2f} per 100k<extra></extra>"))
                fig_nz_pub.update_layout(**THEME, height=260,
                    title="RSV cases \u2014 New Zealand (orange shading = surge periods)",
                    xaxis=dict(gridcolor="#f0f0f0", tickformat="%b %Y"),
                    yaxis=dict(title="Cases per 100k", gridcolor="#f0f0f0"),
                    legend=dict(orientation="h", y=-0.25))
                st.plotly_chart(fig_nz_pub, use_container_width=True)

    st.markdown("*Source: WHO RSV fact sheet · Australian Dept of Health · PHF Science NZ virology reports*")
    r1, r2, r3 = st.columns(3)
    r1.markdown("**Protect yourself**")
    r1.markdown("""- Wash hands with soap for 20+ seconds\n- Avoid touching eyes, nose, mouth\n- Stay home when sick\n- Clean frequently touched surfaces\n- Avoid close contact with people who have cold symptoms""")
    r2.markdown("**Protect vulnerable people**")
    r2.markdown("""- RSV is most dangerous for **infants under 6 months** and **adults over 65**\n- Keep sick children away from newborns\n- Ask your doctor about **nirsevimab** for infants\n- Ask about **RSVPreF vaccine** if you are 60+\n- High-risk people should avoid crowded spaces during surge""")
    r3.markdown("**Stay informed**")
    r3.markdown("""- NNDSS reports: **health.gov.au**\n- WHO RSV: **who.int/rsv**\n- Follow your state health department\n- RSV has no specific treatment — prevention is key\n- Seek early care for infants with breathing difficulty""")

    st.markdown("---")

    # ── COVID Australia + country dropdown ─────────────────────────────────
    st.markdown('<div style="font-size:22px;font-weight:800;color:#003366;margin:1rem 0 0.3rem">🔴 COVID-19</div>', unsafe_allow_html=True)

    covid_aus = covid.copy()
    covid_rt_now = float(covid_aus.dropna(subset=["Rt"]).iloc[-1]["Rt"]) if "Rt" in covid_aus.columns and covid_aus["Rt"].notna().any() else 1.0
    covid_cases_now = int(covid_aus["Cases"].iloc[-1])
    covid_date_now  = covid_aus["Date"].iloc[-1].strftime("%d %b %Y")

    if covid_rt_now>1.05:
        cv_color,cv_icon,cv_status,cv_msg,cv_box=("#c62828","\u26a0\ufe0f","SURGE RISK","COVID-19 cases are growing in Australia.","box-red")
    elif covid_rt_now>0.98:
        cv_color,cv_icon,cv_status,cv_msg,cv_box=("#f57f17","\u26a1","WATCH","COVID-19 transmission is stable in Australia but could shift. Stay up to date with boosters.","box-amber")
    else:
        cv_color,cv_icon,cv_status,cv_msg,cv_box=("#2e7d32","\u2713","DECLINING","COVID-19 cases are declining in Australia. Continue routine precautions.","box-green")

    st.markdown(f"""<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
        <span style="font-size:28px">{cv_icon}</span>
        <span style="font-size:15px;font-weight:700;color:{cv_color}">{cv_status}</span>
        <span style="font-size:12px;color:#666">{covid_cases_now:,} cases in Australia · {covid_date_now}</span>
    </div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="{cv_box}" style="font-size:12px;line-height:1.6;margin-bottom:12px">{cv_msg}</div>',unsafe_allow_html=True)

    who_data, pop_df2 = load_who()
    countries_pub = sorted(who_data["Country"].unique().tolist())
    sel_pub = st.selectbox("Select a country to view COVID-19 trend", countries_pub,
                           index=countries_pub.index("United States of America") if "United States of America" in countries_pub else 0,
                           key="pub_country")
    sub_pub = who_data[who_data["Country"]==sel_pub].copy().sort_values("Date_reported")
    pop_pub = pop_df2[pop_df2["Country"]==sel_pub]
    pop_pub_val = float(pop_pub["Population"].iloc[0]) if len(pop_pub) else None

    with st.spinner(f"Loading {sel_pub}..."):
        feat_pub = build_features(tuple(sub_pub["Date_reported"]), tuple(sub_pub["New_cases"]), pop_pub_val)

    fig_cov_pub=go.Figure()
    recent_pub = trim_trailing_zeros(feat_pub[feat_pub["Cases"] >= 0].tail(156))
    fig_cov_pub.add_trace(go.Scatter(x=recent_pub["Date"],y=recent_pub["Cases"],fill="tozeroy",
        fillcolor="rgba(198,40,40,0.07)",line=dict(color="#c62828",width=2),name=sel_pub,
        hovertemplate="%{x|%b %Y}<br>%{y:,} cases<extra></extra>"))
    fig_cov_pub.update_layout(**THEME,height=300,
        title=f"COVID-19 \u2014 {sel_pub}",
        xaxis=dict(gridcolor="#f0f0f0",tickformat="%b %Y",
                   range=[recent_pub["Date"].min(), recent_pub["Date"].max()]),
        yaxis=dict(title="Weekly cases",gridcolor="#f0f0f0",rangemode="nonnegative"),
        showlegend=False)
    st.plotly_chart(fig_cov_pub,use_container_width=True)

    st.markdown("*Source: WHO COVID-19 public advice & Australian Department of Health*")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Protect yourself**")
    c1.markdown("""- Stay **up to date with COVID-19 vaccines** and boosters\n- Wear a mask in crowded indoor spaces\n- Improve ventilation — open windows when possible\n- Wash hands or use hand sanitiser regularly\n- Test if symptomatic — isolate if positive""")
    c2.markdown("**Protect vulnerable people**")
    c2.markdown("""- COVID-19 is severe for **immunocompromised**, elderly, and chronic illness patients\n- Inform close contacts if you test positive\n- Mask when visiting elderly relatives during surges\n- Antivirals (e.g. **Paxlovid**) available — seek care within **5 days** of symptoms\n- High-risk individuals: discuss preventive options with your doctor""")
    c3.markdown("**Stay informed**")
    c3.markdown("""- WHO advice: **who.int/covid19/advice-for-public**\n- Australian updates: **health.gov.au/covid19**\n- Long COVID support: **health.gov.au/long-covid**\n- Follow your state health department for local alerts\n- Emergency signs — breathing difficulty, chest pain: seek care immediately""")

    st.markdown("---")
    st.caption("Data: NNDSS Fortnightly Reports \u00b7 WHO COVID-19 Global Data \u00b7 ABS / Stats NZ Population Estimates \u00b7 PHF Science NZ Virology \u00b7 Inferra ML model (IST 718)")