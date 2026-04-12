"""
Inferra — Infectious Disease Surge Early Warning System
WHO-style public health dashboard for Australian respiratory disease surveillance.
Stakeholder: Australian public health agencies (AIHW, state health departments)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inferra — Disease Surge Early Warning",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── WHO-style CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #f8f9fa; }
    [data-testid="stSidebar"] { background: #003366; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    [data-testid="stSidebar"] .stRadio label { color: #ffffff !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .metric-label { font-size: 12px; color: #666; margin-bottom: 4px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-value { font-size: 28px; font-weight: 700; color: #003366; }
    .metric-sub { font-size: 12px; color: #999; margin-top: 2px; }
    .alert-green  { background: #e8f5e9; border-left: 4px solid #2e7d32; padding: 12px 16px; border-radius: 4px; }
    .alert-amber  { background: #fff8e1; border-left: 4px solid #f57f17; padding: 12px 16px; border-radius: 4px; }
    .alert-red    { background: #ffebee; border-left: 4px solid #c62828; padding: 12px 16px; border-radius: 4px; }
    .section-header { font-size: 13px; font-weight: 600; color: #003366; text-transform: uppercase; letter-spacing: 0.06em; margin: 1.5rem 0 0.75rem; border-bottom: 2px solid #003366; padding-bottom: 4px; }
    .finding-box { background: #e3f2fd; border-radius: 6px; padding: 12px 16px; font-size: 14px; color: #0d47a1; margin: 0.5rem 0; }
    .who-header { background: #003366; color: white; padding: 1rem 2rem; border-radius: 8px; margin-bottom: 1.5rem; }
    .who-header h1 { color: white; margin: 0; font-size: 22px; }
    .who-header p { color: rgba(255,255,255,0.8); margin: 4px 0 0; font-size: 14px; }
    .state-chip { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; margin: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
DATA = Path(__file__).parent / "data"

@st.cache_data
def load_data():
    rsv_nat  = pd.read_csv(DATA / "rsv_national.csv",    parse_dates=["Date"])
    rsv_st   = pd.read_csv(DATA / "rsv_state.csv",       parse_dates=["Date"])
    rsv_pred = pd.read_csv(DATA / "rsv_predictions.csv", parse_dates=["Date"])
    covid    = pd.read_csv(DATA / "covid_national.csv",  parse_dates=["Date"])
    horizon  = pd.read_csv(DATA / "horizon_results.csv")
    feat_imp = pd.read_csv(DATA / "feature_importance.csv")
    return rsv_nat, rsv_st, rsv_pred, covid, horizon, feat_imp

rsv_nat, rsv_st, rsv_pred, covid, horizon, feat_imp = load_data()

STATES       = ["NSW", "VIC", "QLD", "WA", "SA"]
STATE_POP    = {"NSW": 8_278_956, "VIC": 6_704_281, "QLD": 5_322_929, "WA": 2_808_786, "SA": 1_836_116}
STATE_COLORS = {"NSW": "#1f77b4", "VIC": "#d62728", "QLD": "#ff7f0e", "WA": "#2ca02c", "SA": "#9467bd"}
PLOTLY_THEME = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="Arial, sans-serif", size=12, color="#333"),
    margin=dict(l=50, r=30, t=50, b=50),
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🦠 Inferra")
    st.markdown("*Disease Surge Early Warning*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠  Situation overview",
         "📈  Early warning signals",
         "🗺️  State propagation",
         "🔬  Model & methods"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Data sources**")
    st.markdown("• WHO COVID-19 global data  \n• NNDSS fortnightly reports  \n• ABS population estimates")
    st.markdown("---")
    st.caption("Inferra v1.0 — IST 718 Group Project  \nAustralian public health agencies")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SITUATION OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "overview" in page:

    st.markdown("""
    <div class="who-header">
      <h1>🦠 Inferra — Respiratory Disease Surge Early Warning</h1>
      <p>Australia · Powered by machine learning trained on 239-country WHO COVID-19 data · Updated fortnightly</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Key numbers ──────────────────────────────────────────────────────────
    latest      = rsv_nat.dropna(subset=["Rt"]).iloc[-1]
    prev        = rsv_nat.dropna(subset=["Rt"]).iloc[-2]
    latest_date = latest["Date"].strftime("%d %b %Y")
    rt_now      = latest["Rt"]
    rt_trend    = rt_now - prev["Rt"]
    cases_now   = int(latest["Cases"])
    surge_weeks = int(rsv_pred["surge"].sum())
    pct_surge   = rsv_pred["surge"].mean() * 100

    rt_color  = "#c62828" if rt_now > 1.1 else "#f57f17" if rt_now > 1.0 else "#2e7d32"
    rt_status = "Growing" if rt_now > 1.05 else "Stable" if rt_now > 0.95 else "Declining"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Current Rₜ (national)</div>
            <div class="metric-value" style="color:{rt_color}">{rt_now:.3f}</div>
            <div class="metric-sub">{rt_status} · {latest_date}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        trend_sym = "▲" if rt_trend > 0 else "▼"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Rₜ trend (2-week)</div>
            <div class="metric-value" style="color:{'#c62828' if rt_trend>0 else '#2e7d32'}">{trend_sym} {abs(rt_trend):.3f}</div>
            <div class="metric-sub">vs previous fortnight</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">RSV cases (last fortnight)</div>
            <div class="metric-value">{cases_now:,}</div>
            <div class="metric-sub">national total</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Surge periods detected</div>
            <div class="metric-value">{surge_weeks}</div>
            <div class="metric-sub">{pct_surge:.0f}% of fortnights</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Alert banner ──────────────────────────────────────────────────────────
    if rt_now > 1.05:
        st.markdown(f"""<div class="alert-red">
            <strong>⚠ SURGE RISK — Rₜ = {rt_now:.3f}</strong><br>
            The reproduction number is above 1.0, meaning each case is generating more than one new infection.
            RSV cases are growing. Public health agencies should consider activating surge protocols.
        </div>""", unsafe_allow_html=True)
    elif rt_now > 0.98:
        st.markdown(f"""<div class="alert-amber">
            <strong>⚡ WATCH — Rₜ = {rt_now:.3f}</strong><br>
            The reproduction number is near 1.0. The epidemic is stable but could shift in either direction.
            Continue monitoring closely.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="alert-green">
            <strong>✓ DECLINING — Rₜ = {rt_now:.3f}</strong><br>
            RSV cases are declining. Rₜ below 1.0 means transmission chains are shrinking.
            Routine surveillance recommended.
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── National RSV trend ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">National RSV case trend</div>', unsafe_allow_html=True)

    fig = go.Figure()
    surge_dates = rsv_pred[rsv_pred["surge"] == 1]["Date"]

    # Surge shading
    if len(surge_dates):
        surge_ranges = []
        dates_sorted = sorted(surge_dates)
        start = dates_sorted[0]
        for i in range(1, len(dates_sorted)):
            if (dates_sorted[i] - dates_sorted[i-1]).days > 20:
                surge_ranges.append((start, dates_sorted[i-1]))
                start = dates_sorted[i]
        surge_ranges.append((start, dates_sorted[-1]))
        for s, e in surge_ranges:
            fig.add_vrect(x0=s, x1=e, fillcolor="rgba(198,40,40,0.08)",
                         line_width=0, annotation_text="surge", annotation_position="top left",
                         annotation_font_size=10, annotation_font_color="#c62828")

    fig.add_trace(go.Scatter(
        x=rsv_nat["Date"], y=rsv_nat["Cases"],
        fill="tozeroy", fillcolor="rgba(0,51,102,0.08)",
        line=dict(color="#003366", width=2),
        name="RSV cases", hovertemplate="%{x|%d %b %Y}<br>Cases: %{y:,}<extra></extra>"
    ))

    fig.update_layout(**PLOTLY_THEME, height=320,
        title=dict(text="Australia RSV cases — fortnightly (NNDSS)", font=dict(size=14)),
        xaxis=dict(title="", gridcolor="#f0f0f0"),
        yaxis=dict(title="Cases (fortnightly)", gridcolor="#f0f0f0"),
        legend=dict(orientation="h", y=-0.15),
        showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # ── State status grid ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">State-level status (latest available)</div>', unsafe_allow_html=True)
    st.caption("Based on state Rₜ from NNDSS fortnightly data")

    state_latest = rsv_st.dropna(subset=["Rt_NSW"]).iloc[-1]
    cols = st.columns(5)
    for i, st_code in enumerate(STATES):
        rt_val = state_latest.get(f"Rt_{st_code}", None)
        cases_val = int(state_latest.get(st_code, 0))
        if rt_val and not np.isnan(rt_val):
            if rt_val > 1.05:
                bg, icon, label = "#ffebee", "⚠", "Growing"
            elif rt_val > 0.95:
                bg, icon, label = "#fff8e1", "●", "Stable"
            else:
                bg, icon, label = "#e8f5e9", "▼", "Declining"
            cols[i].markdown(f"""
            <div style="background:{bg};padding:12px;border-radius:8px;text-align:center">
                <div style="font-size:11px;color:#666;font-weight:600">{st_code}</div>
                <div style="font-size:20px;font-weight:700;color:#003366">{icon} {rt_val:.3f}</div>
                <div style="font-size:11px;color:#666">{label}</div>
                <div style="font-size:11px;color:#999">{cases_val:,} cases</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
    <div class="finding-box">
        <strong>What is Rₜ?</strong> The reproduction number is the average number of new infections caused by
        one infected person at time t. Rₜ &gt; 1 means the epidemic is growing. Rₜ &lt; 1 means it is shrinking.
        Public health agencies should act when Rₜ crosses 1.0 — not when cases are already high.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EARLY WARNING SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
elif "warning" in page:

    st.markdown("## Early Warning Signals")
    st.markdown("*How far ahead can we predict a surge? The ML model provides 1–4 weeks of actionable lead time.*")
    st.markdown("")

    # ── Rₜ over time with Rₜ=1 threshold ─────────────────────────────────────
    st.markdown('<div class="section-header">Rₜ over time — the primary early warning signal</div>', unsafe_allow_html=True)

    rt_data = rsv_nat.dropna(subset=["Rt"])
    fig2 = go.Figure()
    fig2.add_hline(y=1.0, line_dash="dash", line_color="#333",
                   annotation_text="Rₜ = 1 (epidemic threshold)",
                   annotation_position="bottom right", annotation_font_size=11)
    fig2.add_trace(go.Scatter(
        x=rt_data["Date"], y=rt_data["Rt"],
        fill="tozeroy", fillcolor="rgba(0,51,102,0.07)",
        line=dict(color="#003366", width=2.5),
        name="National Rₜ",
        hovertemplate="%{x|%d %b %Y}<br>Rₜ = %{y:.3f}<extra></extra>"
    ))
    # Colour surges red
    growing = rt_data[rt_data["Rt"] > 1.0]
    fig2.add_trace(go.Scatter(
        x=growing["Date"], y=growing["Rt"],
        mode="markers", marker=dict(color="#c62828", size=7, symbol="circle"),
        name="Rₜ > 1 (growing)", hoverinfo="skip"
    ))
    fig2.update_layout(**PLOTLY_THEME, height=320,
        title="RSV Rₜ — national (growing = red dots)",
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(title="Rₜ", gridcolor="#f0f0f0", range=[0.8, 1.25]),
        legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig2, use_container_width=True)

    # ── State Rₜ comparison ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">State Rₜ — desynchronised epidemic waves</div>', unsafe_allow_html=True)

    fig3 = go.Figure()
    fig3.add_hline(y=1.0, line_dash="dash", line_color="#333", line_width=1)
    for st_code in STATES:
        col = f"Rt_{st_code}"
        valid = rsv_st.dropna(subset=[col])
        fig3.add_trace(go.Scatter(
            x=valid["Date"], y=valid[col],
            name=st_code, line=dict(color=STATE_COLORS[st_code], width=2),
            mode="lines+markers", marker=dict(size=4),
            hovertemplate=f"{st_code} %{{x|%d %b %Y}}<br>Rₜ = %{{y:.3f}}<extra></extra>"
        ))
    fig3.update_layout(**PLOTLY_THEME, height=360,
        title="State-level Rₜ — note NSW peaks before VIC and SA",
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(title="Rₜ", gridcolor="#f0f0f0", range=[0.8, 1.3]),
        legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div class="finding-box">
        <strong>Key finding:</strong> NSW Rₜ peaks approximately 4 weeks before VIC and 6–8 weeks before SA.
        Health agencies in lagging states can use NSW as a canary — when NSW Rₜ crosses 1.1,
        prepare VIC and SA for a surge within 4–8 weeks.
    </div>
    """, unsafe_allow_html=True)

    # ── Surge detection timeline ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Surge detection — confirmed surge periods</div>', unsafe_allow_html=True)

    merged = rsv_nat.merge(rsv_pred[["Date", "surge"]], on="Date", how="left")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=merged["Date"], y=merged["Cases"],
        fill="tozeroy", fillcolor="rgba(0,51,102,0.06)",
        line=dict(color="#003366", width=1.5), name="RSV cases", showlegend=True
    ))
    surge_mask = merged["surge"] == 1
    fig4.add_trace(go.Scatter(
        x=merged[surge_mask]["Date"], y=merged[surge_mask]["Cases"],
        mode="markers", marker=dict(color="#c62828", size=10, symbol="circle-open", line=dict(width=2)),
        name="Confirmed surge week"
    ))
    fig4.update_layout(**PLOTLY_THEME, height=300,
        title="RSV cases with confirmed surge fortnights highlighted",
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(title="Cases", gridcolor="#f0f0f0"),
        legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig4, use_container_width=True)

    # ── Forecast horizon ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Early warning horizon — how far ahead can we predict?</div>', unsafe_allow_html=True)

    lr_horizon = horizon[horizon["Model"] == "Logistic Regression"].sort_values("Horizon_weeks")
    rf_horizon = horizon[horizon["Model"] == "Random Forest"].sort_values("Horizon_weeks")

    fig5 = go.Figure()
    fig5.add_hrect(y0=0.7, y1=1.0, fillcolor="rgba(46,125,50,0.06)",
                   line_width=0, annotation_text="Actionable zone (AUC > 0.7)",
                   annotation_position="top left", annotation_font_size=10)
    fig5.add_hrect(y0=0.5, y1=0.7, fillcolor="rgba(245,127,23,0.06)",
                   line_width=0, annotation_text="Marginal",
                   annotation_position="top left", annotation_font_size=10)
    for model, df_h, color in [
        ("Logistic Regression", lr_horizon, "#003366"),
        ("Random Forest",       rf_horizon, "#c62828"),
    ]:
        fig5.add_trace(go.Scatter(
            x=df_h["Horizon_weeks"], y=df_h["AUC"],
            name=model, line=dict(color=color, width=2.5),
            mode="lines+markers", marker=dict(size=8),
            hovertemplate=f"{model}<br>Horizon: %{{x}} weeks<br>AUC: %{{y:.3f}}<extra></extra>"
        ))
    fig5.update_layout(**PLOTLY_THEME, height=340,
        title="Surge detection AUC vs forecast horizon (Australia COVID test set)",
        xaxis=dict(title="Weeks ahead", gridcolor="#f0f0f0",
                   tickvals=[1,2,3,4,6,8], ticktext=["1w","2w","3w","4w","6w","8w"]),
        yaxis=dict(title="ROC-AUC", gridcolor="#f0f0f0", range=[0.4, 1.05]),
        legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig5, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("AUC at 1 week", "0.975", "Strong early warning")
    c2.metric("AUC at 2 weeks", "0.906", "Useful lead time")
    c3.metric("AUC at 4 weeks", "0.711", "Actionable for agencies")

    st.caption("AUC > 0.7 = the model correctly ranks surge vs non-surge weeks 70%+ of the time at that horizon. "
               "The 4-week actionable window matches the lead time needed for public communications and hospital surge planning.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — STATE PROPAGATION
# ══════════════════════════════════════════════════════════════════════════════
elif "propagation" in page:

    st.markdown("## State Surge Propagation")
    st.markdown("*Does an RSV surge in one state predict a surge in another state weeks later?*")
    st.markdown("")

    st.markdown("""
    <div class="finding-box" style="font-size:15px;padding:14px 18px">
        <strong>Key finding:</strong> NSW Granger-causes RSV surges in VIC (p=0.002, 4-week lead),
        SA (p=0.001), and WA (p=0.007). VIC → SA shows the strongest correlation of any state pair
        (r=0.957 at 6-week lag). NSW is Australia's <em>canary state</em> for RSV surveillance.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Cross-correlation heatmap ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Lead-lag cross-correlation between states</div>', unsafe_allow_html=True)
    st.caption("Computed from state-level Rₜ series. Cell value = peak Pearson r at optimal lag. "
               "Row state leads column state.")

    # Recompute from data
    rt_matrix = rsv_st[["Date"] + [f"Rt_{s}" for s in STATES]].dropna().reset_index(drop=True)
    MAX_LAG = 4
    corr_mat = np.full((5, 5), np.nan)
    lag_mat  = np.full((5, 5), 0)

    for i, src in enumerate(STATES):
        for j, tgt in enumerate(STATES):
            if i == j:
                corr_mat[i, j] = 1.0
                continue
            x = rt_matrix[f"Rt_{src}"].values
            y = rt_matrix[f"Rt_{tgt}"].values
            best_r, best_lag = 0, 0
            for lag in range(0, MAX_LAG + 1):
                r = np.corrcoef(x[:-lag] if lag > 0 else x,
                                y[lag:]  if lag > 0 else y)[0, 1]
                if abs(r) > abs(best_r):
                    best_r, best_lag = r, lag
            corr_mat[i, j] = best_r
            lag_mat[i, j]  = best_lag

    text_labels = [[f"{corr_mat[i,j]:.2f}<br>L{lag_mat[i,j]}" for j in range(5)] for i in range(5)]

    fig6 = go.Figure(go.Heatmap(
        z=corr_mat, x=STATES, y=STATES,
        colorscale="RdYlGn", zmin=-1, zmax=1,
        text=text_labels, texttemplate="%{text}",
        textfont=dict(size=12),
        hoverongaps=False,
        hovertemplate="Source: %{y} → Target: %{x}<br>r = %{z:.3f}<extra></extra>"
    ))
    fig6.update_layout(**PLOTLY_THEME, height=380,
        title="Peak cross-correlation at optimal lag (Lx = x fortnights)",
        xaxis=dict(title="Target state", side="bottom"),
        yaxis=dict(title="Source state (leads →)"))
    st.plotly_chart(fig6, use_container_width=True)

    # ── Granger causality table ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Granger causality — statistically significant lead relationships</div>', unsafe_allow_html=True)

    granger_data = [
        ("NSW", "VIC", 1, 0.002, True,  "NSW surges 4 weeks before VIC — strongest canary signal"),
        ("NSW", "SA",  1, 0.001, True,  "NSW leads SA by 2+ fortnights"),
        ("NSW", "WA",  1, 0.007, True,  "Cross-continent lead confirmed"),
        ("VIC", "SA",  1, 0.000, True,  "Strongest pair — r=0.957 at 6-week lag"),
        ("VIC", "WA",  1, 0.000, True,  "VIC strongly predicts WA"),
        ("VIC", "NSW", 4, 0.011, True,  "Bidirectional — feedback loop"),
        ("QLD", "NSW", 3, 0.043, True,  "QLD leads NSW by 6 weeks"),
        ("QLD", "WA",  1, 0.035, True,  "Northern states influence west"),
        ("NSW", "QLD", 2, 0.153, False, "Not significant"),
        ("QLD", "VIC", 1, 0.147, False, "Not significant"),
    ]

    df_g = pd.DataFrame(granger_data,
        columns=["Source", "Target", "Best lag (fn)", "p-value", "Significant", "Interpretation"])

    sig_df   = df_g[df_g["Significant"]].copy()
    insig_df = df_g[~df_g["Significant"]].copy()

    for df_show, label in [(sig_df, "Significant (p < 0.05)"), (insig_df, "Not significant (p ≥ 0.05)")]:
        st.markdown(f"**{label}**")
        display = df_show[["Source", "Target", "Best lag (fn)", "p-value", "Interpretation"]].copy()
        display["p-value"] = display["p-value"].apply(lambda x: f"{x:.3f}")
        display["Best lag (fn)"] = display["Best lag (fn)"].apply(lambda x: f"{x} fn ({x*2}w)")
        st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Timeline comparing NSW vs VIC ──────────────────────────────────────────
    st.markdown('<div class="section-header">NSW vs VIC Rₜ — 4-week lead visualised</div>', unsafe_allow_html=True)

    valid = rsv_st.dropna(subset=["Rt_NSW", "Rt_VIC"]).copy()
    nsw_shifted = valid["Rt_NSW"].shift(-2)  # shift 2 fortnights = 4 weeks forward

    fig7 = go.Figure()
    fig7.add_hline(y=1.0, line_dash="dash", line_color="#888", line_width=1)
    fig7.add_trace(go.Scatter(
        x=valid["Date"], y=valid["Rt_NSW"],
        name="NSW Rₜ (actual)", line=dict(color="#1f77b4", width=2.5),
        hovertemplate="NSW %{x|%d %b %Y}<br>Rₜ = %{y:.3f}<extra></extra>"
    ))
    fig7.add_trace(go.Scatter(
        x=valid["Date"], y=valid["Rt_VIC"],
        name="VIC Rₜ (actual)", line=dict(color="#d62728", width=2.5),
        hovertemplate="VIC %{x|%d %b %Y}<br>Rₜ = %{y:.3f}<extra></extra>"
    ))
    fig7.add_trace(go.Scatter(
        x=valid["Date"], y=nsw_shifted,
        name="NSW Rₜ shifted +4 weeks", line=dict(color="#1f77b4", width=1.5, dash="dot"),
        hovertemplate="NSW shifted %{x|%d %b %Y}<br>Rₜ = %{y:.3f}<extra></extra>"
    ))
    fig7.update_layout(**PLOTLY_THEME, height=340,
        title="NSW Rₜ leads VIC Rₜ by 4 weeks (dotted = NSW shifted forward)",
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(title="Rₜ", gridcolor="#f0f0f0"),
        legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("""
    <div class="finding-box">
        <strong>Recommended protocol for public health agencies:</strong>
        When NSW Rₜ exceeds 1.10 for two consecutive fortnights, issue a preparedness advisory
        to VIC (4-week lead time) and SA/WA (6–8 week lead time).
        Combine with the ML model's own 4-week surge probability for confirmation.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL & METHODS
# ══════════════════════════════════════════════════════════════════════════════
elif "methods" in page:

    st.markdown("## Model & Methods")
    st.markdown("*Technical documentation — for scientists and reviewers*")
    st.markdown("")

    # ── Model overview cards ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">System architecture</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown("""**Training data**  
239 countries · WHO COVID-19  
74,090 weekly records  
SMOTE-balanced (50/50 surge)""")
    c2.markdown("""**Features (16 total)**  
Rt (46% importance)  
6 lag features · rolling stats  
Z-score · seasonality encoding""")
    c3.markdown("""**Test sets**  
Test 1: Australia COVID (geographic)  
Test 2: Australia RSV (cross-disease)  
Held out completely from training""")

    # ── Feature importance ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Feature importance — what drives surge detection?</div>', unsafe_allow_html=True)

    feat_sorted = feat_imp.sort_values("Importance", ascending=True)
    colors_fi   = ["#c62828" if f == "Rt" else "#003366" if imp > 0.05 else "#90a4ae"
                   for f, imp in zip(feat_sorted["Feature"], feat_sorted["Importance"])]

    fig8 = go.Figure(go.Bar(
        x=feat_sorted["Importance"], y=feat_sorted["Feature"],
        orientation="h", marker_color=colors_fi,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"
    ))
    fig8.add_vline(x=feat_imp["Importance"].mean(), line_dash="dash",
                   line_color="#f57f17", annotation_text="mean",
                   annotation_position="top")
    fig8.update_layout(**PLOTLY_THEME, height=420,
        title="Feature importance — Random Forest (best model on AUS COVID)",
        xaxis=dict(title="Importance", gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0"))
    st.plotly_chart(fig8, use_container_width=True)

    st.caption("Rt alone accounts for 46% of importance — confirming that the time-varying reproduction "
               "number is the single best predictor of surge onset. growth_4w (4-week growth rate) "
               "and z-score capture additional momentum.")

    # ── Cross-disease transfer ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Cross-disease transfer — COVID model on RSV</div>', unsafe_allow_html=True)

    cross_data = {
        "Model":     ["Logistic Regression", "Gradient Boosting", "Random Forest", "XGBoost"],
        "AUC (RSV)": [0.923,                  0.881,               0.549,           0.518],
        "F1 (before recal.)": [0.000, 0.000, 0.000, 0.000],
        "F1 (Fix A)":         [0.848, 0.612, 0.400, 0.320],
    }
    df_cross = pd.DataFrame(cross_data)

    fig9 = go.Figure()
    fig9.add_trace(go.Bar(
        x=df_cross["Model"], y=df_cross["AUC (RSV)"],
        name="AUC (primary metric)", marker_color="#003366",
        hovertemplate="%{x}<br>AUC = %{y:.3f}<extra></extra>"
    ))
    fig9.add_trace(go.Bar(
        x=df_cross["Model"], y=df_cross["F1 (Fix A)"],
        name="F1 after threshold recalibration", marker_color="#c62828",
        hovertemplate="%{x}<br>F1 = %{y:.3f}<extra></extra>"
    ))
    fig9.add_hline(y=0.7, line_dash="dash", line_color="#2e7d32",
                   annotation_text="AUC threshold for actionable use")
    fig9.update_layout(**PLOTLY_THEME, height=360, barmode="group",
        title="COVID-trained model on Australia RSV — cross-disease generalisation",
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(title="Score", gridcolor="#f0f0f0", range=[0, 1.1]),
        legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig9, use_container_width=True)

    st.markdown("""
    <div class="finding-box">
        <strong>Why F1 was 0 before recalibration:</strong>
        The model's probabilities are calibrated on COVID scales — for RSV, all predicted probabilities
        fall below 0.5 even for true surge weeks. AUC=0.923 proves the <em>ranking</em> is correct
        (surge weeks are ranked above non-surge weeks 92% of the time). Fix A: finding the optimal
        decision threshold from the PR curve brings F1 to 0.848. This is a known challenge in
        cross-disease transfer — the ranking transfers, the scale does not.
    </div>
    """, unsafe_allow_html=True)

    # ── Model summary table ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Full results summary</div>', unsafe_allow_html=True)

    summary = pd.DataFrame({
        "Test": ["AUS COVID (h=0)", "AUS COVID (h=0)", "AUS COVID (h=0)", "AUS COVID (h=0)",
                 "AUS RSV (cross-disease)", "AUS RSV (cross-disease)"],
        "Model": ["Random Forest", "Gradient Boosting", "Logistic Regression", "XGBoost",
                  "Logistic Regression", "Gradient Boosting"],
        "AUC":  [1.000, 1.000, 0.995, 1.000, 0.923, 0.881],
        "F1":   [1.000, 0.981, 0.964, 0.941, 0.848, 0.612],
        "Note": ["","","","","After threshold recalibration","After threshold recalibration"],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        "**Data sources:** WHO COVID-19 Global Data (Jan 2020–Jan 2026) · "
        "NNDSS Fortnightly Reports (Dec 2023–Nov 2025) · ABS Population Estimates 2023  \n"
        "**Model:** Random Forest Classifier (200 trees) + Logistic Regression · "
        "SMOTE for class imbalance · Serial interval = 5.5 days for Rₜ estimation  \n"
        "**Code:** github.com/inferra · IST 718 Group Project"
    )
