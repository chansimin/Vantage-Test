"""APAC Data Centre Market Prioritisation — Dual Cloud / AI Dashboard.

Run with:
    streamlit run dashboard/app.py

Requires ANTHROPIC_API_KEY in environment (or .env file) for AI narrative
generation. Deterministic scoring works without an API key.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from utils.cloud_scoring import (
    score_all_cloud_submarkets,
    get_cloud_band,
    CLOUD_AZ_HARD_LIMIT_KM,
)
from utils.ai_scoring import (
    score_all_ai_submarkets,
    get_ai_band,
    HYPERSCALER_PROFILES,
)
from utils.data_loader import get_all_submarket_packages
from agents.orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="APAC DC Market Prioritisation",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
:root {
  --mc-blue:#051C2C; --mc-teal:#00A9CE; --mc-grey:#4A4F55;
  --mc-silver:#D3D3D3; --mc-green:#00B388; --mc-amber:#FFB81C;
  --mc-red:#E34946; --mc-purple:#7B2D8B;
}
.mc-header{background:var(--mc-blue);padding:1rem 1.8rem;border-radius:4px;margin-bottom:1.2rem}
.mc-header h1{color:white;margin:0;font-size:1.45rem}
.mc-header p{color:var(--mc-silver);margin:.15rem 0 0;font-size:.82rem}
.use-case-badge-cloud{background:#00A9CE;color:white;padding:3px 12px;border-radius:12px;font-weight:700;font-size:.8rem}
.use-case-badge-ai{background:#7B2D8B;color:white;padding:3px 12px;border-radius:12px;font-weight:700;font-size:.8rem}
.metric-card{background:white;border-left:4px solid var(--mc-teal);padding:.9rem 1.1rem;border-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.metric-card .lbl{font-size:.72rem;color:var(--mc-grey);text-transform:uppercase;letter-spacing:.05em}
.metric-card .val{font-size:1.6rem;font-weight:700;color:var(--mc-blue)}
.metric-card .sub{font-size:.78rem;color:var(--mc-grey)}
.band-cloud1{background:#00B388;color:white;padding:2px 9px;border-radius:10px;font-size:.76rem}
.band-cloud2{background:#00A9CE;color:white;padding:2px 9px;border-radius:10px;font-size:.76rem}
.band-ai1   {background:#7B2D8B;color:white;padding:2px 9px;border-radius:10px;font-size:.76rem}
.band-ai2   {background:#A855F7;color:white;padding:2px 9px;border-radius:10px;font-size:.76rem}
.band-watch {background:#FFB81C;color:#051C2C;padding:2px 9px;border-radius:10px;font-size:.76rem}
.band-marg  {background:#E34946;color:white;padding:2px 9px;border-radius:10px;font-size:.76rem}
.section-title{font-size:.9rem;font-weight:700;color:var(--mc-blue);text-transform:uppercase;letter-spacing:.07em;border-bottom:2px solid var(--mc-teal);padding-bottom:.25rem;margin-bottom:.9rem}
.narrative-box{background:white;border:1px solid var(--mc-silver);border-top:4px solid var(--mc-teal);padding:1.2rem 1.5rem;border-radius:4px;line-height:1.65}
.hard-filter-badge{background:#E34946;color:white;padding:1px 7px;border-radius:8px;font-size:.72rem}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------

def _init():
    defaults = {
        "sm_packages": None,
        "cloud_scores": None,
        "ai_scores": None,
        "orch": Orchestrator(),
        "narratives": {},
        "ai_log": [],
        "ai_running": False,
        "last_ai_run": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ---------------------------------------------------------------------------
# Load sub-market packages once
# ---------------------------------------------------------------------------

if st.session_state.sm_packages is None:
    st.session_state.sm_packages = get_all_submarket_packages()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _color(v: float) -> str:
    if v >= 72: return "color:#00B388;font-weight:700"
    if v >= 55: return "color:#00A9CE;font-weight:600"
    if v >= 40: return "color:#FFB81C"
    return "color:#E34946"


BAND_CSS = {
    "Tier 1 Cloud": "band-cloud1",
    "Tier 2 Cloud": "band-cloud2",
    "Tier 1 AI":    "band-ai1",
    "Tier 2 AI":    "band-ai2",
    "Watch":        "band-watch",
    "Marginal":     "band-marg",
}


def _badge(text: str, css: str) -> str:
    return f"<span class='{css}'>{text}</span>"


HS_COLORS = {
    "AWS":    "#FF9900",
    "AZURE":  "#0078D4",
    "GOOGLE": "#4285F4",
    "ORACLE": "#F80000",
    "META":   "#1877F2",
}

TYPE_LABELS = {
    "cloud_core":    "Cloud Core",
    "cloud_stretch": "Cloud Stretch",
    "ai_stretch":    "AI Stretch",
    "regional":      "Regional",
}
TYPE_COLORS = {
    "cloud_core":    "#00A9CE",
    "cloud_stretch": "#00B388",
    "ai_stretch":    "#7B2D8B",
    "regional":      "#9CA3AF",
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(f"""
<div class="mc-header">
  <h1>🏢 APAC Data Centre Market Prioritisation</h1>
  <p>McKinsey Infrastructure Practice &nbsp;|&nbsp; Sub-market level analysis &nbsp;|&nbsp;
     {datetime.now().strftime('%B %Y')} &nbsp;|&nbsp;
     Cloud &amp; AI use-case bifurcation</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.divider()

    use_case = st.radio(
        "Use Case",
        options=["☁️ Cloud", "🤖 AI", "📊 Both"],
        index=2,
        help="Cloud: hard 10km fibre filter from hyperscaler AZ. AI: soft distance scoring.",
    )

    st.divider()

    if use_case in ("🤖 AI", "📊 Both"):
        st.markdown("**AI Hyperscaler Filter**")
        hs_filter = st.multiselect(
            "Show fit for",
            options=["AWS", "AZURE", "GOOGLE", "ORACLE", "META"],
            default=["GOOGLE", "ORACLE", "AWS", "AZURE"],
            format_func=lambda h: HYPERSCALER_PROFILES[h]["label"],
        )
        st.caption("Google & Oracle: location-agnostic, 250 MW+, <30m RFS")
        st.caption("AWS & Azure: cloud-stretch ≤50 km preferred")
        st.caption("Meta: campus own-build, power-cost primary")

        st.divider()
        st.markdown("**AI Score Weights**")
        w_rfs   = st.slider("Speed to Market (RFS)", 10, 50, 30, 5)
        w_pwr   = st.slider("Available Power",       10, 40, 25, 5)
        w_dist  = st.slider("Distance to AZ",        5,  35, 20, 5)
        st.caption(f"Remaining weight auto-allocated to capacity/land/cost/renewables.")
    else:
        hs_filter = ["AWS", "AZURE", "GOOGLE", "ORACLE"]
        w_rfs, w_pwr, w_dist = 30, 25, 20

    st.divider()

    if not _api_available():
        st.warning("Set `ANTHROPIC_API_KEY` for AI narratives.", icon="🔑")

    if st.button("🔄 Refresh Scores", use_container_width=True):
        st.session_state.cloud_scores = None
        st.session_state.ai_scores    = None
        st.rerun()

    st.divider()
    st.markdown("<small>McKinsey Infrastructure Practice<br>APAC DC Strategy</small>",
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Compute scores (cached)
# ---------------------------------------------------------------------------

pkgs = st.session_state.sm_packages

if st.session_state.cloud_scores is None:
    st.session_state.cloud_scores = score_all_cloud_submarkets(pkgs)

if st.session_state.ai_scores is None:
    st.session_state.ai_scores = score_all_ai_submarkets(pkgs)

cloud_scores: list = st.session_state.cloud_scores
ai_scores:    list = st.session_state.ai_scores

# Filtered AI scores by selected hyperscalers (show only those with interest)
if hs_filter:
    ai_display = sorted(
        [
            s for s in ai_scores
            if any(s.get(f"{h.lower()}_ai_fit", 0) > 0 for h in hs_filter)
        ],
        key=lambda x: max(
            (x.get(f"{h.lower()}_ai_fit", 0) or 0) for h in hs_filter
        ),
        reverse=True,
    )
else:
    ai_display = ai_scores

# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------

tier1_cloud = [s for s in cloud_scores if get_cloud_band(s["cloud_score"]) == "Tier 1 Cloud"]
tier1_ai    = [s for s in ai_scores    if get_ai_band(s["blended_ai_score"]) == "Tier 1 AI"]

c1, c2, c3, c4 = st.columns(4)
kpis = [
    (c1, "Cloud Submarkets Eligible", len(cloud_scores),
     f"{CLOUD_AZ_HARD_LIMIT_KM:.0f}km hard filter applied", "#00A9CE"),
    (c2, "Tier 1 Cloud Sites", len(tier1_cloud),
     ", ".join(s["submarket_name"] for s in tier1_cloud[:3]), "#00B388"),
    (c3, "AI Campuses Available", len(ai_scores),
     "across all distance bands", "#7B2D8B"),
    (c4, "Tier 1 AI Sites", len(tier1_ai),
     ", ".join(s["submarket_name"] for s in tier1_ai[:3]), "#A855F7"),
]
for col, label, val, sub, color in kpis:
    col.markdown(
        f"""<div class="metric-card" style="border-left-color:{color}">
          <div class="lbl">{label}</div>
          <div class="val" style="color:{color}">{val}</div>
          <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tabs_labels = {
    "📊 Both": ["☁️ Cloud Markets", "🤖 AI Markets", "📡 Hyperscaler Coverage", "🔍 Drill-Down", "📰 Market Intelligence", "ℹ️ Methodology"],
    "☁️ Cloud": ["☁️ Cloud Markets", "🔍 Drill-Down", "📰 Market Intelligence", "ℹ️ Methodology"],
    "🤖 AI":    ["🤖 AI Markets", "📡 Hyperscaler Coverage", "🔍 Drill-Down", "📰 Market Intelligence", "ℹ️ Methodology"],
}

tab_list = st.tabs(tabs_labels[use_case])
tab_idx  = {name: i for i, name in enumerate(tabs_labels[use_case])}

def get_tab(name: str):
    idx = tab_idx.get(name)
    return tab_list[idx] if idx is not None else None


# ============================================================
# CLOUD MARKETS TAB
# ============================================================

cloud_tab = get_tab("☁️ Cloud Markets")
if cloud_tab:
    with cloud_tab:
        st.markdown(
            "<div class='section-title'>☁️ Cloud-Core & Cloud-Stretch Submarkets "
            "<span class='hard-filter-badge'>Hard filter: ≤10km fibre from AZ</span></div>",
            unsafe_allow_html=True,
        )

        # Region filter
        regions = sorted({s["region"] for s in cloud_scores})
        sel_regions = st.multiselect("Filter by region", regions, default=[], key="c_region")
        filtered_c  = [s for s in cloud_scores if not sel_regions or s["region"] in sel_regions]

        # League table
        for s in filtered_c:
            band = get_cloud_band(s["cloud_score"])
            css  = BAND_CSS.get(band, "band-watch")
            ai_b = "🤖" if s.get("ai_enhanced") else ""

            cols = st.columns([0.5, 2.5, 1.4, 1.2, 0.9, 0.9, 0.9, 0.9, 0.9])
            cols[0].markdown(
                f"<div style='font-size:1.2rem;font-weight:700;color:#051C2C;padding-top:5px'>"
                f"#{s['cloud_rank']}</div>", unsafe_allow_html=True)
            cols[1].markdown(
                f"**{s['submarket_name']}** {ai_b}<br>"
                f"<small style='color:#4A4F55'>{s['city']} · {s['country']} · "
                f"{TYPE_LABELS.get(s['submarket_type'],s['submarket_type'])}"
                f"</small>",
                unsafe_allow_html=True,
            )
            cols[2].markdown(f"<span class='{css}'>{band}</span>", unsafe_allow_html=True)
            cols[3].markdown(
                f"<div style='{_color(s['cloud_score'])};font-size:1.05rem'>"
                f"☁️ {s['cloud_score']:.1f}</div>", unsafe_allow_html=True)
            for i, (h, col_) in enumerate(zip(
                ["AWS","AZURE","GOOGLE","ORACLE"], cols[4:]
            )):
                v = s.get(f"{h.lower()}_cloud_fit") or 0
                cols[4+i].markdown(
                    f"<div style='font-size:.85rem;color:{HS_COLORS[h]}'>"
                    f"{HYPERSCALER_PROFILES[h]['label']}: <b>{v:.0f}</b></div>",
                    unsafe_allow_html=True,
                )
            st.divider()

        # Heatmap — cloud scores by submarket
        st.markdown("<div class='section-title'>Cloud Score Heatmap</div>",
                    unsafe_allow_html=True)

        sm_names  = [s["submarket_name"] for s in filtered_c]
        aws_vals  = [s.get("aws_cloud_fit")    or 0 for s in filtered_c]
        az_vals   = [s.get("azure_cloud_fit")  or 0 for s in filtered_c]
        goog_vals = [s.get("google_cloud_fit") or 0 for s in filtered_c]
        ora_vals  = [s.get("oracle_cloud_fit") or 0 for s in filtered_c]
        base_vals = [s["cloud_score"]                 for s in filtered_c]

        fig_ch = go.Figure(go.Heatmap(
            z=[base_vals, aws_vals, az_vals, goog_vals, ora_vals],
            x=sm_names,
            y=["Base Cloud", "AWS Fit", "Azure Fit", "Google Fit", "Oracle Fit"],
            colorscale=[[0,"#E34946"],[0.5,"#FFB81C"],[0.75,"#00A9CE"],[1,"#00B388"]],
            zmin=0, zmax=100,
            text=[[f"{v:.0f}" for v in row] for row in
                  [base_vals,aws_vals,az_vals,goog_vals,ora_vals]],
            texttemplate="%{text}", textfont={"size":10,"color":"white"},
        ))
        fig_ch.update_layout(
            height=260, margin=dict(l=5,r=5,t=10,b=5),
            xaxis=dict(tickangle=-35,tickfont_size=10),
        )
        st.plotly_chart(fig_ch, use_container_width=True)

        # Scatter: proximity vs score
        st.markdown("<div class='section-title'>Proximity vs Cloud Score</div>",
                    unsafe_allow_html=True)

        c_df = pd.DataFrame(filtered_c)
        fig_cs = px.scatter(
            c_df, x="fibre_distance_km", y="cloud_score",
            size="available_power_mw", color="hyperscaler_count",
            hover_name="submarket_name",
            hover_data={"city":True,"cloud_score":":.1f","fibre_distance_km":":.1f",
                        "available_power_mw":True,"power_committed_pct":True},
            labels={"fibre_distance_km":"Fibre Distance to AZ (km)","cloud_score":"Cloud Score"},
            text="submarket_code",
            color_continuous_scale=["#E34946","#FFB81C","#00B388"],
            range_color=[0,4], size_max=35,
        )
        fig_cs.update_traces(textposition="top center", textfont_size=8)
        fig_cs.add_vline(x=CLOUD_AZ_HARD_LIMIT_KM, line_dash="dash",
                         line_color="#E34946", line_width=1.5,
                         annotation_text=f"Hard limit {CLOUD_AZ_HARD_LIMIT_KM:.0f}km",
                         annotation_font_size=9)
        fig_cs.update_layout(height=380, margin=dict(l=5,r=5,t=10,b=5),
                              plot_bgcolor="#F5F6F7", paper_bgcolor="white")
        st.plotly_chart(fig_cs, use_container_width=True)


# ============================================================
# AI MARKETS TAB
# ============================================================

ai_tab = get_tab("🤖 AI Markets")
if ai_tab:
    with ai_tab:
        st.markdown(
            "<div class='section-title'>🤖 AI Campus Submarkets "
            "<small style='text-transform:none;font-weight:400'>"
            "— Soft distance scoring (0-150+ km), hard on power &amp; RFS</small></div>",
            unsafe_allow_html=True,
        )

        # Filters
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            sel_r = st.multiselect("Region", sorted({s["region"] for s in ai_display}),
                                   default=[], key="ai_region")
        with fcol2:
            max_dist = st.slider("Max distance to AZ (km)", 0, 200, 200, 10)
        with fcol3:
            min_mw = st.slider("Min available power (MW)", 0, 700, 0, 50)

        filtered_ai = [
            s for s in ai_display
            if (not sel_r or s["region"] in sel_r)
            and s["distance_to_az_km"] <= max_dist
            and s["available_power_mw"] >= min_mw
        ]

        # Re-rank after filter
        for i, s in enumerate(filtered_ai):
            s["_display_rank"] = i + 1

        # League table
        for s in filtered_ai:
            band = get_ai_band(s["blended_ai_score"])
            css  = BAND_CSS.get(band, "band-watch")
            t_color = TYPE_COLORS.get(s["submarket_type"], "#9CA3AF")

            cols = st.columns([0.5, 2.5, 1.3, 1.0, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9])
            cols[0].markdown(
                f"<div style='font-size:1.1rem;font-weight:700;color:#051C2C;padding-top:5px'>"
                f"#{s['_display_rank']}</div>", unsafe_allow_html=True)
            cols[1].markdown(
                f"**{s['submarket_name']}**<br>"
                f"<small style='color:#4A4F55'>{s['city']} · {s['country']} · "
                f"<span style='color:{t_color};font-weight:600'>"
                f"{TYPE_LABELS.get(s['submarket_type'],s['submarket_type'])}"
                f"</span></small>",
                unsafe_allow_html=True,
            )
            cols[2].markdown(f"<span class='{css}'>{band}</span>", unsafe_allow_html=True)
            cols[3].markdown(
                f"<div style='{_color(s[\"blended_ai_score\"])};font-size:1.0rem'>"
                f"🤖 {s['blended_ai_score']:.1f}</div>", unsafe_allow_html=True)
            # Key metrics
            cols[4].markdown(
                f"<div style='font-size:.8rem'><b>{s['available_power_mw']}</b> MW</div>",
                unsafe_allow_html=True)
            cols[5].markdown(
                f"<div style='font-size:.8rem;color:{'#00B388' if s['shortest_rfs_months']<=30 else '#FFB81C'}'>"
                f"<b>{s['shortest_rfs_months']}m</b> RFS</div>",
                unsafe_allow_html=True)
            # Hyperscaler fits
            for i, h in enumerate(["AWS","AZURE","GOOGLE","ORACLE","META"]):
                v = s.get(f"{h.lower()}_ai_fit") or 0
                cols[6+i].markdown(
                    f"<div style='font-size:.78rem;color:{HS_COLORS[h]}'>"
                    f"{HYPERSCALER_PROFILES[h]['label']}: <b>{v:.0f}</b></div>",
                    unsafe_allow_html=True)
            st.divider()

        # Column key
        with st.expander("Column guide"):
            st.markdown(
                "**Score** = Blended AI score (60% base model, 40% avg hyperscaler fit)  \n"
                "**MW** = Available uncommitted power  \n"
                "**RFS** = Shortest ready-for-service (months) — 🟢 ≤30m (benchmark)  \n"
                "**Hyperscaler scores** = Use-case fit 0–100 per profile"
            )

        # Scatter: power vs RFS coloured by distance
        st.markdown("<div class='section-title'>Power vs Speed-to-Market</div>",
                    unsafe_allow_html=True)

        ai_df = pd.DataFrame(filtered_ai)
        if not ai_df.empty:
            fig_ai = px.scatter(
                ai_df,
                x="shortest_rfs_months",
                y="available_power_mw",
                size="max_single_site_mw",
                color="distance_to_az_km",
                hover_name="submarket_name",
                hover_data={"city":True,"country":True,
                            "blended_ai_score":":.1f",
                            "shortest_rfs_months":True,
                            "available_power_mw":True,
                            "max_single_site_mw":True,
                            "renewable_pct":True},
                labels={
                    "shortest_rfs_months": "Shortest RFS (months)",
                    "available_power_mw":  "Available Uncommitted Power (MW)",
                    "distance_to_az_km":   "Distance to Cloud AZ (km)",
                },
                text="submarket_code",
                color_continuous_scale=[[0,"#00B388"],[0.5,"#FFB81C"],[1,"#E34946"]],
                range_color=[0, 150], size_max=45,
            )
            fig_ai.update_traces(textposition="top center", textfont_size=8)
            # Reference lines
            fig_ai.add_vline(x=30, line_dash="dash", line_color="#7B2D8B", line_width=1.5,
                             annotation_text="30m RFS benchmark (Google/Oracle)",
                             annotation_font_size=9)
            fig_ai.add_hline(y=250, line_dash="dot", line_color="#FF9900", line_width=1,
                             annotation_text="250MW threshold (Google/Oracle)",
                             annotation_font_size=9, annotation_position="top right")
            fig_ai.update_layout(
                height=420, margin=dict(l=5,r=5,t=10,b=5),
                plot_bgcolor="#F5F6F7", paper_bgcolor="white",
                coloraxis_colorbar=dict(title="AZ Dist (km)"),
            )
            st.plotly_chart(fig_ai, use_container_width=True)

        # Submarket type distribution
        st.markdown("<div class='section-title'>Submarket Type Breakdown</div>",
                    unsafe_allow_html=True)

        type_counts = ai_df["submarket_type"].value_counts().reset_index()
        type_counts.columns = ["type", "count"]
        type_counts["label"] = type_counts["type"].map(TYPE_LABELS)
        type_counts["color"] = type_counts["type"].map(TYPE_COLORS)

        fig_pie = px.bar(
            type_counts, x="label", y="count", color="type",
            color_discrete_map=TYPE_COLORS,
            labels={"label":"Submarket Type","count":"Count"},
        )
        fig_pie.update_layout(height=240, showlegend=False,
                              margin=dict(l=5,r=5,t=10,b=5),
                              plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_pie, use_container_width=True)


# ============================================================
# HYPERSCALER COVERAGE TAB
# ============================================================

hs_tab = get_tab("📡 Hyperscaler Coverage")
if hs_tab:
    with hs_tab:
        st.markdown(
            "<div class='section-title'>📡 Hyperscaler Demand Profile Map</div>",
            unsafe_allow_html=True,
        )

        col_info, col_chart = st.columns([1, 2])

        with col_info:
            for h, profile in HYPERSCALER_PROFILES.items():
                st.markdown(
                    f"<div style='border-left:4px solid {profile['color']};"
                    f"padding:.5rem .8rem;margin-bottom:.6rem;background:white;"
                    f"border-radius:0 4px 4px 0'>"
                    f"<b style='color:{profile['color']}'>{profile['label']}</b><br>"
                    f"<small>{profile['description']}</small></div>",
                    unsafe_allow_html=True,
                )

        with col_chart:
            # Radar of average hyperscaler fit scores per submarket type
            type_hs = {}
            for t in ["cloud_core","cloud_stretch","ai_stretch"]:
                subset = [s for s in ai_scores if s["submarket_type"] == t]
                if subset:
                    type_hs[TYPE_LABELS[t]] = {
                        h: round(sum(s.get(f"{h.lower()}_ai_fit",0) or 0
                                     for s in subset) / len(subset), 1)
                        for h in ["AWS","AZURE","GOOGLE","ORACLE","META"]
                    }

            fig_radar = go.Figure()
            colors_radar = ["#FF9900","#0078D4","#4285F4","#F80000","#1877F2"]
            for h, color in zip(["AWS","AZURE","GOOGLE","ORACLE","META"], colors_radar):
                vals  = [type_hs.get(t, {}).get(h, 0) for t in list(type_hs.keys())]
                vals += [vals[0]]
                theta = list(type_hs.keys()) + [list(type_hs.keys())[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=theta, name=HYPERSCALER_PROFILES[h]["label"],
                    fill="none", line=dict(color=color, width=2),
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(range=[0,100],tickfont_size=8)),
                legend=dict(font_size=9), height=300,
                margin=dict(l=20,r=20,t=20,b=20), paper_bgcolor="white",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("<div class='section-title'>Top 5 Sites per Hyperscaler (AI)</div>",
                    unsafe_allow_html=True)

        hs_cols = st.columns(5)
        for col, h in zip(hs_cols, ["AWS","AZURE","GOOGLE","ORACLE","META"]):
            top5 = sorted(ai_scores, key=lambda s: s.get(f"{h.lower()}_ai_fit",0) or 0,
                          reverse=True)[:5]
            with col:
                st.markdown(
                    f"<div style='background:white;border-top:3px solid {HS_COLORS[h]};"
                    f"padding:.6rem .8rem;border-radius:0 0 4px 4px'>"
                    f"<b style='color:{HS_COLORS[h]}'>{HYPERSCALER_PROFILES[h]['label']}</b></div>",
                    unsafe_allow_html=True,
                )
                for s in top5:
                    v = s.get(f"{h.lower()}_ai_fit", 0) or 0
                    st.markdown(
                        f"<div style='padding:.2rem 0;font-size:.82rem'>"
                        f"<b>{s['submarket_name']}</b> "
                        f"<span style='color:{HS_COLORS[h]}'>{v:.0f}</span><br>"
                        f"<small style='color:#4A4F55'>{s['city']} · "
                        f"{s['available_power_mw']}MW · {s['shortest_rfs_months']}m RFS"
                        f"</small></div>",
                        unsafe_allow_html=True,
                    )


# ============================================================
# DRILL-DOWN TAB
# ============================================================

drill_tab = get_tab("🔍 Drill-Down")
if drill_tab:
    with drill_tab:
        st.markdown("<div class='section-title'>Sub-Market Deep Dive</div>",
                    unsafe_allow_html=True)

        # Build unified list
        all_sm_map = {}
        for s in cloud_scores:
            all_sm_map[s["submarket_code"]] = {"cloud": s, "ai": None}
        for s in ai_scores:
            code = s["submarket_code"]
            if code in all_sm_map:
                all_sm_map[code]["ai"] = s
            else:
                all_sm_map[code] = {"cloud": None, "ai": s}

        options = {
            f"{v['cloud']['submarket_name'] if v['cloud'] else v['ai']['submarket_name']} "
            f"({k}) — "
            f"{'☁️ ' + str(v['cloud']['cloud_score']) + ' | ' if v['cloud'] else ''}"
            f"{'🤖 ' + str(v['ai']['blended_ai_score']) if v['ai'] else ''}": k
            for k, v in all_sm_map.items()
        }
        sel_label = st.selectbox("Select a sub-market", list(options.keys()), index=0)
        sel_code  = options[sel_label]
        sel       = all_sm_map[sel_code]
        sm_cloud  = sel["cloud"]
        sm_ai     = sel["ai"]
        info      = (sm_cloud or sm_ai or {})

        sm_name   = info.get("submarket_name", sel_code)
        sm_city   = info.get("city", "")
        sm_type   = info.get("submarket_type", "")
        t_color   = TYPE_COLORS.get(sm_type, "#9CA3AF")

        # Header
        st.markdown(
            f"<div style='margin-bottom:.8rem'>"
            f"<span style='font-size:1.5rem;font-weight:700;color:#051C2C'>{sm_name}</span>"
            f"&nbsp;&nbsp;"
            f"<span style='background:{t_color};color:white;padding:2px 10px;"
            f"border-radius:10px;font-size:.8rem'>{TYPE_LABELS.get(sm_type,sm_type)}</span>"
            f"</div>"
            f"<div style='color:#4A4F55;margin-bottom:1.2rem'>"
            f"{sm_city} · {info.get('country','')} · {info.get('region','')}</div>",
            unsafe_allow_html=True,
        )

        # Score cards
        score_cols = st.columns(6)
        if sm_cloud:
            score_cols[0].markdown(
                f"<div style='background:white;border-top:3px solid #00A9CE;"
                f"padding:.6rem;text-align:center;border-radius:0 0 4px 4px'>"
                f"<div style='font-size:.65rem;color:#4A4F55;text-transform:uppercase'>☁️ Cloud</div>"
                f"<div style='font-size:1.4rem;font-weight:700;color:#00A9CE'>"
                f"{sm_cloud['cloud_score']:.1f}</div></div>", unsafe_allow_html=True)
            for i, (h, col_) in enumerate(zip(
                ["AWS","AZURE","GOOGLE","ORACLE"], score_cols[1:5]
            )):
                v = sm_cloud.get(f"{h.lower()}_cloud_fit") or 0
                col_.markdown(
                    f"<div style='background:white;border-top:3px solid {HS_COLORS[h]};"
                    f"padding:.6rem;text-align:center;border-radius:0 0 4px 4px'>"
                    f"<div style='font-size:.65rem;color:#4A4F55'>{HYPERSCALER_PROFILES[h]['label']} Cloud</div>"
                    f"<div style='font-size:1.2rem;font-weight:700;color:{HS_COLORS[h]}'>"
                    f"{v:.0f}</div></div>", unsafe_allow_html=True)

        if sm_ai:
            score_cols[5].markdown(
                f"<div style='background:white;border-top:3px solid #7B2D8B;"
                f"padding:.6rem;text-align:center;border-radius:0 0 4px 4px'>"
                f"<div style='font-size:.65rem;color:#4A4F55;text-transform:uppercase'>🤖 AI</div>"
                f"<div style='font-size:1.4rem;font-weight:700;color:#7B2D8B'>"
                f"{sm_ai['blended_ai_score']:.1f}</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Two-column detail
        d_col1, d_col2 = st.columns(2)

        if sm_cloud:
            with d_col1:
                st.markdown("**☁️ Cloud Metrics**")
                cd = next((p["cloud"] for p in pkgs if p["submarket_code"]==sel_code),{})
                st.markdown(
                    f"- Fibre distance to AZ: **{cd.get('fibre_distance_to_az_km','—')} km**\n"
                    f"- Carrier diversity: **{cd.get('carrier_diversity_score','—')} / 5 routes**\n"
                    f"- DC operators in submarket: **{cd.get('existing_dc_operators_count','—')}**\n"
                    f"- Available power: **{cd.get('available_power_mw','—')} MW** "
                    f"({cd.get('power_committed_pct','—')}% committed)\n"
                    f"- Colo pricing: **${cd.get('colo_pricing_usd_kw','—')}/kW/month**"
                )

        if sm_ai:
            with d_col2:
                st.markdown("**🤖 AI Campus Metrics**")
                ad = next((p["ai"] for p in pkgs if p["submarket_code"]==sel_code),{})
                rfs = ad.get('shortest_rfs_months','—')
                rfs_color = "#00B388" if isinstance(rfs, (int,float)) and rfs <= 30 else "#FFB81C"
                st.markdown(
                    f"- Distance to cloud AZ: **{ad.get('distance_to_cloud_az_km','—')} km**\n"
                    f"- Available uncommitted power: **{ad.get('available_uncommitted_power_mw','—')} MW**\n"
                    f"- Shortest RFS: **{rfs} months**\n"
                    f"- Max single-site capacity: **{ad.get('max_single_site_mw','—')} MW**\n"
                    f"- Land available: **{ad.get('land_available_ha','—')} ha**\n"
                    f"- Renewable energy: **{ad.get('renewable_energy_accessible_pct','—')}%**\n"
                    f"- Power cost: **${ad.get('power_cost_usd_mwh','—')}/MWh**\n"
                    f"- Campus-ready: **{'Yes ✅' if ad.get('campus_ready') else 'No (development required)'}**"
                )

        # Notes
        notes = info.get("notes", "")
        if notes:
            st.info(f"📋 {notes}")

        # Hyperscaler AI fit radar (if AI data available)
        if sm_ai:
            st.markdown("<br>**Hyperscaler AI Fit — Radar**")
            hs_vals  = [sm_ai.get(f"{h.lower()}_ai_fit",0) or 0
                        for h in ["AWS","AZURE","GOOGLE","ORACLE","META"]]
            hs_labels = [HYPERSCALER_PROFILES[h]["label"]
                         for h in ["AWS","AZURE","GOOGLE","ORACLE","META"]]

            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=hs_vals + [hs_vals[0]],
                theta=hs_labels + [hs_labels[0]],
                fill="toself", fillcolor="rgba(123,45,139,0.15)",
                line=dict(color="#7B2D8B", width=2),
            ))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True,range=[0,100],tickfont_size=8)),
                showlegend=False, height=280, margin=dict(l=20,r=20,t=20,b=20),
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_r, use_container_width=True)

        # Narrative
        st.markdown("<div class='section-title'>Executive Narrative</div>",
                    unsafe_allow_html=True)

        if sel_code in st.session_state.narratives:
            st.markdown(
                f"<div class='narrative-box'>{st.session_state.narratives[sel_code]}</div>",
                unsafe_allow_html=True,
            )
        elif not _api_available():
            st.info("Set `ANTHROPIC_API_KEY` to generate AI-powered narratives.", icon="🔑")
        else:
            if st.button(f"✍️ Generate narrative for {sm_name}", type="primary"):
                with st.spinner("Claude is writing the narrative…"):
                    tokens = []
                    ph     = st.empty()
                    try:
                        # Build a record compatible with orchestrator.stream_narrative
                        record = {
                            "market_code":     sel_code,
                            "market_name":     sm_name,
                            "country":         info.get("country",""),
                            "composite_score": sm_ai["blended_ai_score"] if sm_ai else sm_cloud["cloud_score"] if sm_cloud else 0,
                            "rank":            sm_ai.get("ai_rank") or sm_cloud.get("cloud_rank") or 1,
                            "demand_analysis": sm_cloud,
                            "supply_analysis": sm_ai,
                            "macro_analysis":  None,
                        }
                        for t in st.session_state.orch.stream_narrative(record, {"demand":0.33,"supply":0.33,"macro":0.34}):
                            tokens.append(t)
                            ph.markdown("".join(tokens))
                        st.session_state.narratives[sel_code] = "".join(tokens)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Narrative failed: {e}")


# ============================================================
# MARKET INTELLIGENCE TAB
# ============================================================

intel_tab = get_tab("📰 Market Intelligence")
if intel_tab:
    with intel_tab:
        from utils.data_loader import get_recent_announcements, load_announcements

        st.markdown(
            "<div class='section-title'>📰 APAC Market Intelligence — Recent Announcements</div>",
            unsafe_allow_html=True,
        )

        all_anns = load_announcements()

        # --- Summary KPIs ---
        demand_anns = [a for a in all_anns if a.get("category") == "demand"]
        supply_anns = [a for a in all_anns if a.get("category") == "supply"]
        total_inv   = sum(
            a.get("investment_usd_m") or 0 for a in all_anns
            if a.get("investment_usd_m")
        )
        total_mw    = sum(
            a.get("size_mw") or 0 for a in all_anns if a.get("size_mw")
        )

        ki1, ki2, ki3, ki4 = st.columns(4)
        for col, label, val, sub, color in [
            (ki1, "Total Announcements", len(all_anns), f"{len(demand_anns)} demand · {len(supply_anns)} supply", "#051C2C"),
            (ki2, "Demand Events", len(demand_anns), "leases, LOIs, pre-leases", "#00B388"),
            (ki3, "Supply Events", len(supply_anns), "construction, land, campus filings", "#7B2D8B"),
            (ki4, "Total MW Announced", f"{total_mw:,.0f}", f"USD {total_inv/1000:.1f}B investment tracked", "#00A9CE"),
        ]:
            col.markdown(
                f"""<div class="metric-card" style="border-left-color:{color}">
                  <div class="lbl">{label}</div>
                  <div class="val" style="color:{color}">{val}</div>
                  <div class="sub">{sub}</div>
                </div>""", unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Filters ---
        fi1, fi2, fi3 = st.columns(3)
        with fi1:
            cat_filter = st.selectbox(
                "Category", ["All", "demand", "supply"], key="ann_cat"
            )
        with fi2:
            cities = sorted({a.get("city", "") for a in all_anns if a.get("city")})
            city_filter = st.multiselect("City", cities, default=[], key="ann_city")
        with fi3:
            party_types = sorted({a.get("party_type", "") for a in all_anns if a.get("party_type")})
            pt_filter = st.multiselect("Party type", party_types, default=[], key="ann_pt")

        filtered_anns = [
            a for a in sorted(all_anns, key=lambda x: x.get("date", ""), reverse=True)
            if (cat_filter == "All" or a.get("category") == cat_filter)
            and (not city_filter or a.get("city") in city_filter)
            and (not pt_filter or a.get("party_type") in pt_filter)
        ]

        st.caption(f"Showing {len(filtered_anns)} of {len(all_anns)} announcements")

        # --- Bar chart: MW by city ---
        st.markdown("<div class='section-title'>Announced MW by City</div>",
                    unsafe_allow_html=True)
        city_mw = {}
        for a in filtered_anns:
            if a.get("size_mw"):
                city_mw[a.get("city", "Unknown")] = (
                    city_mw.get(a.get("city", "Unknown"), 0) + a["size_mw"]
                )
        if city_mw:
            mw_df = pd.DataFrame(
                sorted(city_mw.items(), key=lambda x: x[1], reverse=True),
                columns=["City", "MW"]
            )
            fig_mw = px.bar(
                mw_df, x="City", y="MW",
                color="MW",
                color_continuous_scale=[[0,"#00A9CE"],[1,"#7B2D8B"]],
                labels={"MW": "Announced MW"},
            )
            fig_mw.update_layout(
                height=280, margin=dict(l=5, r=5, t=10, b=5),
                showlegend=False, plot_bgcolor="#F5F6F7", paper_bgcolor="white",
            )
            st.plotly_chart(fig_mw, use_container_width=True)

        # --- Announcement feed ---
        st.markdown("<div class='section-title'>Announcement Feed</div>",
                    unsafe_allow_html=True)

        CAT_COLORS = {"demand": "#00B388", "supply": "#7B2D8B"}
        EVENT_ICONS = {
            "lease": "📋", "pre_lease": "🤝", "loi": "📝", "loi_signed": "✍️",
            "campus_announcement": "🏗️", "construction_start": "⚙️",
            "land_bank": "🏞️", "planning_approval": "✅", "expansion": "📈",
            "anchor_tenant": "⚓", "capacity_addition": "➕", "withdrawal": "🚫",
        }
        PARTY_TYPE_LABELS = {
            "hyperscaler": "Hyperscaler", "neocloud": "Neocloud",
            "dc_operator": "DC Operator", "developer": "Developer",
            "reit": "REIT", "government": "Government", "telco": "Telco",
        }

        for ann in filtered_anns:
            cat   = ann.get("category", "")
            color = CAT_COLORS.get(cat, "#9CA3AF")
            icon  = EVENT_ICONS.get(ann.get("event_type", ""), "📌")
            mw_str = f"**{ann['size_mw']:.0f} MW** · " if ann.get("size_mw") else ""
            inv_str = f"USD {ann['investment_usd_m']:.0f}M · " if ann.get("investment_usd_m") else ""
            pt_label = PARTY_TYPE_LABELS.get(ann.get("party_type", ""), ann.get("party_type", ""))
            verified = "✔" if ann.get("verified") else ""

            with st.container():
                h1, h2 = st.columns([0.15, 0.85])
                with h1:
                    st.markdown(
                        f"<div style='text-align:center;font-size:1.6rem;padding-top:4px'>{icon}</div>"
                        f"<div style='text-align:center;background:{color};color:white;"
                        f"border-radius:8px;padding:2px 6px;font-size:.7rem;font-weight:700'>"
                        f"{cat.upper()}</div>",
                        unsafe_allow_html=True,
                    )
                with h2:
                    st.markdown(
                        f"**{ann.get('party', '—')}** "
                        f"<span style='color:#4A4F55;font-size:.82rem'>({pt_label}) {verified}</span>  \n"
                        f"<small style='color:#4A4F55'>{ann.get('date','')} · "
                        f"{ann.get('city','')} · {ann.get('submarket_name','')}</small>  \n"
                        f"{mw_str}{inv_str}{ann.get('details','')}",
                        unsafe_allow_html=True,
                    )
                    if ann.get("source_url"):
                        st.caption(f"Source: [{ann.get('source', ann['source_url'])}]({ann['source_url']})")
                st.divider()


# ============================================================
# METHODOLOGY TAB
# ============================================================

method_tab = get_tab("ℹ️ Methodology")
if method_tab:
    with method_tab:
        st.markdown("""
## Sub-Market Architecture

This framework analyses APAC data centre markets at **sub-market** level (e.g. Macquarie Park, not
"Sydney") because Cloud and AI tenants have fundamentally different site selection criteria:

| Dimension | Cloud | AI Campus |
|-----------|-------|-----------|
| **Unit of analysis** | Sub-market within 10km fibre of hyperscaler AZ | Submarket with power + land at any distance ≤100km |
| **Sydney example** | Macquarie Park | Eastern Creek / Kemps Creek or Geelong / Moorabool |
| **Malaysia example** | Cyberjaya | Iskandar Puteri / Johor Bahru |
| **Distance constraint** | **Hard filter: ≤10km fibre** from AZ cluster | Soft: 0-50km ideal, 50-100km acceptable, >100km stretch |
| **Power threshold** | 50–200MW (constrained urban location) | **250MW+** (Google/Oracle minimum) |
| **RFS benchmark** | Established (in-service within 12-24m) | **≤30 months** (Google/Oracle critical threshold) |

---

## Cloud Scoring (0–100 after hard filter)

| Dimension | Weight | Logic |
|-----------|--------|-------|
| Proximity to AZ | 25% | 0km = 100, 10km = 0 (linear) |
| Carrier / fibre diversity | 22% | 5 diverse routes = 100 |
| Existing DC ecosystem | 20% | 10 operators = 100 |
| Available uncommitted power | 18% | 200MW = 100; adjusted for % committed |
| Colo pricing (yield proxy) | 15% | $380/kW = 100 |

**Hard filter**: Any submarket with `fibre_distance_to_az_km > 10` is excluded from the cloud
ranking entirely and appears only in the AI view.

---

## AI Scoring (0–100, no hard filter)

**Base model weights** (adjustable via sidebar):

| Dimension | Default | Notes |
|-----------|---------|-------|
| Speed to market (RFS months) | 30% | ≤18m = 100; 30m = 70; >48m decays steeply |
| Available uncommitted power | 25% | 700MW = 100 |
| Distance to cloud AZ | 20% | Soft band: 0-50km full, 50-100km -40pts, 100-150km -30pts |
| Max single-site capacity | 10% | 800MW = 100; Google/Oracle threshold 250MW |
| Land available (ha) | 5% | 320ha = 100 |
| Renewable energy access | 5% | % of power from RE sources |
| Power cost ($/MWh) | 5% | Lower = better; $150/MWh = 0 |

**Blended score** = 60% base model + 40% average hyperscaler fit score (interested hyperscalers only).

---

## Hyperscaler Profiles

| Hyperscaler | Distance tolerance | Power min | RFS criticality | Notes |
|-------------|-------------------|-----------|-----------------|-------|
| **Google** | 100km soft | 250MW+ | <30m critical | Location agnostic; largest AI campus deals |
| **Oracle** | 100km soft | 250MW+ | <30m critical | Fastest procurement in market; OCI region expansion |
| **AWS** | 50km cloud-stretch | 100–200MW | <24m strong | Closer to own AZ than Google; multi-AZ pattern |
| **Azure** | 50km cloud-stretch | 100–200MW | <24m strong | Carrier diversity critical; multi-region pattern |
| **Meta** | 150km | 200MW+ | <30m | Campus own-build; power cost + RE primary filter |

---

## Sub-Market Reference Points (Australia examples)

| Sub-market | Type | Distance to AZ | Use case |
|------------|------|----------------|----------|
| **Macquarie Park, SYD** | Cloud Core | 2km | Cloud (AWS, Azure, Google, Oracle) |
| **Eastern Creek / Kemps Creek, SYD** | Cloud Stretch | 42km | Cloud stretch + AI |
| **Docklands, MEL** | Cloud Core | 2km | Cloud (all hyperscalers) |
| **Moorabool / Lara, MEL** | AI Stretch | 82km | AI campus (Keppel) — 600MW, 22m RFS |
| **Geelong, MEL** | AI Stretch | 78km | AI campus — 400MW, 28m RFS |

---

## Data Sources
- Submarket definitions: DC Byte, JLL, CBRE, operator filings
- Hyperscaler cluster locations: public AZ announcements + colocation records
- Power data: utility reports, development authority publications
- RFS timelines: developer disclosures, planning portal applications
- Pricing: industry surveys, operator rate cards
""")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("""
<hr style="margin-top:2rem;border-color:#D3D3D3">
<div style="text-align:center;color:#4A4F55;font-size:.76rem;padding:.4rem 0 .8rem">
  McKinsey & Company — Infrastructure Practice — APAC DC Strategy Team<br>
  <em>Confidential and Proprietary</em>
</div>
""", unsafe_allow_html=True)
