"""APAC Data Centre Market Prioritisation Dashboard.

Run with:
    streamlit run dashboard/app.py

Requires ANTHROPIC_API_KEY in environment (or .env file) for AI analysis.
Quick scoring (no API key) is always available.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Make repo root importable when running from dashboard/
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(Path(__file__).parent.parent / ".env")

from agents.orchestrator import Orchestrator, DEFAULT_WEIGHTS
from utils.scoring import get_score_band

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
# Custom CSS — McKinsey-inspired palette
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
:root {
    --mc-blue:    #051C2C;
    --mc-teal:    #00A9CE;
    --mc-grey:    #4A4F55;
    --mc-silver:  #D3D3D3;
    --mc-green:   #00B388;
    --mc-amber:   #FFB81C;
    --mc-red:     #E34946;
    --mc-bg:      #F5F6F7;
}

/* Header */
.mc-header {
    background: var(--mc-blue);
    padding: 1.2rem 2rem;
    border-radius: 4px;
    margin-bottom: 1.5rem;
}
.mc-header h1 { color: white; margin: 0; font-size: 1.6rem; }
.mc-header p  { color: var(--mc-silver); margin: 0.2rem 0 0; font-size: 0.85rem; }

/* Score band chips */
.band-tier1  { background:#00B388; color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
.band-tier2  { background:#00A9CE; color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
.band-monitor{ background:#FFB81C; color:#051C2C; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
.band-deprio { background:#E34946; color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; }

/* Metric cards */
.metric-card {
    background: white;
    border-left: 4px solid var(--mc-teal);
    padding: 1rem 1.2rem;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,.08);
}
.metric-card .label { font-size:0.78rem; color:var(--mc-grey); text-transform:uppercase; letter-spacing:.05em; }
.metric-card .value { font-size:1.8rem; font-weight:700; color:var(--mc-blue); }
.metric-card .sub   { font-size:0.82rem; color:var(--mc-grey); }

/* Narrative box */
.narrative-box {
    background: white;
    border: 1px solid var(--mc-silver);
    border-top: 4px solid var(--mc-teal);
    padding: 1.4rem 1.6rem;
    border-radius: 4px;
    line-height: 1.65;
}

/* Section titles */
.section-title {
    font-size: 1.0rem;
    font-weight: 700;
    color: var(--mc-blue);
    text-transform: uppercase;
    letter-spacing: .07em;
    border-bottom: 2px solid var(--mc-teal);
    padding-bottom: .3rem;
    margin-bottom: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_state() -> None:
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = Orchestrator()
    if "weights" not in st.session_state:
        st.session_state.weights = DEFAULT_WEIGHTS.copy()
    if "quick_scores" not in st.session_state:
        st.session_state.quick_scores = None
    if "ai_results" not in st.session_state:
        st.session_state.ai_results = {}
    if "merged_scores" not in st.session_state:
        st.session_state.merged_scores = None
    if "ai_running" not in st.session_state:
        st.session_state.ai_running = False
    if "ai_log" not in st.session_state:
        st.session_state.ai_log = []
    if "last_run" not in st.session_state:
        st.session_state.last_run = None
    if "selected_market" not in st.session_state:
        st.session_state.selected_market = None
    if "narratives" not in st.session_state:
        st.session_state.narratives = {}
    if "portfolio_summary" not in st.session_state:
        st.session_state.portfolio_summary = None


init_state()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BAND_CSS = {
    "Tier 1 Priority": "band-tier1",
    "Tier 2 Priority": "band-tier2",
    "Monitor":         "band-monitor",
    "Deprioritise":    "band-deprio",
}

SCORE_COLORS = {
    "demand": "#00A9CE",
    "supply": "#00B388",
    "macro":  "#FFB81C",
}


def _scores_to_df(scores: List[Dict]) -> pd.DataFrame:
    cols = [
        "rank", "market_name", "country", "tier", "region",
        "composite_score", "demand_score", "supply_score", "macro_score",
        "score_band", "ai_enhanced",
    ]
    df = pd.DataFrame(scores)[cols]
    df.columns = [
        "Rank", "Market", "Country", "Tier", "Region",
        "Composite", "Demand", "Supply", "Macro",
        "Priority Band", "AI Enhanced",
    ]
    return df


def _color_score(val: float) -> str:
    if val >= 75: return "color:#00B388;font-weight:700"
    if val >= 62: return "color:#00A9CE;font-weight:600"
    if val >= 50: return "color:#FFB81C"
    return "color:#E34946"


def _api_key_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Analysis Controls")
    st.divider()

    st.markdown("**Dimension Weights**")
    st.caption("Weights are normalised to sum to 100%.")

    w_demand = st.slider("Demand Signals",   min_value=0, max_value=100, value=40, step=5)
    w_supply = st.slider("Competitive Supply", min_value=0, max_value=100, value=30, step=5)
    w_macro  = st.slider("Macro & Regulatory", min_value=0, max_value=100, value=30, step=5)

    total_w = w_demand + w_supply + w_macro
    if total_w == 0:
        total_w = 1  # avoid div-by-zero

    weights = {
        "demand": w_demand / total_w,
        "supply": w_supply / total_w,
        "macro":  w_macro  / total_w,
    }

    # Show normalised percentages
    st.caption(
        f"Effective: Demand {weights['demand']*100:.0f}% | "
        f"Supply {weights['supply']*100:.0f}% | "
        f"Macro {weights['macro']*100:.0f}%"
    )

    if weights != st.session_state.weights:
        st.session_state.weights = weights
        st.session_state.quick_scores  = None  # invalidate cache
        st.session_state.merged_scores = None

    st.divider()
    st.markdown("**AI-Powered Analysis**")

    if not _api_key_available():
        st.warning(
            "Set `ANTHROPIC_API_KEY` to enable Claude-powered analysis.",
            icon="🔑",
        )
        ai_button_disabled = True
    else:
        ai_button_disabled = st.session_state.ai_running

    if st.button(
        "🤖 Run AI Analysis",
        disabled=ai_button_disabled,
        help="Run specialist Claude agents for all markets (~2-4 min)",
        use_container_width=True,
        type="primary",
    ):
        st.session_state.ai_running = True
        st.session_state.ai_log = []
        st.rerun()

    if st.session_state.last_run:
        st.caption(f"Last AI run: {st.session_state.last_run}")

    st.divider()
    st.markdown("**Data Freshness**")
    st.caption("📊 Structured data: March 2026")
    st.caption("🌐 Web enrichment: on request")

    st.divider()
    st.markdown(
        "<small>McKinsey Infrastructure Practice<br>"
        "APAC Data Centre Strategy</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Compute quick scores (cached until weights change)
# ---------------------------------------------------------------------------

if st.session_state.quick_scores is None:
    with st.spinner("Computing market scores…"):
        st.session_state.quick_scores = (
            st.session_state.orchestrator.run_quick_scoring(weights)
        )

# If we have AI results, merge them
if st.session_state.ai_results and st.session_state.merged_scores is None:
    st.session_state.merged_scores = st.session_state.orchestrator.merge_ai_analysis(
        st.session_state.quick_scores,
        st.session_state.ai_results,
        weights,
    )

active_scores = st.session_state.merged_scores or st.session_state.quick_scores

# ---------------------------------------------------------------------------
# AI analysis runner (triggered via sidebar button)
# ---------------------------------------------------------------------------

if st.session_state.ai_running:
    st.markdown(
        "<div class='section-title'>🤖 AI Analysis in Progress</div>",
        unsafe_allow_html=True,
    )
    log_container = st.empty()
    progress_bar  = st.progress(0.0)

    def _progress(msg: str) -> None:
        st.session_state.ai_log.append(msg)

    orch = st.session_state.orchestrator
    with st.spinner("Running Claude specialist agents across all APAC markets…"):
        try:
            results = orch.run_ai_analysis(progress_callback=_progress)
            st.session_state.ai_results = results
            st.session_state.merged_scores = orch.merge_ai_analysis(
                st.session_state.quick_scores, results, weights
            )
            st.session_state.last_run  = datetime.now().strftime("%d %b %Y %H:%M")
            st.session_state.ai_running = False
            st.success("AI analysis complete! Scores updated with Claude insights.", icon="✅")
            time.sleep(1)
            st.rerun()
        except Exception as exc:
            st.session_state.ai_running = False
            st.error(f"AI analysis failed: {exc}")

    # Show log
    log_container.code("\n".join(st.session_state.ai_log[-20:]))
    progress_bar.progress(1.0)
    st.stop()

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.markdown(
    f"""
<div class="mc-header">
  <h1>🏢 APAC Data Centre Market Prioritisation</h1>
  <p>McKinsey Infrastructure Practice &nbsp;|&nbsp;
     {datetime.now().strftime('%B %Y')} &nbsp;|&nbsp;
     {"🤖 AI-enhanced" if st.session_state.merged_scores else "📊 Quick scoring"}
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------

tier1 = [s for s in active_scores if s["score_band"] == "Tier 1 Priority"]
tier2 = [s for s in active_scores if s["score_band"] == "Tier 2 Priority"]
top1  = active_scores[0] if active_scores else {}

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""<div class="metric-card">
          <div class="label">Top Market</div>
          <div class="value">{top1.get('market_name','—')}</div>
          <div class="sub">Score: {top1.get('composite_score','—')}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""<div class="metric-card">
          <div class="label">Tier 1 Priority Markets</div>
          <div class="value">{len(tier1)}</div>
          <div class="sub">{', '.join(m['market_code'] for m in tier1[:4])}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""<div class="metric-card">
          <div class="label">Tier 2 Priority Markets</div>
          <div class="value">{len(tier2)}</div>
          <div class="sub">{', '.join(m['market_code'] for m in tier2[:4])}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with col4:
    ai_pct = (
        int(
            sum(1 for s in active_scores if s.get("ai_enhanced"))
            / len(active_scores)
            * 100
        )
        if active_scores
        else 0
    )
    st.markdown(
        f"""<div class="metric-card">
          <div class="label">AI Coverage</div>
          <div class="value">{ai_pct}%</div>
          <div class="sub">Claude specialist agents</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_league, tab_heatmap, tab_drilldown, tab_portfolio, tab_method = st.tabs([
    "📋 League Table",
    "🔥 Score Heatmap",
    "🔍 Market Drill-Down",
    "📄 Portfolio Summary",
    "ℹ️ Methodology",
])

# ============================================================
# TAB 1 — LEAGUE TABLE
# ============================================================

with tab_league:
    st.markdown(
        "<div class='section-title'>Market Prioritisation Rankings</div>",
        unsafe_allow_html=True,
    )

    # Filter controls
    fcol1, fcol2 = st.columns([1, 3])
    with fcol1:
        region_filter = st.multiselect(
            "Filter by Region",
            options=sorted({s["region"] for s in active_scores}),
            default=[],
        )
    with fcol2:
        tier_filter = st.multiselect(
            "Filter by Tier",
            options=[1, 2, 3],
            default=[],
        )

    filtered = active_scores
    if region_filter:
        filtered = [s for s in filtered if s["region"] in region_filter]
    if tier_filter:
        filtered = [s for s in filtered if s["tier"] in tier_filter]

    # Render table
    for record in filtered:
        band = record["score_band"]
        css  = BAND_CSS.get(band, "")
        ai_badge = "🤖" if record.get("ai_enhanced") else ""

        col_rank, col_market, col_band, col_comp, col_d, col_s, col_m = st.columns(
            [0.5, 2.5, 1.8, 1, 1, 1, 1]
        )

        col_rank.markdown(
            f"<div style='font-size:1.3rem;font-weight:700;color:#051C2C;padding-top:6px'>"
            f"#{record['rank']}</div>",
            unsafe_allow_html=True,
        )
        col_market.markdown(
            f"**{record['market_name']}** {ai_badge}<br>"
            f"<small style='color:#4A4F55'>{record['country']} · Tier {record['tier']}</small>",
            unsafe_allow_html=True,
        )
        col_band.markdown(
            f"<span class='{css}'>{band}</span>",
            unsafe_allow_html=True,
        )
        col_comp.markdown(
            f"<div style='{_color_score(record['composite_score'])};font-size:1.15rem'>"
            f"{record['composite_score']:.1f}</div>",
            unsafe_allow_html=True,
        )
        col_d.markdown(
            f"<div style='{_color_score(record['demand_score'])}'>"
            f"{record['demand_score']:.1f}</div>",
            unsafe_allow_html=True,
        )
        col_s.markdown(
            f"<div style='{_color_score(record['supply_score'])}'>"
            f"{record['supply_score']:.1f}</div>",
            unsafe_allow_html=True,
        )
        col_m.markdown(
            f"<div style='{_color_score(record['macro_score'])}'>"
            f"{record['macro_score']:.1f}</div>",
            unsafe_allow_html=True,
        )
        st.divider()

    # Column legend
    with st.expander("Column definitions"):
        st.markdown(
            """
| Column    | Definition |
|-----------|------------|
| **Composite** | Weighted composite of all three dimensions |
| **Demand**    | Demand signals score (hyperscaler activity, AI workload, cloud growth, enterprise demand, absorption) |
| **Supply**    | Supply opportunity score (vacancy tightness, pipeline risk, pricing, market depth) |
| **Macro**     | Macro & regulatory environment score (power, land, regulation, data sovereignty, stability) |
| 🤖           | Score has been refined by Claude AI agents |
"""
        )

# ============================================================
# TAB 2 — HEATMAP
# ============================================================

with tab_heatmap:
    st.markdown(
        "<div class='section-title'>Score Heatmap — All Markets × Dimensions</div>",
        unsafe_allow_html=True,
    )

    markets_ordered = [s["market_name"] for s in active_scores]
    demand_vals  = [s["demand_score"]  for s in active_scores]
    supply_vals  = [s["supply_score"]  for s in active_scores]
    macro_vals   = [s["macro_score"]   for s in active_scores]
    composite    = [s["composite_score"] for s in active_scores]

    z = [demand_vals, supply_vals, macro_vals, composite]
    y_labels = ["Demand Signals", "Competitive Supply", "Macro & Regulatory", "Composite Score"]

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=z,
            x=markets_ordered,
            y=y_labels,
            colorscale=[
                [0.0,  "#E34946"],
                [0.45, "#FFB81C"],
                [0.65, "#00A9CE"],
                [1.0,  "#00B388"],
            ],
            zmin=0,
            zmax=100,
            text=[[f"{v:.0f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 11, "color": "white"},
            hoverongaps=False,
            colorbar=dict(title="Score", thickness=12),
        )
    )
    fig_heatmap.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=20, b=20),
        xaxis=dict(tickangle=-35, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
        plot_bgcolor="#F5F6F7",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>Composite Score — Bar Chart</div>",
        unsafe_allow_html=True,
    )

    score_colors = [
        "#00B388" if s >= 75 else
        "#00A9CE" if s >= 62 else
        "#FFB81C" if s >= 50 else
        "#E34946"
        for s in composite
    ]
    fig_bar = go.Figure(
        go.Bar(
            x=markets_ordered,
            y=composite,
            marker_color=score_colors,
            text=[f"{v:.1f}" for v in composite],
            textposition="outside",
            textfont=dict(size=10),
        )
    )
    fig_bar.update_layout(
        height=320,
        yaxis=dict(range=[0, 100], title="Composite Score"),
        xaxis=dict(tickangle=-35),
        margin=dict(l=10, r=10, t=20, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    # Priority band reference lines
    for score, label, color in [
        (75, "Tier 1 threshold", "#00B388"),
        (62, "Tier 2 threshold", "#00A9CE"),
        (50, "Monitor threshold", "#FFB81C"),
    ]:
        fig_bar.add_hline(
            y=score,
            line_dash="dot",
            line_color=color,
            line_width=1,
            annotation_text=label,
            annotation_position="top right",
            annotation_font_size=9,
        )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>Demand vs Supply Opportunity Bubble Chart</div>",
        unsafe_allow_html=True,
    )

    bubble_df = pd.DataFrame(active_scores)
    fig_bubble = px.scatter(
        bubble_df,
        x="demand_score",
        y="supply_score",
        size="macro_score",
        color="composite_score",
        hover_name="market_name",
        hover_data={
            "country": True,
            "demand_score": ":.1f",
            "supply_score": ":.1f",
            "macro_score": ":.1f",
            "composite_score": ":.1f",
        },
        labels={
            "demand_score": "Demand Score",
            "supply_score": "Supply Opportunity Score",
            "composite_score": "Composite",
        },
        color_continuous_scale=[
            (0.0, "#E34946"),
            (0.5, "#FFB81C"),
            (0.75, "#00A9CE"),
            (1.0, "#00B388"),
        ],
        range_color=[40, 90],
        size_max=35,
        text="market_code",
    )
    fig_bubble.update_traces(textposition="top center", textfont_size=9)
    fig_bubble.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=20, b=20),
        plot_bgcolor="#F5F6F7",
        paper_bgcolor="white",
        coloraxis_colorbar=dict(title="Composite"),
    )
    # Reference quadrant lines
    fig_bubble.add_vline(x=65, line_dash="dash", line_color="#4A4F55", line_width=1)
    fig_bubble.add_hline(y=65, line_dash="dash", line_color="#4A4F55", line_width=1)

    # Quadrant annotations
    for (x, y, txt) in [
        (78, 82, "High Demand<br>High Opportunity"),
        (50, 82, "Emerging Demand<br>High Opportunity"),
        (78, 48, "High Demand<br>Saturating Supply"),
        (50, 48, "Monitor"),
    ]:
        fig_bubble.add_annotation(
            x=x, y=y, text=txt, showarrow=False,
            font=dict(size=9, color="#4A4F55"),
            align="center",
            bgcolor="rgba(255,255,255,0.6)",
            borderpad=3,
        )

    st.plotly_chart(fig_bubble, use_container_width=True)

# ============================================================
# TAB 3 — MARKET DRILL-DOWN
# ============================================================

with tab_drilldown:
    st.markdown(
        "<div class='section-title'>Market Deep-Dive</div>",
        unsafe_allow_html=True,
    )

    market_options = {
        f"#{s['rank']} {s['market_name']} ({s['market_code']}) — {s['composite_score']:.1f}": s
        for s in active_scores
    }
    selected_label = st.selectbox(
        "Select a market to analyse",
        options=list(market_options.keys()),
        index=0,
    )
    selected = market_options[selected_label]
    st.session_state.selected_market = selected["market_code"]

    # Score radar
    categories = ["Demand", "Supply\nOpportunity", "Macro &\nRegulatory"]
    scores_vals = [
        selected["demand_score"],
        selected["supply_score"],
        selected["macro_score"],
    ]

    dcol1, dcol2 = st.columns([1, 2])

    with dcol1:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=scores_vals + [scores_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(0,169,206,0.2)",
            line=dict(color="#00A9CE", width=2),
            name=selected["market_name"],
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont_size=9),
                angularaxis=dict(tickfont_size=10),
            ),
            showlegend=False,
            height=280,
            margin=dict(l=30, r=30, t=20, b=20),
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with dcol2:
        band    = selected["score_band"]
        css_cls = BAND_CSS.get(band, "")
        st.markdown(
            f"""
<div style="margin-bottom:.5rem">
  <span style="font-size:1.6rem;font-weight:700;color:#051C2C">{selected['market_name']}</span>
  &nbsp;&nbsp;
  <span class="{css_cls}">{band}</span>
  {"&nbsp;<span title='AI enhanced'>🤖</span>" if selected.get('ai_enhanced') else ""}
</div>
<div style="color:#4A4F55;margin-bottom:1rem">{selected['country']} · Tier {selected['tier']} · {selected['region']}</div>
""",
            unsafe_allow_html=True,
        )

        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, label, val, colour in [
            (mc1, "Composite", selected["composite_score"], "#051C2C"),
            (mc2, "Demand",    selected["demand_score"],    "#00A9CE"),
            (mc3, "Supply",    selected["supply_score"],    "#00B388"),
            (mc4, "Macro",     selected["macro_score"],     "#FFB81C"),
        ]:
            col.markdown(
                f"""<div style="background:white;border-top:3px solid {colour};
                    padding:.7rem;border-radius:4px;text-align:center;
                    box-shadow:0 1px 3px rgba(0,0,0,.07)">
                  <div style="font-size:.7rem;color:#4A4F55;text-transform:uppercase">{label}</div>
                  <div style="font-size:1.5rem;font-weight:700;color:{colour}">{val:.1f}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # AI analysis detail (if available)
    if selected.get("ai_enhanced"):
        col_d, col_s, col_m = st.columns(3)

        def _render_agent_card(col, title, analysis, icon, color):
            if not analysis:
                return
            with col:
                st.markdown(
                    f"<div style='font-weight:700;color:{color};margin-bottom:.4rem'>"
                    f"{icon} {title}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"*{analysis.get('rationale','')}*")
                if analysis.get("key_signals"):
                    st.markdown("**Signals:**")
                    for sig in analysis["key_signals"]:
                        st.markdown(f"- {sig}")
                if analysis.get("risk_flags"):
                    st.markdown("**Risks:**")
                    for rf in analysis["risk_flags"]:
                        st.markdown(f"- ⚠️ {rf}")
                confidence = analysis.get("confidence", "")
                if confidence:
                    st.caption(f"Confidence: {confidence}")

        _render_agent_card(
            col_d, "Demand Analysis",
            selected.get("demand_analysis"), "📈", "#00A9CE"
        )
        _render_agent_card(
            col_s, "Supply Analysis",
            selected.get("supply_analysis"), "🏗️", "#00B388"
        )
        _render_agent_card(
            col_m, "Macro Analysis",
            selected.get("macro_analysis"), "⚖️", "#FFB81C"
        )

        st.markdown("<br>", unsafe_allow_html=True)

    # Narrative section
    st.markdown(
        "<div class='section-title'>Executive Narrative</div>",
        unsafe_allow_html=True,
    )

    mcode = selected["market_code"]
    if mcode in st.session_state.narratives:
        st.markdown(
            f"<div class='narrative-box'>{st.session_state.narratives[mcode]}</div>",
            unsafe_allow_html=True,
        )
    else:
        if not _api_key_available():
            st.info(
                "Set `ANTHROPIC_API_KEY` to generate AI-powered market narratives.",
                icon="🔑",
            )
        else:
            if st.button(
                f"✍️ Generate Narrative for {selected['market_name']}",
                type="primary",
            ):
                with st.spinner("Claude is writing the market narrative…"):
                    narrative_tokens = []
                    narrative_placeholder = st.empty()
                    try:
                        for token in st.session_state.orchestrator.stream_narrative(
                            selected, weights
                        ):
                            narrative_tokens.append(token)
                            narrative_placeholder.markdown(
                                "".join(narrative_tokens)
                            )
                        st.session_state.narratives[mcode] = "".join(narrative_tokens)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Narrative generation failed: {exc}")

# ============================================================
# TAB 4 — PORTFOLIO SUMMARY
# ============================================================

with tab_portfolio:
    st.markdown(
        "<div class='section-title'>Portfolio-Level Executive Summary</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.portfolio_summary:
        st.markdown(
            f"<div class='narrative-box'>{st.session_state.portfolio_summary}</div>",
            unsafe_allow_html=True,
        )
        if st.button("🔄 Regenerate Summary"):
            st.session_state.portfolio_summary = None
            st.rerun()
    else:
        if not _api_key_available():
            st.info(
                "Set `ANTHROPIC_API_KEY` to generate an AI-powered portfolio summary.",
                icon="🔑",
            )
        else:
            st.markdown(
                "Generate a McKinsey-style executive summary across the full APAC market universe."
            )
            if st.button("📄 Generate Portfolio Summary", type="primary"):
                with st.spinner("Claude is writing the portfolio summary…"):
                    tokens = []
                    placeholder = st.empty()
                    try:
                        for token in st.session_state.orchestrator.stream_portfolio_summary(
                            active_scores, weights
                        ):
                            tokens.append(token)
                            placeholder.markdown("".join(tokens))
                        st.session_state.portfolio_summary = "".join(tokens)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Portfolio summary failed: {exc}")

    # Always show the score summary table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>Full Rankings — Export View</div>",
        unsafe_allow_html=True,
    )

    export_df = _scores_to_df(active_scores)
    st.dataframe(
        export_df.style.format(
            {
                "Composite": "{:.1f}",
                "Demand":    "{:.1f}",
                "Supply":    "{:.1f}",
                "Macro":     "{:.1f}",
            }
        ).background_gradient(
            subset=["Composite", "Demand", "Supply", "Macro"],
            cmap="RdYlGn",
            vmin=40,
            vmax=90,
        ),
        height=550,
        use_container_width=True,
    )

    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name=f"apac_dc_prioritisation_{datetime.now().strftime('%Y%m')}.csv",
        mime="text/csv",
    )

# ============================================================
# TAB 5 — METHODOLOGY
# ============================================================

with tab_method:
    st.markdown(
        "<div class='section-title'>Methodology & Scoring Framework</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
## Multi-Agent Architecture

This tool uses a team of four specialised Claude AI agents coordinated by an Orchestrator:

| Agent | Role | Claude Tools |
|-------|------|--------------|
| **Demand Intelligence Agent** | Analyses hyperscaler activity, AI workload growth, cloud adoption, enterprise demand, and DC absorption | `get_demand_data`, `submit_demand_analysis` |
| **Competitive Supply Agent** | Evaluates vacancy rates, pipeline risk, pricing power, operator concentration, and entry windows | `get_supply_data`, `submit_supply_analysis` |
| **Macro & Regulatory Agent** | Assesses power availability, regulatory simplicity, data sovereignty, political stability, and incentives | `get_macro_data`, `submit_macro_analysis` |
| **Narrative Agent** | Synthesises specialist analyses into executive market narratives (streaming) | Streaming text |

The **Orchestrator** runs the three specialist agents in parallel (per market) and applies
user-defined dimension weights to produce composite scores and rankings.

---

## Scoring Dimensions

### 1. Demand Signals (adjustable weight, default 40%)

| Sub-metric | Weight | Normalisation |
|------------|--------|---------------|
| Hyperscaler activity (1–10) | 30% | ÷ 10 |
| AI workload index (1–10) | 25% | ÷ 10 |
| Cloud adoption growth (% YoY) | 20% | Cap at 60% |
| Enterprise demand index (1–10) | 15% | ÷ 10 |
| DC absorption last 12m (MW) | 10% | Cap at 400MW |

### 2. Competitive Supply Opportunity (adjustable weight, default 30%)

Measures supply *attractiveness for a new entrant* (higher = more opportunistic):

| Sub-metric | Weight | Logic |
|------------|--------|-------|
| Vacancy tightness | 35% | Inverted (low vacancy = high score) |
| Pipeline pressure | 25% | Adjusted for pre-commitment rate |
| Pricing attractiveness ($/kW) | 25% | Normalised to $350/kW |
| Market depth (total MW) | 15% | Scale proxy |

### 3. Macro & Regulatory Environment (adjustable weight, default 30%)

| Sub-metric | Weight |
|------------|--------|
| Power availability | 20% |
| Regulatory simplicity | 18% |
| Data sovereignty score | 17% |
| Political stability | 15% |
| FX stability | 10% |
| Land availability | 10% |
| Skilled workforce | 5% |
| Incentives | 5% |

---

## Priority Bands

| Band | Composite Score | Action |
|------|----------------|--------|
| 🟢 **Tier 1 Priority** | ≥ 75 | Active pursuit — build or acquire |
| 🔵 **Tier 2 Priority** | 62–74 | Develop business case — 12-month window |
| 🟡 **Monitor** | 50–61 | Track signals — revisit quarterly |
| 🔴 **Deprioritise** | < 50 | Not recommended without strategic rationale |

---

## Data Sources

- **Structured data**: Internal market intelligence (March 2026)
- **Demand metrics**: JLL, CBRE, DC Byte, Synergy Research (proxied)
- **Supply metrics**: Operator filings, planning portals, industry databases
- **Macro metrics**: World Bank, EIU, GSMA, national grid operators

---

## Model

All AI analysis uses **Claude Opus 4.6** with adaptive thinking enabled for
nuanced multi-step reasoning across complex market dynamics.
"""
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
<hr style="margin-top:2rem;border-color:#D3D3D3">
<div style="text-align:center;color:#4A4F55;font-size:0.78rem;padding:.5rem 0 1rem">
  McKinsey & Company — Infrastructure Practice — APAC DC Strategy Team<br>
  <em>Confidential and Proprietary — For internal use only</em>
</div>
""",
    unsafe_allow_html=True,
)
