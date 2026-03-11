"""Deterministic scoring engine for quick (non-LLM) market scoring.

Each dimension produces a score 0–100. Higher is always better from a
site-selection perspective (supply score measures *opportunity*, not volume).
"""

from __future__ import annotations
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Demand Score
# ---------------------------------------------------------------------------

def calculate_demand_score(data: Dict[str, Any]) -> float:
    """Score demand signals 0–100.

    Weights:
        Hyperscaler activity   30 %
        AI workload index      25 %
        Cloud growth (% YoY)   20 %
        Enterprise demand      15 %
        DC absorption (MW)     10 %
    """
    hyperscaler   = (data["hyperscaler_activity"] / 10) * 100
    ai_workload   = (data["ai_workload_index"] / 10) * 100
    cloud_growth  = min(data["cloud_adoption_growth_pct"] / 60, 1) * 100
    enterprise    = (data["enterprise_demand_index"] / 10) * 100
    absorption    = min(data["dc_absorption_mw"] / 400, 1) * 100

    score = (
        hyperscaler  * 0.30
        + ai_workload  * 0.25
        + cloud_growth * 0.20
        + enterprise   * 0.15
        + absorption   * 0.10
    )
    return round(score, 1)


# ---------------------------------------------------------------------------
# Supply Opportunity Score
# ---------------------------------------------------------------------------

def calculate_supply_score(data: Dict[str, Any]) -> float:
    """Score supply *opportunity* 0–100.

    A market scores higher when it is under-supplied, tightly occupied,
    commands premium pricing, and has limited competitive pipeline.

    Weights:
        Vacancy tightness          35 %
        Pipeline pressure          25 %
        Pricing attractiveness     25 %
        Market depth (scale)       15 %
    """
    # Low vacancy → tight market → high score
    vacancy_score = max(0.0, (1 - data["vacancy_rate_pct"] / 15)) * 100

    # Low pipeline-to-stock ratio → less competitive threat
    pipeline_ratio  = data["pipeline_mw"] / max(data["total_commissioned_mw"], 1)
    # Also discount if pipeline is largely pre-committed (less additive supply)
    pre_commit_adj  = data.get("pre_committed_pipeline_pct", 50) / 100
    effective_ratio = pipeline_ratio * (1 - pre_commit_adj * 0.5)
    pipeline_score  = max(0.0, (1 - min(effective_ratio, 1.5) / 1.5)) * 100

    # Higher colo price → better yield potential (normalised to $350/kW)
    price_score = min(data["colo_pricing_usd_kw"] / 350, 1) * 100

    # Market scale: larger installed base → more demand anchor (norm to 1 800 MW)
    depth_score = min(data["total_commissioned_mw"] / 1800, 1) * 100

    score = (
        vacancy_score  * 0.35
        + pipeline_score * 0.25
        + price_score    * 0.25
        + depth_score    * 0.15
    )
    return round(score, 1)


# ---------------------------------------------------------------------------
# Macro & Regulatory Score
# ---------------------------------------------------------------------------

def calculate_macro_score(data: Dict[str, Any]) -> float:
    """Score macro & regulatory environment 0–100.

    Weights:
        Power availability      20 %
        Regulatory simplicity   18 %
        Data sovereignty        17 %
        Political stability     15 %
        FX stability            10 %
        Land availability       10 %
        Skilled workforce        5 %
        Incentives               5 %
    """
    to_pct = lambda key: (data[key] / 10) * 100

    score = (
        to_pct("power_availability_score")    * 0.20
        + to_pct("regulatory_simplicity_score") * 0.18
        + to_pct("data_sovereignty_score")      * 0.17
        + to_pct("political_stability_score")   * 0.15
        + to_pct("fx_stability_score")          * 0.10
        + to_pct("land_availability_score")     * 0.10
        + to_pct("skilled_workforce_score")     * 0.05
        + to_pct("incentives_score")            * 0.05
    )
    return round(score, 1)


# ---------------------------------------------------------------------------
# Composite + Ranking
# ---------------------------------------------------------------------------

def calculate_composite_score(
    demand_score: float,
    supply_score: float,
    macro_score: float,
    weights: Dict[str, float],
) -> float:
    """Weighted composite of the three dimension scores."""
    w_d = weights.get("demand", 0.40)
    w_s = weights.get("supply", 0.30)
    w_m = weights.get("macro",  0.30)
    total_w = w_d + w_s + w_m
    if total_w == 0:
        return 0.0
    return round(
        (demand_score * w_d + supply_score * w_s + macro_score * w_m) / total_w, 1
    )


def score_all_markets(
    market_packages: list,
    weights: Dict[str, float],
) -> list:
    """Compute quick scores for all markets and return ranked list."""
    results = []
    for pkg in market_packages:
        demand_score = calculate_demand_score(pkg["demand"])
        supply_score = calculate_supply_score(pkg["supply"])
        macro_score  = calculate_macro_score(pkg["macro"])
        composite    = calculate_composite_score(
            demand_score, supply_score, macro_score, weights
        )
        results.append({
            "market_code":    pkg["market_code"],
            "market_name":    pkg["market_info"].get("name", pkg["market_code"]),
            "country":        pkg["market_info"].get("country", ""),
            "tier":           pkg["market_info"].get("tier", 3),
            "region":         pkg["market_info"].get("region", ""),
            "demand_score":   demand_score,
            "supply_score":   supply_score,
            "macro_score":    macro_score,
            "composite_score": composite,
        })

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results


def get_score_band(score: float) -> str:
    """Return a priority band label for a composite score."""
    if score >= 75:
        return "Tier 1 Priority"
    elif score >= 62:
        return "Tier 2 Priority"
    elif score >= 50:
        return "Monitor"
    else:
        return "Deprioritise"
