"""AI use-case scoring engine.

Logic:
  • No hard filter on distance — distance is a scored dimension (soft constraint).
  • Sites are scored on: speed to market, power, capacity scale, distance from AZ,
    land, renewable energy, and power cost.
  • Hyperscaler-specific profiles apply multipliers for Google/Oracle vs AWS/Azure vs Meta.

Scoring weights (default):
  Speed to market (RFS months)          30 %
  Available uncommitted power (MW)      25 %
  Distance to cloud AZ (soft)           20 %
  Max single-site capacity (MW)         10 %
  Land availability (ha)                 5 %
  Renewable energy access (%)            5 %
  Power cost ($/MWh)                     5 %

Hyperscaler profiles:
  Google / Oracle  : Location agnostic; 250 MW+ and <30 months RFS critical
  AWS / Azure      : Cloud-stretch preferred (≤50 km); 100-200 MW threshold
  Meta / TikTok    : Power and land first; campus own-build; 200 MW+
"""

from __future__ import annotations
from typing import Any, Dict, List


# Distance bands (km from nearest cloud AZ)
_DIST_SCORE_BANDS = [
    (0,   50,  100.0, 0.0),    # 0–50 km: full score
    (50,  100, 100.0, -40.0),  # 50–100 km: score falls 40 pts
    (100, 150, 60.0,  -30.0),  # 100–150 km: score falls another 30 pts
    (150, 999, 30.0,  -20.0),  # >150 km: steep decay
]


def _distance_to_score(dist_km: float) -> float:
    for lo, hi, start, decay in _DIST_SCORE_BANDS:
        if lo <= dist_km < hi:
            fraction = (dist_km - lo) / (hi - lo)
            return max(0.0, start + decay * fraction)
    return 0.0


def calculate_ai_score(data: Dict[str, Any]) -> float:
    """Return 0–100 AI opportunity score."""

    # Speed to market (lower RFS = higher score; 30 months is the benchmark)
    rfs = data["shortest_rfs_months"]
    if rfs <= 18:
        rfs_score = 100.0
    elif rfs <= 30:
        rfs_score = 100.0 - ((rfs - 18) / 12) * 30   # 70–100
    elif rfs <= 48:
        rfs_score = 70.0  - ((rfs - 30) / 18) * 40   # 30–70
    else:
        rfs_score = max(0.0, 30.0 - ((rfs - 48) / 24) * 30)

    # Available uncommitted power (normalised to 700 MW — Moorabool scale)
    power_score = min(data["available_uncommitted_power_mw"] / 700, 1) * 100

    # Distance to cloud AZ (soft scoring)
    dist_score = _distance_to_score(data["distance_to_cloud_az_km"])

    # Max single-site capacity (Google/Oracle threshold: 250 MW)
    cap_score = min(data["max_single_site_mw"] / 800, 1) * 100

    # Land availability (normalised to 320 ha — Moorabool scale)
    land_score = min(data["land_available_ha"] / 320, 1) * 100

    # Renewable energy (%)
    renew_score = min(data["renewable_energy_accessible_pct"], 100)

    # Power cost — lower is better; normalised to $150/MWh
    cost_score = max(0.0, (1 - data["power_cost_usd_mwh"] / 150) * 100)

    score = (
        rfs_score   * 0.30
        + power_score * 0.25
        + dist_score  * 0.20
        + cap_score   * 0.10
        + land_score  * 0.05
        + renew_score * 0.005   # renew_score already 0-100, weight 5% → 0.05 → scale
        + cost_score  * 0.05
    )
    # Fix: renew_score already 0-100, multiply weight as 0.05
    score = (
        rfs_score   * 0.30
        + power_score * 0.25
        + dist_score  * 0.20
        + cap_score   * 0.10
        + land_score  * 0.05
        + (renew_score / 100) * 100 * 0.05
        + cost_score  * 0.05
    )
    return round(score, 1)


def calculate_hyperscaler_ai_fit(
    data: Dict[str, Any],
    hyperscaler: str,
) -> float:
    """Return a 0–100 AI fit score for a specific hyperscaler.

    Each hyperscaler has a distinct site-selection profile:

    Google / Oracle:
      - Location agnostic (within 100 km soft limit)
      - Must have 250 MW+ single-site capacity
      - RFS < 30 months is critical
      - Strong weight on total capacity and land

    AWS / Azure:
      - Prefer cloud-stretch (≤50 km from own AZ)
      - 100–200 MW per site preferred
      - More sensitive to distance than Google/Oracle

    Meta / TikTok:
      - Own-build campus focus
      - 200 MW+ preferred
      - Power cost and renewable energy are primary filters
      - Distance is secondary
    """
    h = hyperscaler.upper()

    d   = data
    rfs = d["shortest_rfs_months"]
    mw  = d["available_uncommitted_power_mw"]
    cap = d["max_single_site_mw"]
    dist= d["distance_to_cloud_az_km"]
    ren = d["renewable_energy_accessible_pct"]
    cost= d["power_cost_usd_mwh"]
    land= d["land_available_ha"]

    interest_field = f"{h.lower()}_ai_interest"
    has_interest   = bool(d.get(interest_field, 0))
    interest_bonus = 10.0 if has_interest else 0.0

    if h in ("GOOGLE", "ORACLE"):
        # Hard-ish capacity threshold (must have 250 MW+ single site)
        cap_penalty = -20 if cap < 250 else 0

        # RFS: 30 months is the benchmark
        rfs_score = 100 if rfs <= 30 else max(0, 100 - (rfs - 30) * 5)

        # Distance: within 100 km preferred, 100–150 km acceptable
        dist_score = _distance_to_score(dist) if dist <= 150 else _distance_to_score(dist) * 0.5

        # Power: need 250 MW+
        pwr_score = min(mw / 500, 1) * 100

        score = (
            rfs_score  * 0.35
            + pwr_score  * 0.25
            + dist_score * 0.15
            + cap_penalty
            + (min(cap / 800, 1) * 100) * 0.15
            + (ren / 100) * 100 * 0.05
            + max(0, (1 - cost / 150) * 100) * 0.05
            + interest_bonus
        )

    elif h in ("AWS", "AZURE"):
        # Cloud-stretch focus: ≤50 km strongly preferred
        if dist <= 50:
            dist_score = 100.0
        elif dist <= 80:
            dist_score = 100 - ((dist - 50) / 30) * 50
        else:
            dist_score = max(0, 50 - ((dist - 80) / 70) * 50)

        # 100–200 MW per site sweet spot
        pwr_score = min(mw / 250, 1) * 100 if mw >= 100 else (mw / 100) * 60

        rfs_score = 100 if rfs <= 24 else max(0, 100 - (rfs - 24) * 3)

        score = (
            rfs_score  * 0.25
            + pwr_score  * 0.25
            + dist_score * 0.35
            + (min(cap / 400, 1) * 100) * 0.10
            + (ren / 100) * 100 * 0.05
            + interest_bonus
        )

    else:  # META / TIKTOK — campus, own-build
        # Power cost is primary; renewable energy matters
        pwr_score  = min(mw / 500, 1) * 100
        land_score = min(land / 300, 1) * 100
        cost_score = max(0, (1 - cost / 120) * 100)
        rfs_score  = 100 if rfs <= 30 else max(0, 100 - (rfs - 30) * 4)
        dist_score = _distance_to_score(min(dist, 150))

        score = (
            pwr_score  * 0.30
            + land_score * 0.20
            + cost_score * 0.20
            + rfs_score  * 0.15
            + (ren / 100) * 100 * 0.10
            + dist_score * 0.05
            + interest_bonus
        )

    return round(min(max(score, 0), 100), 1)


def score_all_ai_submarkets(submarket_packages: list) -> list:
    """Score all AI-eligible submarkets and return sorted list."""
    results = []
    for pkg in submarket_packages:
        ai   = pkg["ai"]
        info = pkg["submarket_info"]

        if "ai" not in info.get("use_cases", []):
            continue

        base_score = calculate_ai_score(ai)
        hs_scores  = {
            h: calculate_hyperscaler_ai_fit(ai, h)
            for h in ("AWS", "AZURE", "GOOGLE", "ORACLE", "META")
        }

        # Average across interested hyperscalers
        interested = [
            hs_scores[h]
            for h, field in [
                ("AWS",    "aws_ai_interest"),
                ("AZURE",  "azure_ai_interest"),
                ("GOOGLE", "google_ai_interest"),
                ("ORACLE", "oracle_ai_interest"),
                ("META",   "meta_ai_interest"),
            ]
            if ai.get(field, 0)
        ]
        avg_hs_score = round(sum(interested) / len(interested), 1) if interested else 0.0

        # Blend: 60% base model, 40% hyperscaler preference signal
        blended_score = round(base_score * 0.60 + avg_hs_score * 0.40, 1)

        top_hs = max(hs_scores, key=lambda k: hs_scores[k])

        results.append({
            "submarket_code":            pkg["submarket_code"],
            "submarket_name":            info.get("name", pkg["submarket_code"]),
            "city":                      info.get("city", ""),
            "city_code":                 info.get("city_code", ""),
            "country":                   info.get("country", ""),
            "region":                    info.get("region", ""),
            "submarket_type":            info.get("submarket_type", ""),
            "distance_to_az_km":         ai["distance_to_cloud_az_km"],
            "available_power_mw":        ai["available_uncommitted_power_mw"],
            "shortest_rfs_months":       ai["shortest_rfs_months"],
            "max_single_site_mw":        ai["max_single_site_mw"],
            "land_available_ha":         ai["land_available_ha"],
            "renewable_pct":             ai["renewable_energy_accessible_pct"],
            "power_cost_usd_mwh":        ai["power_cost_usd_mwh"],
            "campus_ready":              bool(ai.get("campus_ready", 0)),
            "ai_score":                  base_score,
            "avg_hyperscaler_score":     avg_hs_score,
            "blended_ai_score":          blended_score,
            "aws_ai_fit":                hs_scores["AWS"],
            "azure_ai_fit":              hs_scores["AZURE"],
            "google_ai_fit":             hs_scores["GOOGLE"],
            "oracle_ai_fit":             hs_scores["ORACLE"],
            "meta_ai_fit":               hs_scores["META"],
            "top_hyperscaler":           top_hs,
            "notes":                     info.get("notes", ""),
            "ai_enhanced":               False,
            "narrative":                 None,
        })

    results.sort(key=lambda x: x["blended_ai_score"], reverse=True)
    for i, r in enumerate(results):
        r["ai_rank"] = i + 1
    return results


def get_ai_band(score: float) -> str:
    if score >= 72: return "Tier 1 AI"
    if score >= 55: return "Tier 2 AI"
    if score >= 40: return "Watch"
    return "Marginal"


HYPERSCALER_PROFILES = {
    "GOOGLE": {
        "label": "Google",
        "color": "#4285F4",
        "description": "Location agnostic · 250 MW+ · <30 months RFS · within 100 km",
    },
    "ORACLE": {
        "label": "Oracle",
        "color": "#F80000",
        "description": "Location agnostic · 250 MW+ · <30 months RFS · within 100 km · fastest procurement",
    },
    "AWS": {
        "label": "AWS",
        "color": "#FF9900",
        "description": "Cloud stretch ≤50 km · 100–200 MW · speed-sensitive",
    },
    "AZURE": {
        "label": "Azure",
        "color": "#0078D4",
        "description": "Cloud stretch ≤50 km · 100–200 MW · carrier diversity critical",
    },
    "META": {
        "label": "Meta",
        "color": "#1877F2",
        "description": "Campus own-build · 200 MW+ · power cost and renewable primary",
    },
}
