"""Cloud use-case scoring engine.

Logic:
  • Hard filter: fibre_distance_to_az_km > 10 → score = None (excluded)
  • Score reflects attractiveness for a *cloud-first* tenant: AWS/Azure/Google/Oracle
    looking for sub-10km proximity to their existing AZ cluster for latency,
    redundancy, and connectivity density reasons.

Score dimensions:
  Proximity to AZ (hard-capped 10km)  25 %
  Carrier / fibre diversity            22 %
  Existing DC ecosystem density        20 %
  Available uncommitted power          18 %
  Colo pricing (yield proxy)           15 %
"""

from __future__ import annotations
from typing import Any, Dict, Optional


CLOUD_AZ_HARD_LIMIT_KM: float = 10.0


def calculate_cloud_score(data: Dict[str, Any]) -> Optional[float]:
    """Return 0–100 cloud score or None if the hard filter excludes the submarket."""
    dist = data["fibre_distance_to_az_km"]

    # Hard filter
    if dist > CLOUD_AZ_HARD_LIMIT_KM:
        return None

    # Proximity score: 0 km → 100, 10 km → 0
    proximity_score = (1 - dist / CLOUD_AZ_HARD_LIMIT_KM) * 100

    # Carrier diversity (1–5 diverse fibre routes)
    carrier_score = (data["carrier_diversity_score"] / 5) * 100

    # Existing DC operator count — more operators = more established market
    dc_eco_score = min(data["existing_dc_operators_count"] / 10, 1) * 100

    # Available uncommitted power — normalised to 200 MW (large campus)
    committed_adj = 1 - (data["power_committed_pct"] / 100) * 0.8
    power_score   = min(data["available_power_mw"] / 200, 1) * 100 * committed_adj

    # Colo pricing — higher = better developer yield (normalised to $380/kW = SG peak)
    if data["colo_pricing_usd_kw"] == 0:
        price_score = 0.0
    else:
        price_score = min(data["colo_pricing_usd_kw"] / 380, 1) * 100

    score = (
        proximity_score * 0.25
        + carrier_score   * 0.22
        + dc_eco_score    * 0.20
        + power_score     * 0.18
        + price_score     * 0.15
    )
    return round(score, 1)


def calculate_hyperscaler_cloud_fit(
    data: Dict[str, Any],
    hyperscaler: str,
) -> Optional[float]:
    """Return a cloud fit score for a specific hyperscaler.

    Applies hyperscaler-specific multipliers on top of the base cloud score.
    Returns None if the base cloud score is None (hard excluded).
    """
    base = calculate_cloud_score(data)
    if base is None:
        return None

    h = hyperscaler.upper()

    # Presence bonus: hyperscaler already active here = validates the location
    presence_field = f"{h.lower()}_present"
    already_present = bool(data.get(presence_field, 0))
    presence_bonus  = 8.0 if already_present else 0.0

    # Each hyperscaler weights proximity differently
    if h == "AWS":
        # AWS is strict on proximity — best within 5 km of own AZ
        dist     = data["fibre_distance_to_az_km"]
        dist_adj = -10 if dist > 6 else 0
        score    = base + presence_bonus + dist_adj
    elif h == "AZURE":
        # Azure tolerates up to 8 km; highly values carrier diversity
        carrier_bonus = (data["carrier_diversity_score"] - 3) * 3
        score         = base + presence_bonus + carrier_bonus
    elif h in ("GOOGLE", "ORACLE"):
        # Google/Oracle value established ecosystem more than raw proximity
        eco_bonus = (data["existing_dc_operators_count"] / 10) * 5
        score     = base + presence_bonus + eco_bonus
    else:
        score = base + presence_bonus

    return round(min(max(score, 0), 100), 1)


def score_all_cloud_submarkets(
    submarket_packages: list,
) -> list:
    """Score all cloud-eligible submarkets and return sorted list.

    Excludes submarkets that fail the hard 10 km filter.
    """
    results = []
    for pkg in submarket_packages:
        cd   = pkg["cloud"]
        info = pkg["submarket_info"]

        base_score = calculate_cloud_score(cd)
        if base_score is None:
            continue  # Hard filtered out

        hyperscaler_scores = {
            h: calculate_hyperscaler_cloud_fit(cd, h)
            for h in ("AWS", "AZURE", "GOOGLE", "ORACLE")
        }

        # Leading hyperscaler
        valid_hs  = {k: v for k, v in hyperscaler_scores.items() if v is not None}
        top_hs    = max(valid_hs, key=lambda k: valid_hs[k]) if valid_hs else "—"
        hs_count  = sum(1 for v in (cd.get("aws_present",0), cd.get("azure_present",0),
                                    cd.get("google_present",0), cd.get("oracle_present",0))
                        if v)

        results.append({
            "submarket_code":     pkg["submarket_code"],
            "submarket_name":     info.get("name", pkg["submarket_code"]),
            "city":               info.get("city", ""),
            "city_code":          info.get("city_code", ""),
            "country":            info.get("country", ""),
            "region":             info.get("region", ""),
            "submarket_type":     info.get("submarket_type", ""),
            "fibre_distance_km":  cd["fibre_distance_to_az_km"],
            "carrier_diversity":  cd["carrier_diversity_score"],
            "available_power_mw": cd["available_power_mw"],
            "power_committed_pct":cd["power_committed_pct"],
            "colo_pricing_usd_kw":cd["colo_pricing_usd_kw"],
            "hyperscaler_count":  hs_count,
            "cloud_score":        base_score,
            "aws_cloud_fit":      hyperscaler_scores.get("AWS"),
            "azure_cloud_fit":    hyperscaler_scores.get("AZURE"),
            "google_cloud_fit":   hyperscaler_scores.get("GOOGLE"),
            "oracle_cloud_fit":   hyperscaler_scores.get("ORACLE"),
            "top_hyperscaler":    top_hs,
            "notes":              info.get("notes", ""),
            "ai_enhanced":        False,
            "narrative":          None,
        })

    results.sort(key=lambda x: x["cloud_score"], reverse=True)
    for i, r in enumerate(results):
        r["cloud_rank"] = i + 1
    return results


def get_cloud_band(score: float) -> str:
    if score >= 72: return "Tier 1 Cloud"
    if score >= 55: return "Tier 2 Cloud"
    if score >= 40: return "Watch"
    return "Marginal"
