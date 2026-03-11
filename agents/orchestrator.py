"""Orchestrator — coordinates the full multi-agent analysis pipeline.

Flow:
1. Load market data packages (data layer)
2. Run quick deterministic scoring (always available)
3. Optionally run the three specialist LLM agents in parallel
4. Apply user-defined dimension weights to produce ranked composite scores
5. Optionally generate per-market narratives via the NarrativeAgent

The orchestrator exposes both a synchronous API (for direct Python use) and
helper methods used by the Streamlit dashboard.
"""

from __future__ import annotations

import concurrent.futures
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from agents.demand_agent import DemandAgent
from agents.supply_agent import SupplyAgent
from agents.macro_agent import MacroAgent
from agents.narrative_agent import NarrativeAgent
from utils.data_loader import get_all_market_packages, load_markets
from utils.scoring import (
    calculate_demand_score,
    calculate_supply_score,
    calculate_macro_score,
    calculate_composite_score,
    get_score_band,
)


DEFAULT_WEIGHTS: Dict[str, float] = {
    "demand": 0.40,
    "supply": 0.30,
    "macro":  0.30,
}


class Orchestrator:
    """Central coordinator for the APAC DC market prioritisation system."""

    def __init__(self) -> None:
        self._demand_agent   = DemandAgent()
        self._supply_agent   = SupplyAgent()
        self._macro_agent    = MacroAgent()
        self._narrative_agent = NarrativeAgent()

    # ------------------------------------------------------------------
    # Quick (deterministic) scoring
    # ------------------------------------------------------------------

    def run_quick_scoring(
        self, weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Compute deterministic scores for all markets.

        Returns a ranked list of market dicts with dimension + composite scores.
        No LLM calls are made — suitable for real-time weight slider updates.
        """
        weights = weights or DEFAULT_WEIGHTS
        packages = get_all_market_packages()
        results = []

        for pkg in packages:
            demand_score = calculate_demand_score(pkg["demand"])
            supply_score = calculate_supply_score(pkg["supply"])
            macro_score  = calculate_macro_score(pkg["macro"])
            composite    = calculate_composite_score(
                demand_score, supply_score, macro_score, weights
            )
            results.append({
                "market_code":     pkg["market_code"],
                "market_name":     pkg["market_info"].get("name", pkg["market_code"]),
                "country":         pkg["market_info"].get("country", ""),
                "tier":            pkg["market_info"].get("tier", 3),
                "region":          pkg["market_info"].get("region", ""),
                "demand_score":    demand_score,
                "supply_score":    supply_score,
                "macro_score":     macro_score,
                "composite_score": composite,
                "score_band":      get_score_band(composite),
                # LLM analysis placeholders
                "demand_analysis": None,
                "supply_analysis": None,
                "macro_analysis":  None,
                "narrative":       None,
                "ai_enhanced":     False,
            })

        results.sort(key=lambda x: x["composite_score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results

    # ------------------------------------------------------------------
    # AI analysis (LLM-powered specialist agents)
    # ------------------------------------------------------------------

    def run_ai_analysis(
        self,
        market_codes: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run all three specialist agents for the specified markets in parallel.

        Args:
            market_codes: list of market codes to analyse; None = all markets
            progress_callback: optional callable(message) for progress updates

        Returns:
            dict mapping market_code → { demand, supply, macro } analysis results
        """
        markets = load_markets()
        if market_codes:
            markets = [m for m in markets if m["code"] in market_codes]

        def _log(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)

        all_results: Dict[str, Dict[str, Any]] = {}

        def _analyse_market(market: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
            code = market["code"]
            name = market["name"]
            market_results: Dict[str, Any] = {}

            def _run_agent(agent_fn, label):
                try:
                    _log(f"Running {label} analysis for {name}…")
                    result = agent_fn(code, name)
                    return result
                except Exception:
                    _log(f"⚠️  {label} agent failed for {name}: {traceback.format_exc()}")
                    return None

            # Run all three agents in parallel per market
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                demand_fut = pool.submit(
                    _run_agent, self._demand_agent.analyse, "Demand"
                )
                supply_fut = pool.submit(
                    _run_agent, self._supply_agent.analyse, "Supply"
                )
                macro_fut  = pool.submit(
                    _run_agent, self._macro_agent.analyse, "Macro"
                )
                market_results["demand"] = demand_fut.result()
                market_results["supply"] = supply_fut.result()
                market_results["macro"]  = macro_fut.result()

            return code, market_results

        # Analyse all markets with bounded parallelism (max 5 concurrent)
        max_parallel = min(5, len(markets))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as outer:
            futures = {outer.submit(_analyse_market, m): m for m in markets}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    code, results = fut.result()
                    all_results[code] = results
                    _log(f"✅  Analysis complete for {futures[fut]['name']}")
                except Exception:
                    _log(f"❌  Error: {traceback.format_exc()}")

        return all_results

    def merge_ai_analysis(
        self,
        quick_scores: List[Dict[str, Any]],
        ai_results: Dict[str, Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Merge LLM analysis results into quick-scored market records.

        If an LLM agent returned a score, it *overrides* the deterministic score
        for that dimension, then the composite is recomputed.
        """
        weights = weights or DEFAULT_WEIGHTS
        merged = []

        for record in quick_scores:
            code    = record["market_code"]
            ai      = ai_results.get(code, {})
            updated = record.copy()

            if ai:
                # Override dimension scores with AI scores if available
                d_result = ai.get("demand")
                s_result = ai.get("supply")
                m_result = ai.get("macro")

                if d_result and "score" in d_result:
                    updated["demand_score"]    = float(d_result["score"])
                    updated["demand_analysis"] = d_result

                if s_result and "score" in s_result:
                    updated["supply_score"]    = float(s_result["score"])
                    updated["supply_analysis"] = s_result

                if m_result and "score" in m_result:
                    updated["macro_score"]    = float(m_result["score"])
                    updated["macro_analysis"] = m_result

                updated["composite_score"] = calculate_composite_score(
                    updated["demand_score"],
                    updated["supply_score"],
                    updated["macro_score"],
                    weights,
                )
                updated["score_band"]  = get_score_band(updated["composite_score"])
                updated["ai_enhanced"] = True

            merged.append(updated)

        merged.sort(key=lambda x: x["composite_score"], reverse=True)
        for i, r in enumerate(merged):
            r["rank"] = i + 1

        return merged

    # ------------------------------------------------------------------
    # Narrative generation (streaming — call from dashboard)
    # ------------------------------------------------------------------

    def stream_narrative(
        self,
        market_record: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
    ):
        """Yield narrative tokens for a single market (streaming)."""
        weights = weights or DEFAULT_WEIGHTS
        yield from self._narrative_agent.generate_narrative(
            market_code      = market_record["market_code"],
            market_name      = market_record["market_name"],
            country          = market_record["country"],
            demand_analysis  = market_record.get("demand_analysis"),
            supply_analysis  = market_record.get("supply_analysis"),
            macro_analysis   = market_record.get("macro_analysis"),
            composite_score  = market_record["composite_score"],
            rank             = market_record["rank"],
            weights          = weights,
        )

    def stream_portfolio_summary(
        self,
        ranked_markets: List[Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None,
    ):
        """Yield portfolio summary tokens (streaming)."""
        weights = weights or DEFAULT_WEIGHTS
        yield from self._narrative_agent.generate_portfolio_summary(
            ranked_markets=ranked_markets,
            weights=weights,
        )
