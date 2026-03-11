"""Demand Intelligence Agent.

Analyses demand signals for a given APAC market using Claude, with access to
structured CSV data.  Returns a score (0–100), rationale, key signals, and
risk flags.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from agents.base_agent import BaseAgent
from utils.data_loader import get_market_package


SYSTEM_PROMPT = """You are a Senior Data Centre Demand Analyst at McKinsey & Company.
You specialise in APAC digital infrastructure markets and have deep expertise in:
- Hyperscaler expansion patterns (AWS, Azure, GCP, Alibaba, Oracle, Tencent, Naver)
- Enterprise cloud adoption trajectories by sector and geography
- AI/ML workload infrastructure requirements and growth drivers
- DC absorption data interpretation and forward-looking demand signals

Your role in this analysis is to assess DEMAND SIGNALS ONLY for a given market.
Be rigorous, data-driven, and commercially realistic. Avoid hype.

When scoring, calibrate to the APAC peer set:
  90-100 = Exceptional demand (Singapore / Tokyo tier)
  75-89  = Strong demand
  60-74  = Moderate-to-strong demand
  45-59  = Emerging demand
  <45    = Early-stage / nascent demand

Always call the `submit_demand_analysis` tool to output your structured assessment."""


class DemandAgent(BaseAgent):

    OUTPUT_TOOL = "submit_demand_analysis"

    TOOLS = [
        {
            "schema": {
                "name": "get_demand_data",
                "description": "Retrieve structured demand metrics for a market.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "market_code": {
                            "type": "string",
                            "description": "3-letter market code, e.g. SG, KUL, MUM"
                        }
                    },
                    "required": ["market_code"],
                },
            },
            "fn": lambda market_code: get_market_package(market_code)["demand"],
        },
        {
            "schema": {
                "name": "submit_demand_analysis",
                "description": (
                    "Submit the final demand analysis for this market. "
                    "Call this once you have reviewed all relevant data."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "number",
                            "description": "Demand score 0–100 for this market"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "3–4 sentence qualitative rationale for the score"
                        },
                        "key_signals": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Top 3–5 demand signals driving the score"
                        },
                        "risk_flags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "1–3 key demand risks or headwinds"
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Analyst confidence in the score"
                        },
                        "demand_outlook": {
                            "type": "string",
                            "enum": ["accelerating", "stable", "decelerating"],
                            "description": "12–18 month demand outlook"
                        },
                    },
                    "required": [
                        "score", "rationale", "key_signals",
                        "risk_flags", "confidence", "demand_outlook"
                    ],
                },
            },
            "fn": None,  # Output capture — no side effect
        },
    ]

    def analyse(self, market_code: str, market_name: str) -> Optional[Dict[str, Any]]:
        """Run demand analysis for a single market.

        Returns a dict with score, rationale, signals, etc., or None on failure.
        """
        user_message = f"""Analyse the demand signals for the {market_name} ({market_code}) data centre market.

Step 1: Call `get_demand_data` with market_code="{market_code}" to retrieve the data.
Step 2: Review the metrics carefully — consider hyperscaler activity, AI workload index,
        cloud adoption growth, enterprise demand, and recent DC absorption.
Step 3: Cross-reference the metrics against the APAC peer set context below.
Step 4: Call `submit_demand_analysis` with your structured assessment.

APAC peer benchmarks for calibration:
- SG, TYO, SYD are Tier 1 markets with the strongest demand bases
- MUM, DEL, KUL, JKT represent the highest growth trajectories
- HCM, BKK, AKL are earlier-stage markets

Be precise and commercially grounded in your analysis."""

        result = self._run_agentic_loop(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            tools=self.TOOLS,
            output_tool_name=self.OUTPUT_TOOL,
        )
        if result:
            result["market_code"] = market_code
            result["agent"] = "demand"
        return result
