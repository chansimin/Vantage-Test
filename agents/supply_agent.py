"""Competitive Supply Agent.

Analyses the supply landscape and competitive dynamics for a given APAC DC
market using Claude.  Returns a supply *opportunity* score (0–100), where a
higher score means a more attractive supply environment for a new entrant.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.base_agent import BaseAgent
from utils.data_loader import get_market_package, get_announcements_for_market


SYSTEM_PROMPT = """You are a Senior Data Centre Market Intelligence Analyst at McKinsey & Company.
You specialise in APAC colocation and hyperscale supply-side analysis with expertise in:
- Operator competitive landscapes (Equinix, Digital Realty, STT, GDS, NTT, Keppel, etc.)
- Supply pipeline assessment: speculative vs pre-committed, time-to-market risk
- Vacancy rate dynamics and pricing power in constrained markets
- Barriers to entry: power procurement, land zoning, permitting timelines

Your role in this analysis is to assess COMPETITIVE SUPPLY DYNAMICS ONLY.
From a site-selection perspective, score the *opportunity* not the market size —
a high score means the market is attractive because supply is tight or early-stage.

Scoring calibration (APAC peer set):
  85-100 = Severely undersupplied; white-space opportunity (e.g. early KUL, HCM)
  70-84  = Undersupplied with first-mover advantage windows remaining
  55-69  = Balanced; competitive but selectable niches exist
  40-54  = Well-supplied; incremental opportunity only
  <40    = Over-supplied or high pipeline risk; avoid unless differentiated

You have access to real-world supply announcements via `get_supply_announcements`.
These represent actual construction starts, campus filings, land banks, and expansions
by operators — use them to validate or challenge the static supply metrics.

Always call the `submit_supply_analysis` tool to output your structured assessment."""


class SupplyAgent(BaseAgent):

    OUTPUT_TOOL = "submit_supply_analysis"

    TOOLS = [
        {
            "schema": {
                "name": "get_supply_data",
                "description": "Retrieve structured supply and competitive metrics for a market.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "market_code": {
                            "type": "string",
                            "description": "3-letter market code, e.g. SG, KUL, TYO"
                        }
                    },
                    "required": ["market_code"],
                },
            },
            "fn": lambda market_code: get_market_package(market_code)["supply"],
        },
        {
            "schema": {
                "name": "get_supply_announcements",
                "description": (
                    "Retrieve real-world supply-side announcements for a market "
                    "(construction starts, planning filings, land acquisitions, "
                    "campus expansions, operator entries). "
                    "Call this to ground your competitive analysis in actual pipeline activity."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "market_code": {
                            "type": "string",
                            "description": "3-letter market code, e.g. SYD, JHR, TYO"
                        }
                    },
                    "required": ["market_code"],
                },
            },
            "fn": lambda market_code: get_announcements_for_market(
                market_code, category="supply"
            ),
        },
        {
            "schema": {
                "name": "submit_supply_analysis",
                "description": (
                    "Submit the final supply opportunity analysis for this market. "
                    "Call this once you have reviewed all supply data."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "number",
                            "description": "Supply opportunity score 0–100"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "3–4 sentence qualitative rationale"
                        },
                        "key_signals": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Top 3–5 supply signals (e.g. vacancy rate, pipeline risk)"
                        },
                        "risk_flags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "1–3 supply-side risks (e.g. hyperscaler self-build, oversupply)"
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "market_tightness": {
                            "type": "string",
                            "enum": [
                                "severely_constrained",
                                "tight",
                                "balanced",
                                "oversupplied",
                            ],
                            "description": "Current supply tightness classification"
                        },
                        "entry_window": {
                            "type": "string",
                            "enum": ["open", "closing", "closed"],
                            "description": "Estimated first-mover/entry window status"
                        },
                    },
                    "required": [
                        "score", "rationale", "key_signals",
                        "risk_flags", "confidence", "market_tightness", "entry_window"
                    ],
                },
            },
            "fn": None,
        },
    ]

    def analyse(self, market_code: str, market_name: str) -> Optional[Dict[str, Any]]:
        """Run supply analysis for a single market."""
        user_message = f"""Analyse the competitive supply landscape for the {market_name} ({market_code}) data centre market.

Step 1: Call `get_supply_data` with market_code="{market_code}" to retrieve structured metrics.
Step 2: Call `get_supply_announcements` with market_code="{market_code}" to retrieve real-world
        supply pipeline activity (construction starts, land banks, planning approvals).
Step 3: Assess the combined picture:
        - Static metrics: vacancy, pipeline, pre-commitment, pricing, operator HHI
        - Announcements: who is entering, how much MW is being planned, how speculative vs committed?
        - Large announced MW additions change the competitive outlook even before completion.
Step 4: Judge the opportunity from a new-entrant or M&A acquirer perspective:
        - Is there a white-space gap? Is existing supply pre-committed?
        - Is the pipeline genuinely additive or largely hyperscaler captive?
        - Does pricing support attractive yields?
Step 5: Call `submit_supply_analysis` with your structured assessment.

Key context:
- SG and HK have the tightest supply but also high barriers to entry
- KUL, JKT, MUM have significant pipeline but nascent competitive operators
- HCM, BKK are truly early stage with minimal institutional supply"""

        result = self._run_agentic_loop(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            tools=self.TOOLS,
            output_tool_name=self.OUTPUT_TOOL,
        )
        if result:
            result["market_code"] = market_code
            result["agent"] = "supply"
        return result
