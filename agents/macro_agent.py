"""Macro & Regulatory Agent.

Assesses the macro-economic and regulatory environment for a given APAC DC
market.  Returns a score (0–100) reflecting the quality of the enabling
environment for data centre development.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.base_agent import BaseAgent
from utils.data_loader import get_market_package


SYSTEM_PROMPT = """You are a Senior Public Policy & Regulatory Affairs Analyst at McKinsey & Company.
You specialise in APAC digital infrastructure investment environments with expertise in:
- Power infrastructure: grid reliability, renewable energy commitments, utility tariffs
- Land & permitting: zoning frameworks, environmental approvals, development timelines
- Data sovereignty & localisation: PDPA, DPDPA, PIPL-adjacent regulations, cross-border rules
- Government incentives: fiscal regimes, enterprise zone benefits, infrastructure grants
- Political & FX stability: sovereign risk, currency hedging implications

Your role in this analysis is to assess the MACRO & REGULATORY ENVIRONMENT ONLY.
A high score means the market provides an excellent enabling environment for long-duration
infrastructure capital deployment.

Scoring calibration (APAC peer set):
  85-100 = Tier-1 enabling environment (SG, AUS standard)
  70-84  = Strong with manageable constraints
  55-69  = Moderate: notable friction but deployable with care
  40-54  = Elevated risk: material regulatory or infrastructure constraints
  <40    = High risk: significant barriers requiring local partnership or rethink

Always call the `submit_macro_analysis` tool to output your structured assessment."""


class MacroAgent(BaseAgent):

    OUTPUT_TOOL = "submit_macro_analysis"

    TOOLS = [
        {
            "schema": {
                "name": "get_macro_data",
                "description": "Retrieve macro and regulatory metrics for a market.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "market_code": {
                            "type": "string",
                            "description": "3-letter market code, e.g. SG, MUM, HCM"
                        }
                    },
                    "required": ["market_code"],
                },
            },
            "fn": lambda market_code: get_market_package(market_code)["macro"],
        },
        {
            "schema": {
                "name": "submit_macro_analysis",
                "description": (
                    "Submit the final macro & regulatory analysis for this market. "
                    "Call this once you have reviewed all relevant data."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "number",
                            "description": "Macro & regulatory environment score 0–100"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "3–4 sentence qualitative rationale"
                        },
                        "key_enablers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Top 3–5 macro/regulatory enablers"
                        },
                        "risk_flags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "1–3 macro/regulatory risks"
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "power_outlook": {
                            "type": "string",
                            "enum": ["improving", "stable", "deteriorating"],
                            "description": "12–18 month power infrastructure outlook"
                        },
                        "regulatory_trend": {
                            "type": "string",
                            "enum": ["liberalising", "stable", "tightening"],
                            "description": "Direction of regulatory travel"
                        },
                    },
                    "required": [
                        "score", "rationale", "key_enablers",
                        "risk_flags", "confidence", "power_outlook", "regulatory_trend"
                    ],
                },
            },
            "fn": None,
        },
    ]

    def analyse(self, market_code: str, market_name: str) -> Optional[Dict[str, Any]]:
        """Run macro & regulatory analysis for a single market."""
        user_message = f"""Analyse the macro and regulatory environment for the {market_name} ({market_code}) data centre market.

Step 1: Call `get_macro_data` with market_code="{market_code}" to retrieve the data.
Step 2: Assess: power availability and reliability, land access, regulatory simplicity,
        data sovereignty laws, political stability, FX stability, government incentives,
        skilled workforce availability, permitting timelines, and renewable energy %.
Step 3: Consider implications for a DC developer or investor:
        - Can power be procured reliably at scale for 10+ years?
        - Are permitting timelines tolerable for IRR targets?
        - Does the regulatory framework protect investor rights?
        - Are there data localisation requirements that attract or restrict tenants?
Step 4: Call `submit_macro_analysis` with your structured assessment.

Key regional context:
- Singapore and Australia set the benchmark for enabling environment in APAC
- India improving post-DPDPA 2023 with PLI scheme incentives for digital infra
- Philippines and Vietnam have elevated power reliability risks
- HK faces elevated data sovereignty scrutiny post-NSL"""

        result = self._run_agentic_loop(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            tools=self.TOOLS,
            output_tool_name=self.OUTPUT_TOOL,
        )
        if result:
            result["market_code"] = market_code
            result["agent"] = "macro"
        return result
