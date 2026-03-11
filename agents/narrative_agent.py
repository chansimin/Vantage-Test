"""Narrative Agent.

Synthesises the outputs of the three specialist agents into a concise
executive market narrative suitable for a McKinsey client deliverable.
Generates text via streaming so the dashboard can display it in real time.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Generator, Optional

from agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are a Principal-level McKinsey consultant specialising in APAC data centre strategy.
You write crisp, insightful, evidence-based market narratives for C-suite and investment committee audiences.

Your writing style:
- Direct and concise — no management jargon
- Lead with the most important insight
- Quantify wherever possible
- Identify the commercial implication, not just describe the data
- Use present tense for current state, future tense for outlook

Structure each market narrative as:
1. **Headline sentence**: One sentence encapsulating the market's priority status.
2. **Demand picture**: 2–3 sentences covering demand drivers and trajectory.
3. **Supply landscape**: 2–3 sentences covering competitive dynamics and opportunity.
4. **Macro & regulatory context**: 2–3 sentences on the enabling environment.
5. **Strategic recommendation**: 1–2 sentences with a clear action (invest, monitor, avoid, partner).

Total length: ~220–280 words. Use markdown headers (##, **bold**) for structure."""


class NarrativeAgent(BaseAgent):

    def generate_narrative(
        self,
        market_code: str,
        market_name: str,
        country: str,
        demand_analysis: Optional[Dict[str, Any]],
        supply_analysis: Optional[Dict[str, Any]],
        macro_analysis: Optional[Dict[str, Any]],
        composite_score: float,
        rank: int,
        weights: Dict[str, float],
    ) -> Generator[str, None, None]:
        """Stream a market narrative. Yields text tokens as they arrive."""

        demand_json = json.dumps(demand_analysis, indent=2) if demand_analysis else "Not available"
        supply_json = json.dumps(supply_analysis, indent=2) if supply_analysis else "Not available"
        macro_json  = json.dumps(macro_analysis,  indent=2) if macro_analysis  else "Not available"

        w_d = round(weights.get("demand", 0.4) * 100)
        w_s = round(weights.get("supply", 0.3) * 100)
        w_m = round(weights.get("macro",  0.3) * 100)

        user_message = f"""Write an executive market narrative for **{market_name}, {country}**.

Market ranking: #{rank} in APAC (composite score: {composite_score:.1f}/100)
Weight methodology: Demand {w_d}% | Supply {w_s}% | Macro {w_m}%

--- DEMAND ANALYSIS ---
{demand_json}

--- SUPPLY ANALYSIS ---
{supply_json}

--- MACRO & REGULATORY ANALYSIS ---
{macro_json}

Write the narrative following the prescribed structure. Be specific, cite the key
metrics from the analyses above, and conclude with a clear strategic recommendation.
Address the narrative to an investment committee reviewing APAC DC site selection."""

        yield from self._stream_text(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=1024,
        )

    def generate_portfolio_summary(
        self,
        ranked_markets: list,
        weights: Dict[str, float],
    ) -> Generator[str, None, None]:
        """Stream a 1-page portfolio-level summary of the top markets."""

        top_5 = ranked_markets[:5]
        summary_data = json.dumps(
            [
                {
                    "rank": m["rank"],
                    "market": m["market_name"],
                    "country": m["country"],
                    "composite_score": m["composite_score"],
                    "demand_score": m["demand_score"],
                    "supply_score": m["supply_score"],
                    "macro_score": m["macro_score"],
                }
                for m in top_5
            ],
            indent=2,
        )

        w_d = round(weights.get("demand", 0.4) * 100)
        w_s = round(weights.get("supply", 0.3) * 100)
        w_m = round(weights.get("macro",  0.3) * 100)

        user_message = f"""Write a concise portfolio-level executive summary for an APAC DC market
prioritisation study. Methodology weights: Demand {w_d}% | Supply {w_s}% | Macro {w_m}%.

Top 5 markets by composite score:
{summary_data}

Structure:
## APAC DC Market Prioritisation — Executive Summary
**Month Year | McKinsey Infrastructure Practice**

1. **Market Context** (~60 words): State of APAC DC demand, key macro tailwinds.
2. **Priority Markets** (~80 words): Characterise the top 3 markets and why they lead.
3. **Emerging Opportunities** (~60 words): Highlight 2–3 markets to watch and why.
4. **Strategic Implications** (~60 words): 2–3 portfolio-level recommendations.

Total: ~260 words. Write for a Managing Director audience."""

        yield from self._stream_text(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=1024,
        )
