"""Base agent class — wraps the Claude API with a structured tool-use loop."""

from __future__ import annotations

import json
import os
import anthropic
from typing import Any, Dict, List, Optional


class BaseAgent:
    """Provides a reusable agentic loop backed by Claude.

    Subclasses define ``system_prompt``, tools (as raw JSON schemas +
    local callables), and a ``run()`` method that returns structured data.

    The loop forces Claude to call a designated *output* tool once it has
    reasoned through the evidence, giving us reliable structured output
    without requiring the beta messages.parse() endpoint.
    """

    MODEL = "claude-opus-4-6"

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_agentic_loop(
        self,
        system_prompt: str,
        user_message: str,
        tools: List[Dict[str, Any]],
        output_tool_name: str,
        max_iterations: int = 6,
    ) -> Optional[Dict[str, Any]]:
        """Run the agentic loop and return the input of the first call
        to *output_tool_name*.

        ``tools`` is a list of dicts, each containing:
            - ``schema``: Anthropic tool schema (name, description, input_schema)
            - ``fn`` (optional): callable(input_dict) → str/dict; if None the
              tool is treated as pure output capture.
        """
        api_tools       = [t["schema"] for t in tools]
        tool_fns        = {t["schema"]["name"]: t.get("fn") for t in tools}
        messages: list  = [{"role": "user", "content": user_message}]

        for _ in range(max_iterations):
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=4096,
                thinking={"type": "adaptive"},
                system=system_prompt,
                tools=api_tools,
                # Force Claude to use at least one tool on every turn
                tool_choice={"type": "any"},
                messages=messages,
            )

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            # Capture output tool call immediately
            for block in tool_use_blocks:
                if block.name == output_tool_name:
                    return block.input

            # Execute side-effect tools and feed results back
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in tool_use_blocks:
                fn = tool_fns.get(block.name)
                if fn:
                    try:
                        result = fn(**block.input)
                        content = (
                            json.dumps(result)
                            if not isinstance(result, str)
                            else result
                        )
                    except Exception as exc:
                        content = f"Error: {exc}"
                else:
                    content = f"Tool {block.name} is not available."

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                })
            messages.append({"role": "user", "content": tool_results})

            if response.stop_reason == "end_turn":
                break

        return None

    # ------------------------------------------------------------------
    # Streaming helper (used by NarrativeAgent)
    # ------------------------------------------------------------------

    def _stream_text(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 2048,
    ):
        """Yield text tokens from a streaming Claude call."""
        with self.client.messages.stream(
            model=self.MODEL,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text
