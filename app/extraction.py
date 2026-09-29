"""
Phase 1: Structure Extraction.

Parses UNSTRUCTURED siis_response reference text into a Goal object --
title, goal sentence, actions (grouped by physical screen), step text.
Deliberately produces NO deeplinks here (that's matcher.py's job, applied
afterward) and does NOT try to enforce exact word-count rules in the
prompt (that's constraints.py's job, applied afterward) -- each stage
owns exactly one concern, per their own "Core Engineering Challenge"
column in the pipeline table.

The hard rule this file exists to serve: pitfall #3, "No Hallucinated
Steps" -- a troubleshooting plan must derive PURELY from the provided
reference text. Enforced two ways:
  1. This function refuses to run at all without real siis_response text
     (see the guard below) -- no reference text means nothing to ground
     extraction in, so main.py must route that case to cache/no_match
     instead of calling this.
  2. The prompt is explicit that inventing a settings path or step not
     present in the reference text is a failure, not a nice-to-have.
"""

from __future__ import annotations

from app.config import settings
from app.llm import LLMResult, generate_structured
from app.schema import Goal
from app.scrubber import scrub_goal


class NoReferenceTextError(ValueError):
    """Raised when siis_response is empty -- there's nothing to ground
    extraction in, so this stage must not run (caller should fall back to
    cache lookup or no_match instead)."""


_SYSTEM_PROMPT = """You are a structure-extraction component in a device \
troubleshooting system. You are given RAW, UNSTRUCTURED customer-support \
reference text and a customer's complaint. Your job is to parse the \
reference text into a structured troubleshooting plan.

CRITICAL RULE: every step and every action you produce must be DERIVED \
DIRECTLY from the reference text below. Do not invent a settings path, a \
step, or a screen name that is not mentioned or clearly implied by the \
reference text. If the reference text does not actually address the \
customer's complaint, still extract whatever it DOES say -- do not pad \
with generic advice from your own training.

Output ONLY a JSON object with this shape:
{
  "goal": "Follow these steps to perform this <Topic> Troubleshooting",
  "title": "<2-3 word sentence-case summary of the core issue>",
  "score": <float 0.0-1.0, your confidence the reference text actually addresses the complaint>,
  "actions": [
    {
      "actionName": "<Title Case name of first screen or feature>",
      "description": "<starts with 'It will', explains the concrete benefit in 5-7 words>",
      "category": "auto",
      "stepGroups": [
        {
          "steps": ["<clear imperative instruction, one UI interaction per step>"]
        }
      ]
    },
    {
      "actionName": "<Title Case name of second screen or feature>",
      "description": "<starts with 'It will', explains the concrete benefit in 5-7 words>",
      "category": "manual",
      "stepGroups": [
        {
          "steps": ["<clear imperative instruction, one UI interaction per step>"]
        }
      ]
    }
  ]
}

Rules:
- Replace <Topic> with the actual subject (e.g. "Battery Fast Drain", "Swipe Navigation").
- "actions" MUST be a JSON array of JSON OBJECTS. Every element in "actions" must be an object with actionName, description, category, and stepGroups. Never put strings, colons, or empty values in the actions array.
- If multiple steps happen on the SAME screen/feature, group them under ONE action -- never create two actions for the same screen. One action = one screen.
- category: "auto" = a standard settings screen the customer can reach themselves. "critical" = disruptive or irreversible (factory reset, restart, firmware update, safe mode) -- these must be the LAST actions if present. "manual" = a physical intervention (e.g. cleaning a lens, force restart buttons, connecting cables) that cannot have a deeplink.
- No URLs, no markdown links, no web addresses anywhere in any field.
- Do not include any text outside the JSON object.
"""


def extract_goal(query: str, siis_response: str) -> tuple[Goal, LLMResult, list[str]]:
    """Returns (scrubbed Goal, llm metadata, scrub warnings).

    Raises NoReferenceTextError if siis_response is empty -- this is a
    deliberate hard stop, not something to soften into a default, because
    running extraction with no source text is exactly the hallucination
    risk pitfall #3 warns about.
    """
    if not siis_response or not siis_response.strip():
        raise NoReferenceTextError(
            "extract_goal() called with no siis_response text -- refusing to run, "
            "since extraction with no reference text has nothing to ground it. "
            "Route this case to a cache lookup or a no_match response instead."
        )

    user_prompt = f"""Customer complaint: "{query}"

Reference text (this is your ONLY source of truth -- do not add anything not in here):
\"\"\"
{siis_response.strip()}
\"\"\"
"""

    result = generate_structured(
        model=settings.generation_model,
        system=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=Goal,
        max_tokens=settings.generation_max_tokens,
    )

    goal: Goal = result.parsed  # type: ignore[assignment]
    scrubbed_goal, warnings = scrub_goal(goal)
    return scrubbed_goal, result, warnings
