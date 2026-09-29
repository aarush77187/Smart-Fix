"""
Pitfall #5 from the spec, verbatim: "Prompt-Only Constraints: Asking an LLM
to respect word count constraints in natural language is unreliable.
Enforce programmatic validation, trimming, and correction loops in the
application layer." This file is that application layer.

Philosophy: fix what's mechanically fixable (truncate an over-long field,
add a missing prefix, rebuild a malformed template sentence), warn about
what isn't (a description too short to safely pad without inventing
content -- padding would itself be a hallucination). Never fabricate
domain facts to satisfy a word count.

Also enforces "one action = one screen" at the code level (merge actions
sharing an actionName) rather than trusting the extraction prompt alone.
"""

from __future__ import annotations

import re

from app.schema import Action, Goal, StepGroup

_GOAL_TEMPLATE_RE = re.compile(
    r"^Follow these steps to perform this .+ (Troubleshooting|Configuration)$"
)


def _word_count(text: str) -> int:
    return len(text.split())


def _fix_title(title: str, warnings: list[str]) -> str:
    words = title.strip().split()
    if len(words) > 3:
        warnings.append(f"title had {len(words)} words, truncated to 3: '{title}'")
        words = words[:3]
    elif len(words) < 2:
        warnings.append(f"title had only {len(words)} word(s), below the 2-3 word target: '{title}'")
        # Not padded -- inventing a second word would misrepresent the issue.
    if words:
        words = [words[0].capitalize()] + [w.lower() for w in words[1:]]
    return " ".join(words)


def _fix_description(description: str, warnings: list[str]) -> str:
    text = description.strip()
    if not text.lower().startswith("it will"):
        original_dec = text[0].lower() + text[1:] if text else text
        text = f"It will {original_dec}"
        warnings.append(f"description missing 'It will' prefix, prepended: '{description}'")

    words = text.split()
    if len(words) > 7:
        warnings.append(f"description had {len(words)} words, truncated to 7: '{description}'")
        words = words[:7]
    elif len(words) < 5:
        warnings.append(
            f"description has only {len(words)} word(s), below the 5-7 word target: "
            f"'{description}' -- left as-is rather than padded, to avoid inventing content."
        )
    return " ".join(words)


def _fix_goal_sentence(goal_text: str, title: str) -> tuple[str, bool]:
    if _GOAL_TEMPLATE_RE.match(goal_text.strip()):
        return goal_text.strip(), False
    topic = title.title() if title else "Device"
    rebuilt = f"Follow these steps to perform this {topic} Troubleshooting"
    return rebuilt, True


def _merge_duplicate_actions(actions: list[Action], warnings: list[str]) -> list[Action]:
    """Enforces 'one action = one screen' -- if the model produced two
    actions with the same actionName (case-insensitive), merge their
    stepGroups into the first one rather than emitting two screens for
    the same feature."""
    merged: dict[str, Action] = {}
    order: list[str] = []

    for action in actions:
        key = action.actionName.strip().lower()
        if key not in merged:
            merged[key] = action
            order.append(key)
        else:
            warnings.append(
                f"Duplicate action for the same screen ('{action.actionName}') -- merged "
                f"stepGroups into the first occurrence rather than creating two actions."
            )
            existing = merged[key]
            combined_step_groups = existing.stepGroups + action.stepGroups
            merged[key] = existing.model_copy(update={"stepGroups": combined_step_groups})

    return [merged[k] for k in order]


def fix_goal(goal: Goal) -> tuple[Goal, list[str]]:
    """Applies every programmatic fix and returns (corrected Goal, warnings).
    Never raises -- this is a best-effort correction pass, not a gate. A
    genuinely broken Goal (e.g. zero actions) is still valid Pydantic-wise;
    callers decide whether an empty/thin result should become a no_match."""
    warnings: list[str] = []

    fixed_title = _fix_title(goal.title, warnings)

    fixed_actions = []
    for action in goal.actions:
        fixed_description = _fix_description(action.description, warnings)
        fixed_actions.append(action.model_copy(update={"description": fixed_description}))
    fixed_actions = _merge_duplicate_actions(fixed_actions, warnings)

    fixed_goal_text, was_rebuilt = _fix_goal_sentence(goal.goal, fixed_title)
    if was_rebuilt:
        warnings.append(f"goal sentence did not match the required template, rebuilt: '{goal.goal}'")

    corrected = goal.model_copy(update={
        "title": fixed_title,
        "goal": fixed_goal_text,
        "actions": fixed_actions,
    })
    return corrected, warnings
