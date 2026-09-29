"""
Non-negotiable operational constraint #1 from the spec: Zero URL Leaks.
Models frequently inject generic help URLs from pretraining memory
("Visit samsung.com/support") -- this must be caught programmatically,
never trusted to a prompt instruction alone (their own pitfall #5 says
prompt-only constraints are unreliable).

Scope: this scrubs LLM-GENERATED text fields only -- goal, title,
actionName, action description, and step text. It never touches the
`deeplink` field or a catalog-sourced Deeplink's description/message,
since those come from our trusted deeplinks.json catalog, not the model.
Scrubbing trusted data would be pointless and would mask a real catalog
match if one legitimately contained a URL-shaped string.
"""

from __future__ import annotations

import re

from app.schema import Action, Goal, StepGroup

_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
# Catches bare domains with no protocol -- "samsung.com/support" -- which is
# exactly the pitfall example given in the spec and would be missed by a
# protocol-only regex.
_BARE_DOMAIN_RE = re.compile(
    r"\b[a-zA-Z0-9-]+\.(?:com|net|org|io|co|in|dev)(?:/\S*)?\b", re.IGNORECASE
)


def scrub_text(text: str) -> tuple[str, bool]:
    """Returns (cleaned_text, had_leak). Markdown links are collapsed to
    just their label text (so the instruction still reads naturally);
    bare URLs and domains are removed outright."""
    if not text:
        return text, False

    had_leak = False

    def _md_repl(m: re.Match) -> str:
        nonlocal had_leak
        had_leak = True
        return m.group(1)

    text = _MARKDOWN_LINK_RE.sub(_md_repl, text)

    if _URL_RE.search(text):
        had_leak = True
        text = _URL_RE.sub("", text)

    if _BARE_DOMAIN_RE.search(text):
        had_leak = True
        text = _BARE_DOMAIN_RE.sub("", text)

    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\s+([.,!?])", r"\1", text)  # tidy up double spaces before punctuation left by removal
    return text, had_leak


def scrub_goal(goal: Goal) -> tuple[Goal, list[str]]:
    """Walks every LLM-generated text field in a Goal and returns a
    cleaned copy plus a list of human-readable warnings (one per leak
    found), so validation.py can log what was caught."""
    warnings: list[str] = []

    def _track(label: str, original: str, cleaned: str, had_leak: bool) -> None:
        if had_leak:
            warnings.append(f"URL leak scrubbed from {label}: '{original[:60]}' -> '{cleaned[:60]}'")

    goal_text, leak = scrub_text(goal.goal)
    _track("goal.goal", goal.goal, goal_text, leak)

    title_text, leak = scrub_text(goal.title)
    _track("goal.title", goal.title, title_text, leak)

    new_actions: list[Action] = []
    for action in goal.actions:
        name_text, leak = scrub_text(action.actionName)
        _track(f"action.actionName ({action.actionName})", action.actionName, name_text, leak)

        desc_text, leak = scrub_text(action.description)
        _track(f"action.description ({action.actionName})", action.description, desc_text, leak)

        new_step_groups: list[StepGroup] = []
        for sg in action.stepGroups:
            new_steps = []
            for step in sg.steps:
                cleaned_step, leak = scrub_text(step)
                _track("step", step, cleaned_step, leak)
                new_steps.append(cleaned_step)
            # actionableDeeplink / validationDeeplink are catalog-sourced --
            # deliberately NOT scrubbed here, see module docstring.
            new_step_groups.append(sg.model_copy(update={"steps": new_steps}))

        new_actions.append(action.model_copy(update={
            "actionName": name_text, "description": desc_text, "stepGroups": new_step_groups,
        }))

    cleaned_goal = goal.model_copy(update={"goal": goal_text, "title": title_text, "actions": new_actions})
    return cleaned_goal, warnings


def scrub_query_variations(variations: list[str]) -> tuple[list[str], list[str]]:
    """query_variations are LLM-generated paraphrases -- also scrubbed,
    since nothing stops a model from injecting a URL there either."""
    warnings: list[str] = []
    cleaned = []
    for v in variations:
        c, leak = scrub_text(v)
        if leak:
            warnings.append(f"URL leak scrubbed from query_variation: '{v[:60]}'")
        cleaned.append(c)
    return cleaned, warnings
