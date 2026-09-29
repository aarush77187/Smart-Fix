"""
Where Phase 1 (extraction) and Phase 2 (deeplink mapping + sequencing)
meet. Takes a constraint-fixed Goal (no deeplinks yet) and returns the
final Goal: every eligible action's first stepGroup gets a real
actionableDeeplink (+ validationDeeplink when the catalog provides one),
or the dummy_positive placeholder, and actions are reordered safe-first /
critical-last.

Rules here come directly from the real catalog and spec, not from score
tuning:
  - category == "manual": NEVER gets a deeplink. A physical intervention
    cannot be one-tap actionable, regardless of match score.
  - category in {"auto", "critical"} with no catalog match above
    threshold: gets voiceassist://dummy_positive. Per the catalog's own
    dummy entry (id: DL-DUMMY), WE must author description/message
    ourselves in that case (5-7 words, naming the concrete screen) --
    it is not a generic copy-paste string.
  - A real catalog match's `validation` object (when present) maps
    directly onto StepGroup.validationDeeplink -- both share the same
    field shape (deeplink, key, resultType, condition, value).
"""

from __future__ import annotations

from app.api_contract import DUMMY_POSITIVE_DEEPLINK
from app.matcher import get_matcher, sequence_actions
from app.schema import Action, Deeplink, Goal, actionCategory


def _search_text_for_action(action: Action) -> str:
    step_text = " ".join(s for sg in action.stepGroups for s in sg.steps)
    return f"{action.actionName} {action.description} {step_text}"


def _author_dummy_description(action: Action) -> str:
    """Self-authored per the catalog's own instruction for dummy_positive:
    "Write description and message yourself (5-7 words, naming the
    concrete screen from the steps)." Built from the action name so it
    stays under 7 words and names the actual screen."""
    name = action.actionName.strip()
    words = f"Opens the {name} settings screen".split()
    return " ".join(words[:7])


def assemble_goal(goal: Goal) -> tuple[Goal, list[str]]:
    matcher = get_matcher()
    warnings: list[str] = []
    new_actions: list[Action] = []

    for action in goal.actions:
        if action.category == actionCategory.manual:
            new_actions.append(action)  # never gets a deeplink
            continue

        match = matcher.best_match(_search_text_for_action(action))

        if match:
            entry = match.entry
            deeplink = Deeplink(
                deeplink=entry.deeplink,
                description=entry.description,
                message=entry.message,
                originalType=entry.originalType,
            )
            validation_deeplink = entry.validation  # already a ValidationDeepLink or None
        else:
            deeplink = Deeplink(
                deeplink=DUMMY_POSITIVE_DEEPLINK,
                description=_author_dummy_description(action),
                message=action.actionName,
            )
            validation_deeplink = None
            warnings.append(
                f"Action '{action.actionName}' (category={action.category.value}) had no catalog "
                f"match above threshold ({match.score if match else 'n/a'}) -- assigned "
                f"dummy_positive with a self-authored description."
            )

        if not action.stepGroups:
            new_actions.append(action)
            continue

        first_group, *rest_groups = action.stepGroups
        updated_first = first_group.model_copy(update={
            "actionableDeeplink": deeplink,
            "validationDeeplink": validation_deeplink,
        })
        new_actions.append(action.model_copy(update={"stepGroups": [updated_first, *rest_groups]}))

    return goal.model_copy(update={"actions": sequence_actions(new_actions)}), warnings
