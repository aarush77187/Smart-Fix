"""
The ONLY file that talks to the Groq API directly.

Why this exists as its own module: extraction.py should read like "here is
our prompt, here is the shape we expect back" -- not be cluttered with
retry loops, markdown-fence stripping, token bookkeeping, or provider
error handling. That plumbing lives here, once.

If you ever swap providers, this is the only file that changes.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Type, TypeVar

import groq
from pydantic import BaseModel, ValidationError

from app.config import settings

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("troubleshoot.llm")

_client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# Models known to support Groq's response_format={"type": "json_object"}.
# When the model isn't in this set we skip the flag rather than risk a
# 400 from the API -- the prompt's "JSON only" instruction plus the
# strip-fences/retry path still catches almost everything either way.
_JSON_MODE_MODELS = {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}

# Reasoning models -- get reasoning_effort capped to keep latency/cost
# down for a task that doesn't need multi-step reasoning, just structured
# extraction (see config.py's llm_reasoning_effort comment).
_REASONING_MODELS = {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class LLMResult:
    """Everything downstream code needs from one LLM call."""
    parsed: BaseModel
    input_tokens: int
    output_tokens: int
    attempts: int          # 1 = succeeded first try, 2 = needed the retry
    model: str


class LLMGenerationError(Exception):
    """Raised when the model still won't produce valid output after retrying,
    OR when the underlying API call itself fails (missing/invalid API key,
    network error, rate limit, etc). Callers (main.py) treat all of these
    the same way: degrade to a no_match response rather than crash --
    a transient provider issue should never surface as a raw 500."""


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _drop_stray_closers(text: str) -> str:
    """Removes closing brackets that don't match the currently-open one.

    Observed in real runs: the model emits valid content with ONE extra "}"
    (e.g. after a stepGroups array), which makes the whole reply invalid JSON.
    This only ever DELETES a mismatched closer -- it never adds, edits or
    reorders content -- so anything it recovers is still exactly what the model
    wrote, and still goes through full Pydantic validation afterwards.
    String contents (including escaped quotes) are skipped so braces inside
    step text are never touched."""
    out: list[str] = []
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                continue  # stray closer -> drop it
        out.append(ch)
    return "".join(out)


def _parse_json_tolerant(text: str):
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = _drop_stray_closers(cleaned)
        if repaired == cleaned:
            raise
        data = json.loads(repaired)  # if this still fails, the retry path handles it
        logger.warning("Repaired malformed model JSON by dropping stray closing bracket(s).")
        return data


def _failed_generation(err: "groq.BadRequestError") -> "str | None":
    """Groq's JSON mode rejects invalid JSON with a 400 whose body carries the
    model's output in `failed_generation`. Returns it, or None if this 400 is
    some other kind of bad request."""
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and inner.get("code") == "json_validate_failed":
            fg = inner.get("failed_generation")
            if isinstance(fg, str) and fg.strip():
                return fg
    return None


def _call(model: str, system: str, user_prompt: str, max_tokens: int) -> tuple[str, int, int]:
    if not _client.api_key:
        # Fails fast with a clear message instead of letting an empty
        # "Bearer " header crash three layers deep inside httpcore with a
        # cryptic LocalProtocolError -- this is almost always a forgotten
        # `export GROQ_API_KEY=...` in the current shell session.
        raise LLMGenerationError(
            "GROQ_API_KEY is not set (or is empty) in this process's environment. "
            "Set it in the SAME terminal session before starting the server, e.g. "
            "PowerShell: $env:GROQ_API_KEY=\"your_key\" ; bash: export GROQ_API_KEY=your_key"
        )

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=settings.llm_temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
    )
    if model in _JSON_MODE_MODELS:
        # Provider-level enforcement that the reply IS valid JSON. This does
        # NOT guarantee it matches our Pydantic schema -- that's still
        # checked below -- but it eliminates most stray "Sure, here's the
        # JSON:" preambles before we even get to parsing.
        kwargs["response_format"] = {"type": "json_object"}
    if model in _REASONING_MODELS:
        kwargs["reasoning_effort"] = settings.llm_reasoning_effort

    try:
        response = _client.chat.completions.create(**kwargs)
    except groq.AuthenticationError as e:
        raise LLMGenerationError(f"Groq authentication failed -- check GROQ_API_KEY is valid: {e}") from e
    except groq.NotFoundError as e:
        raise LLMGenerationError(
            f"Groq model '{model}' not found or not accessible on this account "
            f"(often means it was deprecated -- check https://console.groq.com/docs/deprecations "
            f"and update settings.generation_model in config.py): {e}"
        ) from e
    except groq.BadRequestError as e:
        failed = _failed_generation(e)
        if failed is None:
            raise LLMGenerationError(f"Groq rejected the request (400): {e}") from e
        logger.warning(
            "Groq rejected the model's JSON (json_validate_failed) -- recovering the raw "
            "output from the error and attempting a local repair instead of discarding it."
        )
        # Token counts aren't reported on an error response; estimate (~4 chars/token)
        # so cost isn't reported as zero for a call that was actually made.
        return failed, (len(system) + len(user_prompt)) // 4, len(failed) // 4
    except groq.APIConnectionError as e:
        raise LLMGenerationError(f"Could not reach Groq's API (network issue): {e}") from e
    except groq.RateLimitError as e:
        raise LLMGenerationError(f"Groq rate limit hit: {e}") from e
    except groq.APIError as e:
        raise LLMGenerationError(f"Groq API error: {e}") from e

    choice = response.choices[0]
    if choice.finish_reason == "length":
        logger.warning(
            "Model output was TRUNCATED by the token limit (max_tokens=%s) -- the JSON is "
            "likely incomplete and will fail validation. Raise generation_max_tokens in config.py.",
            max_tokens,
        )
    text = choice.message.content or ""
    usage = response.usage
    return text, usage.prompt_tokens, usage.completion_tokens


def generate_structured(
    *,
    model: str,
    system: str,
    user_prompt: str,
    schema: Type[T],
    max_tokens: int,
) -> LLMResult:
    """Call the model, parse its reply as JSON, validate against `schema`.

    On a parse/validation failure, retries ONCE with the error message
    appended to the prompt -- this mirrors the "one retry, then degrade"
    contract used later in validation.py for the full pipeline. If the
    retry also fails, raises LLMGenerationError so the caller can fall back
    (e.g. to canonical_steps) instead of crashing the request.
    """
    total_input_tokens = 0
    total_output_tokens = 0
    last_error: Exception | None = None
    prompt = user_prompt

    max_attempts = 1 + settings.llm_retry_on_parse_error
    for attempt in range(1, max_attempts + 1):
        raw_text, in_tok, out_tok = _call(model, system, prompt, max_tokens)
        total_input_tokens += in_tok
        total_output_tokens += out_tok

        try:
            data = _parse_json_tolerant(raw_text)
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                data = data[0]  # model wrapped the one object in a list
            if isinstance(data, dict) and isinstance(data.get("actions"), list):
                # Clean up any empty strings, nulls, or stray non-dict elements from actions array
                data["actions"] = [a for a in data["actions"] if isinstance(a, dict) and a]
            parsed = schema(**data)
            return LLMResult(
                parsed=parsed,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                attempts=attempt,
                model=model,
            )
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            last_error = e
            logger.warning(
                "Parse attempt %d/%d failed (%s). Model output started with: %r | error: %s",
                attempt, max_attempts, type(e).__name__, raw_text[:150].replace("\n", " "), str(e)[:300].replace("\n", " "),
            )
            prompt = (
                f"{user_prompt}\n\n"
                f"Your previous reply could not be parsed: {e}\n"
                f"IMPORTANT: The 'actions' field must be an array of JSON objects. Every element inside 'actions' must be an object with {{\"actionName\": \"...\", \"description\": \"...\", \"category\": \"...\", \"stepGroups\": [...]}}.\n"
                f"Never include empty strings, raw strings, or colons directly in the 'actions' list.\n"
                f"Reply with ONLY valid JSON matching the required shape. "
                f"No markdown, no commentary, no code fences."
            )

    raise LLMGenerationError(
        f"Model '{model}' failed to produce valid {schema.__name__} JSON "
        f"after {max_attempts} attempt(s): {last_error}"
    )


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price = settings.price_per_1k_input.get(model, 0.0)
    out_price = settings.price_per_1k_output.get(model, 0.0)
    return (input_tokens / 1000) * in_price + (output_tokens / 1000) * out_price
