"""
Pure orchestration -- every real decision lives in the module that owns
it. This file just implements the exact request flow from the spec:

  siis_response provided:
    extract_goal (Phase 1, LLM) -> fix_goal (word-count/template rules)
    -> assemble_goal (Phase 2: deeplink matching + sequencing)
    -> cache.put (warms Phase 3 for future paraphrases)
    -> return contexts=[goal]

  siis_response omitted:
    cache.get (Phase 3 semantic lookup) -> hit: return cached goal
                                         -> miss: return contexts=[],
                                                  fallback="no_match"
                                                  (no reference text means
                                                  nothing to safely derive
                                                  steps from -- pitfall #3)

Never fabricates a fallback plan of our own when the LLM path fails --
that would just be a different flavor of hallucination. A failure at any
point routes to the same honest no_match response. That failure reason is
still LOGGED to the server console (not swallowed silently) -- a no_match
caused by a real config problem (missing API key, deprecated model, etc.)
should be loud in the terminal even though it's quiet in the API response.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI

from app.api_contract import ResponseMeta, TroubleshootRequest, TroubleshootResponse
from app.assembler import assemble_goal
from app.cache import cache
from app.constraints import fix_goal
from app.extraction import NoReferenceTextError, extract_goal
from app.llm import LLMGenerationError, estimate_cost_usd
from app.matcher import get_matcher
from app.schema import ContextDeeplinkResponse

logger = logging.getLogger("troubleshoot")

app = FastAPI(title="Smart Guided Troubleshooting Engine")


@app.on_event("startup")
def _startup() -> None:
    get_matcher()  # build the BM25/TF-IDF index once at boot, not per-request


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/debug/catalog-stats")
def catalog_stats() -> dict:
    """Not part of the spec's contract -- a convenience endpoint for
    inspecting the loaded catalog size during development/demo."""
    matcher = get_matcher()
    return {"indexed_entries": len(matcher.entries)}


@app.get("/debug/cache-stats")
def cache_stats() -> dict:
    return cache.stats()


def _no_match_response(
    query: str, start_time: float, reason: str, model: str = "none", cost_usd: float = 0.0
) -> TroubleshootResponse:
    # Every no_match says WHY in the server log -- "no_match" alone can mean
    # a config problem, an API failure, a schema failure, or a model that
    # legitimately found nothing, and those need very different responses.
    logger.warning("no_match for query %r -- reason: %s", query[:70], reason)
    return TroubleshootResponse(
        query=query,
        response=ContextDeeplinkResponse(contexts=[]),
        meta=ResponseMeta(
            latency_ms=(time.perf_counter() - start_time) * 1000,
            cache_hit=False,
            model=model,
            cost_usd=cost_usd,
            fallback="no_match",
        ),
    )


def run_troubleshoot(req: TroubleshootRequest) -> TroubleshootResponse:
    """Core pipeline logic, factored out of the endpoint so the batch
    runner (scripts/run_batch.py) can call the exact same code path used
    in production, rather than a second, divergent copy of it."""
    start = time.perf_counter()

    if req.siis_response is None:
        hit = cache.get(req.query)
        if hit is not None:
            return TroubleshootResponse(
                query=req.query,
                response=ContextDeeplinkResponse(contexts=[hit.goal]),
                meta=ResponseMeta(
                    latency_ms=(time.perf_counter() - start) * 1000,
                    cache_hit=True, model="cache", cost_usd=0.0,
                ),
            )
        return _no_match_response(req.query, start, reason="no siis_response provided and no semantic-cache hit")

    reference_text = req.siis_response.as_reference_text()

    try:
        goal, llm_result, _scrub_warnings = extract_goal(req.query, reference_text)
    except (NoReferenceTextError, LLMGenerationError) as e:
        return _no_match_response(req.query, start, reason=f"extraction failed: {e}")

    if not goal.actions:
        # The model call SUCCEEDED and parsed, but produced zero actions. Report
        # the real cost -- a call was made and paid for.
        return _no_match_response(
            req.query, start,
            reason=f"model returned a valid Goal with ZERO actions (title={goal.title!r}, score={goal.score})",
            model=llm_result.model,
            cost_usd=estimate_cost_usd(llm_result.model, llm_result.input_tokens, llm_result.output_tokens),
        )

    fixed_goal, _fix_warnings = fix_goal(goal)
    final_goal, _assemble_warnings = assemble_goal(fixed_goal)

    cache.put(req.query, final_goal)

    cost = estimate_cost_usd(llm_result.model, llm_result.input_tokens, llm_result.output_tokens)
    return TroubleshootResponse(
        query=req.query,
        response=ContextDeeplinkResponse(contexts=[final_goal]),
        meta=ResponseMeta(
            latency_ms=(time.perf_counter() - start) * 1000,
            cache_hit=False, model=llm_result.model, cost_usd=cost,
        ),
    )


@app.post("/v1/troubleshoot", response_model=TroubleshootResponse)
def troubleshoot(req: TroubleshootRequest) -> TroubleshootResponse:
    return run_troubleshoot(req)
