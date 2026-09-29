"""
Everything in schema.py is Samsung's exact contract (Appendix A) -- copied
verbatim, untouched. This file holds OUR additions: the request/response
envelope needed to expose this as a real API, and the shape of one row of
the real deeplinks.json catalog. Keeping these separate means schema.py
can be diffed against the original file with zero drift.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.schema import ContextDeeplinkResponse, ValidationDeepLink, actionCategory

# Confirmed from the real deeplinks.json's own dummy entry (id: DL-DUMMY).
DUMMY_POSITIVE_DEEPLINK = "voiceassist://dummy_positive"


# =============================================================================
# API request/response envelope (ours -- not part of Appendix A)
# =============================================================================

class SiisResponsePayload(BaseModel):
    """Matches the real siis_responses.json shape exactly: {title, content}.
    This is what the grader actually sends in POST /v1/troubleshoot's
    siis_response field -- confirmed from the real file's _readme:
    "siis_response is the payload your API must accept"."""
    title: str
    content: str

    def as_reference_text(self) -> str:
        return f"{self.title}\n\n{self.content}"


class TroubleshootRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    siis_response: Optional[SiisResponsePayload] = None


class ResponseMeta(BaseModel):
    latency_ms: float
    cache_hit: bool
    model: str
    cost_usd: float
    fallback: Optional[str] = None  # "no_match" when contexts is empty


class TroubleshootResponse(BaseModel):
    query: str
    response: ContextDeeplinkResponse
    meta: ResponseMeta


# =============================================================================
# Catalog entry -- shape of one row of the REAL data/deeplinks.json
# =============================================================================

class CatalogEntry(BaseModel):
    id: str
    deeplink: str
    description: str
    message: str = ""
    originalType: Optional[str] = None
    control_type: Optional[int] = None
    qna_description: str = ""
    # The catalog's own validation object maps field-for-field onto
    # ValidationDeepLink (deeplink, key, resultType, condition, value) --
    # imported directly rather than redefined.
    validation: Optional[ValidationDeepLink] = None
