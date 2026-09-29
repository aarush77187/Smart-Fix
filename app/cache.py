"""
Phase 3: Fast-Path Semantic Cache.

Deliberately NOT exact-string keying -- the spec names that as a named
pitfall ("Exact-String Cache Keying... paraphrase queries miss the
cache"). Instead, a query is compared against every previously cached
query via TF-IDF cosine similarity (the same no-internet, local stand-in
for dense embeddings used in matcher.py) and counts as a hit above a
similarity threshold, regardless of exact wording.

Kept as a simple in-memory list + a TF-IDF vectorizer refit on demand --
appropriate for a prototype's scale (dozens to low hundreds of cached
queries), not for millions. The get/put interface is what would carry
over to a real vector-DB-backed cache later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.schema import Goal
from app.text_utils import normalize_synonyms


@dataclass
class CacheEntry:
    query: str
    goal: Goal
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0


class SemanticCache:
    def __init__(self) -> None:
        self._entries: list[CacheEntry] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._dirty = True  # index needs rebuilding

    def _rebuild_index(self) -> None:
        if not self._entries:
            self._vectorizer = None
            self._matrix = None
            self._dirty = False
            return
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([normalize_synonyms(e.query) for e in self._entries])
        self._dirty = False

    def get(self, query: str) -> "CacheEntry | None":
        if self._dirty:
            self._rebuild_index()
        if not self._entries or self._vectorizer is None:
            return None

        expired_cutoff = time.time() - settings.cache_ttl_seconds
        qv = self._vectorizer.transform([normalize_synonyms(query)])
        sims = cosine_similarity(qv, self._matrix)[0]
        best_i = sims.argmax()

        if sims[best_i] < settings.semantic_cache_threshold:
            return None
        entry = self._entries[best_i]
        if entry.created_at < expired_cutoff:
            return None

        entry.hit_count += 1
        return entry

    def put(self, query: str, goal: Goal) -> None:
        self._entries.append(CacheEntry(query=query, goal=goal))
        self._dirty = True

    def stats(self) -> dict:
        return {"entries": len(self._entries)}


cache = SemanticCache()
