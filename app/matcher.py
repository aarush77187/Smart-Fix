"""
Phase 2: Deeplink Mapping & Sequencing, against the REAL data/deeplinks.json
(578 entries, general Settings catalog -- no domain/category metadata of
its own; battery/display/camera/performance was an assumption from the
PDF overview that the real catalog doesn't reflect, so there is no domain
filter here, only free-text matching across the whole catalog).

Matching is DUAL RETRIEVAL: BM25 (keyword) + TF-IDF cosine (vector-space)
combined -- an honest, local substitute for "dense embeddings" as named
in the spec, since this sandbox has no internet access to download an
embedding model. The public interface (match/best_match) is what matters
for swapping in real embeddings later; only __init__/match's insides
would need to change.

Matching happens ONLY on description/message/qna_description -- never on
the deeplink URI string itself (the catalog's own _readme says the same:
"match on description, message, qna_description and originalType, then
copy the URI verbatim").
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.api_contract import DUMMY_POSITIVE_DEEPLINK, CatalogEntry
from app.config import settings
from app.schema import Action, actionCategory
from app.text_utils import normalize_synonyms


def _tokenize(text: str) -> list[str]:
    # Stopwords are stripped from both retrieval paths. Standard English
    # stopwords aren't enough here: nearly every catalog description shares
    # boilerplate like "Opens the X settings page in device Settings on the
    # device," and those repeated words inflate similarity for topically
    # WRONG matches (e.g. "clear cache" falsely matching "app notification
    # settings" at a deceptively high score, purely on shared boilerplate).
    # Left in, this silently breaks both the no_match threshold and the
    # precision of real matches. Synonym normalization runs first so
    # "pair"/"connect", "earbuds"/"headphones" etc. collide on the same
    # token -- see text_utils.py for why and its limits.
    tokens = re.findall(r"[a-z0-9]+", normalize_synonyms(text).lower())
    return [t for t in tokens if t not in _STOPWORDS]


_CATALOG_BOILERPLATE = {
    "opens", "settings", "page", "device", "via", "toggle", "toggles",
    "allow", "allows", "let", "lets", "your", "you",
}
_STOPWORDS = ENGLISH_STOP_WORDS | _CATALOG_BOILERPLATE


@dataclass
class MatchResult:
    entry: CatalogEntry
    score: float  # combined hybrid score, roughly 0..1


class DeeplinkMatcher:
    def __init__(self, catalog_path: str):
        raw = json.loads(Path(catalog_path).read_text())
        rows = raw["deeplinks"] if isinstance(raw, dict) else raw

        # The catalog's own dummy_positive row is excluded from the
        # searchable index -- it must never be "matched" by score, only
        # used explicitly as the assembler's fallback when nothing else
        # clears the threshold.
        self.entries: list[CatalogEntry] = [
            CatalogEntry(**r) for r in rows if r.get("deeplink") != DUMMY_POSITIVE_DEEPLINK
        ]

        self._texts = [normalize_synonyms(self._searchable_text(e)) for e in self.entries]
        self._tokenized = [_tokenize(t) for t in self._texts]

        self._bm25 = BM25Okapi(self._tokenized)
        self._tfidf_vectorizer = TfidfVectorizer(stop_words=list(_STOPWORDS))
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(self._texts)

    @staticmethod
    def _searchable_text(entry: CatalogEntry) -> str:
        return f"{entry.description} {entry.message} {entry.qna_description}"

    def match(self, query_text: str, top_k: int = 5) -> list[MatchResult]:
        tokens = _tokenize(query_text)
        bm25_raw = np.array(self._bm25.get_scores(tokens))

        query_vec = self._tfidf_vectorizer.transform([normalize_synonyms(query_text)])
        tfidf_raw = cosine_similarity(query_vec, self._tfidf_matrix)[0]

        # ABSOLUTE scoring, not per-query relative min-max: min-max
        # normalization guarantees the top candidate always gets pushed
        # toward 1.0 regardless of whether it's actually a good match --
        # that made the no_match threshold meaningless (confirmed by
        # testing: unrelated queries scored 0.98+ before this fix). BM25's
        # raw scale is far more discriminative than TF-IDF's on this
        # catalog (empirically: a true near-verbatim match scores ~40+,
        # while topically wrong "matches" driven by shared boilerplate
        # stay under ~12), so BM25 is weighted higher and capped against a
        # fixed, empirically-chosen ceiling rather than the query's own max.
        bm25_scaled = np.minimum(bm25_raw / settings.bm25_scale_cap, 1.0)
        combined = 0.7 * bm25_scaled + 0.3 * tfidf_raw

        ranked = sorted(range(len(self.entries)), key=lambda i: combined[i], reverse=True)[:top_k]
        return [MatchResult(entry=self.entries[i], score=float(combined[i])) for i in ranked]

    def best_match(self, query_text: str) -> "MatchResult | None":
        """Returns None (not a low-scoring guess) when nothing clears
        settings.match_score_threshold -- feeds the dummy_positive
        fallback in assembler.py, or a whole-response no_match if nothing
        in a Goal matches at all."""
        results = self.match(query_text, top_k=1)
        if not results or results[0].score < settings.match_score_threshold:
            return None
        return results[0]


_matcher: "DeeplinkMatcher | None" = None


def get_matcher() -> DeeplinkMatcher:
    global _matcher
    if _matcher is None:
        _matcher = DeeplinkMatcher(settings.deeplinks_path)
    return _matcher


def sequence_actions(actions: list[Action]) -> list[Action]:
    """Non-invasive settings first, critical/irreversible operations last.
    'manual' (physical intervention) sits in the middle -- neither a safe
    one-tap toggle nor a destructive action."""
    order = {actionCategory.auto: 0, actionCategory.manual: 1, actionCategory.critical: 2}
    return sorted(actions, key=lambda a: order.get(a.category, 1))
