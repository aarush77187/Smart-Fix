# System Performance Metrics & Evaluation Report
**Model(s):** openai/gpt-oss-120b (extraction)
**Environment:** sandboxed container, no GPU

---

## 1. Schema & Rule Compliance
Evaluated on the real 20-query dataset (data/siis_responses.json).

| Metric | Target | Measured Value |
|---|---|---|
| Schema-valid output lines | >=99% | 100% |
| Absolute URL leaks | 0 | 0 (scrubbed programmatically, see scrubber.py) |
| Deeplink catalog validity (exact URI match) | 100% | 100% (only IDs copied verbatim from matcher.py, never LLM-generated) |

---

## 2. Accuracy Benchmarks
Evaluated against the real 20-query set. No independently-labelled ground truth was
provided beyond sample_output.json's single worked example, so this reports
observed pipeline behavior rather than a scored accuracy metric.

| Metric | Value |
|---|---|
| Queries processed | 20 |
| Queries resolved with actions | 20 |
| Queries resulting in no_match | 0 |
| Total actions generated | 50 |
| Category distribution | {'auto': 18, 'manual': 32} |
| Actions using a real catalog deeplink | 17 |
| Actions using dummy_positive fallback | 1 |

---

## 3. Latency Benchmarks (N=20 requests)

| Execution Path | P50 (ms) | P95 (ms) |
|---|---|---|
| Cold query (full pipeline: extraction + matching) | 8694 | 22053 |

---

## 4. Operational Cost & Cache Efficacy

| Metric Item | Measured Value |
|---|---|
| Total cost across 20 queries | $0.00927 |
| Average cost per query | $0.00046 |
| Cost derivation method | (prompt_tokens + completion_tokens) x per-model rate, see config.py |

---

## 5. Known Edge Cases & System Limitations

- Matching is BM25 + TF-IDF cosine, not true dense embeddings -- this sandbox has no
  internet access to download an embedding model. A hand-picked synonym map
  (text_utils.py) mitigates the most common paraphrase gaps (pair/connect,
  earbuds/headphones, etc.) but does not generalize the way real embeddings would.
- match_score_threshold (0.45) favors precision over recall: some legitimate but
  weakly-scored matches (e.g. a generic "turn on bluetooth" query) fall back to
  dummy_positive rather than risk a topically wrong deeplink.
- Several real queries in this dataset have siis_response reference text that only
  tangentially addresses the actual symptom (e.g. a screen-flicker complaint matched
  to a generic email-connectivity KB article) -- the extraction stage's `score` field
  is designed to reflect this honestly rather than force full confidence.
- Grammar of auto-corrected fields (e.g. a "It will" prefix prepended to a description
  that started mid-sentence) is not re-conjugated, so occasional awkward phrasing like
  "It will shows..." can occur. Word-count/prefix rules are still satisfied.
