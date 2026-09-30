# Smart Guided Troubleshooting Engine

**Samsung PRISM GenAI Hackathon 2026-27 — Theme 02**  
**GitHub Tag:** `PRISM_GENAI_HACKATHON_Y2026`

Parses a customer complaint + a raw internal-KB reference article (`siis_response`) into a structured, validated troubleshooting plan with real in-app Settings deeplinks — grounded strictly in the provided reference text, with zero hallucinated URLs or steps.

---

## 📋 Hackathon Submission Deliverables

- **Presentation (PPTX):** [`MS_Ramaiah Institute of Technology._MSRIT_SmartFix__Submission_ppt.pptx`](MSRIT_SmartFix_Submission_ppt.pptx)
- **Demo Video:** [Watch Demo Video on Google Drive](https://drive.google.com/drive/folders/19F6xwTE1ZxT-JhygwhOUABaD-rETMz_4)
- **AI Disclosure:** [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md)
- **Evaluation Report:** [`metrics.md`](metrics.md) (100% schema compliance, 20/20 resolved, $0.00046 avg cost)
- **Evaluation Outputs:** [`results.jsonl`](results.jsonl) (all 20 structured responses)
- **Unit Test Suite:** [`tests/test_validation.py`](tests/test_validation.py) (16 automated tests)

---

## Quick start

```bash
export GROQ_API_KEY=your_key_here
docker compose up --build
```

API live at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

### Without Docker

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn app.main:app --reload
```

### Demo UI

```bash
streamlit run demo/streamlit_app.py
```

### Run the real 20-query batch (produces results.jsonl + metrics.md)

```bash
export GROQ_API_KEY=your_key_here
python -m scripts.run_batch
```

### Run unit tests (offline, zero-cost, 16 safety & schema checks)

```bash
python -m unittest tests/test_validation.py
```

## API contract

`POST /v1/troubleshoot`
```json
{
  "query": "My phone screen is completely black and won't turn on",
  "siis_response": { "title": "...", "content": "..." }
}
```
`siis_response` is optional — if omitted, the engine only checks the
semantic cache (Phase 3) and returns `{"contexts": [], "meta": {"fallback": "no_match"}}`
if nothing matches, rather than fabricating a plan with no grounding text.

`GET /health` → `{"status": "ok"}`

## Architecture

```
query + siis_response
  -> extract_goal()      Phase 1: LLM parses reference text into Goal/Action/StepGroup
  -> scrub_goal()         zero-URL-leak scrubbing (runs inside extraction, automatically)
  -> fix_goal()            Phase 1b: programmatic word-count/prefix/template correction
  -> assemble_goal()      Phase 2: BM25+TF-IDF deeplink matching + safe-first sequencing
  -> cache.put()           Phase 3: warms the semantic cache for future paraphrases
  -> TroubleshootResponse
```

Core guarantee: **the LLM never writes a deeplink.** It only produces
step text and an action category; `matcher.py` does an independent
lookup against the real `data/deeplinks.json` catalog (578 entries) via
BM25 + TF-IDF cosine similarity, matched only on `description` /
`message` / `qna_description` — never on the deeplink URI string itself.

If nothing in the catalog clears the confidence threshold for an
`auto`/`critical` action, the engine uses `voiceassist://dummy_positive`
(the catalog's own reserved placeholder) with a self-authored
description, exactly as the catalog's own `DL-DUMMY` entry instructs.
`manual` actions never get a deeplink at all.

See `app/*.py` — each file is short and documented at the top with why
it exists.

## Known limitations (disclosed, not hidden)

- **No real dense embeddings.** This environment has no internet access
  to download an embedding model, so matching uses BM25 + TF-IDF cosine
  (`matcher.py`) plus a small hand-picked synonym map (`text_utils.py`)
  as a local stand-in. This favors precision over recall — some
  legitimate but weakly-scored matches (e.g. a generic "turn on
  bluetooth") fall back to `dummy_positive` rather than risk a wrong
  deeplink. Swapping in real embeddings later is a contained change
  inside `DeeplinkMatcher` — the `match()`/`best_match()` interface
  doesn't change.
- **Reference text can be genuinely off-topic.** Several of the real 20
  queries in `data/siis_responses.json` are matched (by Samsung's own
  SIIS system) to KB articles that only tangentially address the actual
  symptom. Extraction's `score` field is designed to reflect this
  honestly rather than force high confidence.
- **Auto-corrected text isn't re-conjugated.** Constraint fixes (e.g.
  prepending "It will" to a description) can occasionally read slightly
  awkward grammatically. Word-count/prefix rules are still satisfied.

## Tech stack

Python, FastAPI, Pydantic, Groq API (`openai/gpt-oss-120b`),
rank-bm25 + scikit-learn (TF-IDF) for matching, Streamlit demo, Docker.
