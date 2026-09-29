# AI Usage DISCLOSURE FORM

## 1. Team Details
- **Team Name:** MSRIT_SMARTFIX
- **Project / Product Name:** Smart Guided Troubleshooting Engine (Theme 02)
- **Organization / Institution (if any):** MS Ramaiah Institute of Technology
- **Submission Date:** September 30, 2026

---

## 2. AI Usage Declaration
- **Did your team use any Artificial Intelligence (AI) in developing this project?**  
  **Yes**
- **Details:**  
  Yes. AI was utilized in two distinct roles:
  1. **Runtime System Component:** An LLM (`openai/gpt-oss-120b` / `llama-3.3-70b-versatile` via Groq API, temperature=0) operates within the production pipeline solely as a constrained **Phase 1 Structure Extraction engine** to parse unstructured customer-support reference text into standardized JSON.
  2. **Development Assistance:** AI coding assistants (Anthropic Claude and Google Antigravity) were utilized for code scaffolding, prompt optimization, regex design, and unit test generation.

---

## 3. Purpose of AI Usage (Brief Details)
- **Idea generation / brainstorming:**  
  Analyzed device troubleshooting failure modes (hallucinated settings paths, URL leaks, dangerous step sequencing) and designed the 5-layer decoupled architecture with a strict anti-hallucination boundary.
- **Code generation or assistance:**  
  Assisted in implementing the FastAPI REST endpoint structure, Pydantic v2 data models, Okapi BM25 + TF-IDF cosine dual-retrieval matcher, and tolerant JSON bracket-repair routines.
- **UI / UX design:**  
  Assisted in structuring the Streamlit interactive frontend layout, visual safety badges (🟢 Auto, 🟡 Manual, 🔴 Critical), real-time telemetry metrics, and the interactive HTML5 video presentation player.
- **Content creation:**  
  Assisted in drafting technical documentation (`README.md`, `metrics.md`), the 12-slide presentation deck matching the official Samsung PRISM template, and video voiceover scripts.
- **Data analysis:**  
  Assisted in analyzing the 20 real benchmark scenarios from `data/siis_responses.json`, evaluating action category distributions, token consumption, and latency percentiles (P50/P95).
- **Testing / debugging:**  
  Assisted in authoring the 16 offline automated unit tests (`tests/test_validation.py`) and debugging Groq rate-limit backoffs, bracket repairs, and list-wrapping issues.
- **Other (Runtime LLM Component):**  
  The Groq API executes in-pipeline structure extraction (`app/extraction.py`), strictly grounded in Samsung Internal Information System (SIIS) reference text.

---

## 4. Feature Origin Classification

### Feature 1: LLM-Powered Structure Extraction (`app/extraction.py`)
- **Self-Generated / AI-Generated / Both:** Both
- **Description:**
  - *AI Tools/Platform Used:* Groq API (`openai/gpt-oss-120b`, `llama-3.3-70b-versatile`), Claude, Google Antigravity.
  - *Prompt Used:* System prompt with explicit two-action few-shot JSON schema instructing the model to parse unstructured SIIS reference text into Goal, Title, Action names, step texts, and category (`auto`, `manual`, `critical`) with zero invented paths.
  - *Output Summary:* Structured JSON object conforming to Pydantic `Goal` schema with grounded troubleshooting instructions.
  - *Modification:* Implemented a hardcoded guard (`NoReferenceTextError`) that aborts extraction when reference text is empty to prevent ungrounded hallucination; implemented a programmatic bracket-repair tokenizer in `app/llm.py` to recover valid JSON from stray brackets without altering content.

### Feature 2: Anti-Hallucination Deeplink Matcher (`app/matcher.py` & `app/assembler.py`)
- **Self-Generated / AI-Generated / Both:** Self-Generated (Assisted by AI for optimization)
- **Description:**
  - *AI Tools/Platform Used:* Claude / Antigravity for code review and vector algebra assistance.
  - *Prompt Used:* None (100% deterministic non-LLM algorithm).
  - *Output Summary:* Maps extracted natural-language actions to official URIs from the 578-entry `data/deeplinks.json` catalog using dual-retrieval (BM25 Okapi keyword matching + TF-IDF cosine similarity) above a 0.45 threshold.
  - *Modification:* Architecturally forbade the LLM from generating deep link URIs; implemented automatic fallback to `voiceassist://dummy_positive` (DL-DUMMY) with self-authored 5–7 word descriptions for unmatched auto/critical actions, and null for manual physical steps.

### Feature 3: Programmatic Sanitizer & Constraint Enforcement (`app/scrubber.py` & `app/constraints.py`)
- **Self-Generated / AI-Generated / Both:** Self-Generated (Assisted by AI for regex design)
- **Description:**
  - *AI Tools/Platform Used:* AI assistance for crafting regex patterns covering bare domains and markdown links.
  - *Prompt Used:* N/A (Deterministic Python code running post-extraction).
  - *Output Summary:* Programmatic regex filter that eliminates bare domains (`samsung.com/support`) and markdown links, clamps titles to 2–3 words, prepends `"It will "` to descriptions, clamps descriptions to 5–7 words, and merges duplicate screens.
  - *Modification:* Recognized that prompt-only constraints fail probabilistically; moved all word-count and link suppression logic into Python application layers.

### Feature 4: Fast-Path Semantic Cache (`app/cache.py`)
- **Self-Generated / AI-Generated / Both:** Both
- **Description:**
  - *AI Tools/Platform Used:* AI assistance for TF-IDF cosine similarity matrix management.
  - *Prompt Used:* N/A (Deterministic algorithmic cache).
  - *Output Summary:* In-memory vector cache matching paraphrase queries (e.g. "battery drains fast" vs "phone won't hold charge") with $\ge 0.55$ cosine similarity.
  - *Modification:* Added domain synonym normalization (`app/text_utils.py`) to align device terms (pair/connect, earbuds/headphones) and refit the vectorizer on demand, yielding sub-15ms responses at $0.00 cost.

### Feature 5: Safe Action Sequencing (`app/matcher.py`)
- **Self-Generated / AI-Generated / Both:** Self-Generated
- **Description:**
  - *AI Tools/Platform Used:* Claude / Antigravity for conceptual discussion.
  - *Prompt Used:* N/A.
  - *Output Summary:* Deterministic comparator that sorts actions: non-destructive `auto` and `manual` actions come first, while destructive `critical` actions (reboot, wipe cache, factory reset) are strictly pushed to the end.
  - *Modification:* Ensures customer device safety regardless of LLM output order.

### Feature 6: Automated Verification & Evaluation Suite (`tests/test_validation.py` & `scripts/run_batch.py`)
- **Self-Generated / AI-Generated / Both:** Both
- **Description:**
  - *AI Tools/Platform Used:* AI assisted in generating comprehensive `unittest.TestCase` test fixtures.
  - *Prompt Used:* Prompted AI to generate exhaustive test coverage for schema parsing, URL scrubbing, constraint clamping, deeplink catalog matching, and semantic caching.
  - *Output Summary:* 16 automated offline unit tests executing in 0.50s, plus a batch evaluation runner that benchmarked all 20 real queries to generate `results.jsonl` and `metrics.md`.
  - *Modification:* Added rate-limit pacing and exact per-token cost accounting formulas.

---

## 5. Ethical & Compliance Confirmation
- **AI usage complies with guidelines and policies:**  
  **Yes**
- **No proprietary or copyrighted data misused:**  
  **I Agree**

---

## 6. Declaration & Sign-Off
- **Name of Team Representative:** Aarush Mudgil
- **Role:** Team Lead / Developer
- **Signature:** Aarush Mudgil
- **Date:** September 29, 2026
