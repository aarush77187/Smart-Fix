"""
Single source of truth for model choice, pricing, and pipeline thresholds.
Nothing in extraction.py / matcher.py / cache.py should hardcode a model
name, a price, or a threshold -- it should import that value from here.
That's what lets you retune the system by editing one file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- LLM provider: Groq (OpenAI-compatible endpoint) ---
    # llama-3.3-70b-versatile was deprecated by Groq for free/developer-tier
    # accounts (shutdown 08/16/26) in favor of openai/gpt-oss-120b -- see
    # https://console.groq.com/docs/deprecations. If you're on an Enterprise
    # committed-spend plan the old model may still work; everyone else needs
    # this value.
    generation_model: str = "openai/gpt-oss-120b"

    # $ per 1K tokens. Verify current pricing at https://groq.com/pricing
    # before quoting these to the jury -- Groq's pricing changes often.
    price_per_1k_input: dict[str, float] = {
        "openai/gpt-oss-120b": 0.00015,
        "openai/gpt-oss-20b": 0.00005,
    }
    price_per_1k_output: dict[str, float] = {
        "openai/gpt-oss-120b": 0.00060,
        "openai/gpt-oss-20b": 0.00020,
    }

    # --- Generation limits ---
    generation_max_tokens: int = 4096  # reasoning tokens count against this
    llm_temperature: float = 0.0  # spec grades deterministic execution
    llm_retry_on_parse_error: int = 1  # retries after the FIRST attempt
    # GPT-OSS models on Groq are reasoning models -- left at their default
    # effort they spend a lot of tokens (and wall-clock time) on internal
    # chain-of-thought before the final JSON, which is most of why measured
    # latency ran well above the spec's own <=8000ms cold-path target.
    # "low" is enough for a structured-extraction task like this one.
    llm_reasoning_effort: str = "low"

    # --- Cache ---
    cache_ttl_seconds: int = 6 * 60 * 60  # 6 hours

    # --- Paths ---
    deeplinks_path: str = "data/deeplinks.json"
    siis_responses_path: str = "data/siis_responses.json"
    request_log_path: str = "logs/requests.jsonl"

    # --- Matching (dual retrieval: BM25 + TF-IDF cosine as a local,
    # no-internet stand-in for dense embeddings -- see matcher.py) ---
    # bm25_scale_cap: empirically-chosen ceiling a raw BM25 score is
    # divided by before capping to 1.0. Calibrated against the real
    # deeplinks.json catalog: a near-verbatim true match scores ~40+ raw,
    # while boilerplate-driven false matches stay under ~12.
    bm25_scale_cap: float = 20.0
    match_score_threshold: float = 0.45  # below this: no_match / dummy_positive fallback
    semantic_cache_threshold: float = 0.55  # TF-IDF cosine similarity floor for a cache hit

    class Config:
        env_prefix = "TSHOOT_"  # e.g. TSHOOT_GENERATION_MODEL=...


settings = Settings()
