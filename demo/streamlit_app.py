"""
Demo frontend -- a genuine HTTP client of the FastAPI backend, calling the
real POST /v1/troubleshoot contract. Every result shown here came from a
real API call against the real deeplinks.json catalog.
"""

import json

import requests
import streamlit as st

st.set_page_config(page_title="Smart Guided Troubleshooting Engine", page_icon="🛠️", layout="centered")

with st.sidebar:
    st.subheader("Settings")
    api_base = st.text_input("API base URL", value="http://localhost:8000")
    st.caption("Point this at your running FastAPI server.")

st.title("🛠️ Smart Guided Troubleshooting Engine")
st.caption("Samsung PRISM GenAI Hackathon — Theme 02")

tab_troubleshoot, tab_batch, tab_debug = st.tabs(["Troubleshoot", "Load from siis_responses.json", "Debug info"])

# ---------------------------------------------------------------------------
# Tab 1: manual query + optional siis_response JSON
# ---------------------------------------------------------------------------
with tab_troubleshoot:
    query = st.text_area("Customer complaint", placeholder="e.g. My screen is completely black and won't turn on", height=80)

    st.caption("Optional: paste the siis_response reference text (title + content). "
               "Without it, the engine only checks the semantic cache.")
    siis_title = st.text_input("siis_response title (optional)")
    siis_content = st.text_area("siis_response content (optional)", height=150)

    if st.button("Troubleshoot", type="primary", disabled=not query.strip()):
        payload = {"query": query.strip()}
        if siis_title.strip() and siis_content.strip():
            payload["siis_response"] = {"title": siis_title.strip(), "content": siis_content.strip()}

        try:
            with st.spinner("Running pipeline..."):
                resp = requests.post(f"{api_base}/v1/troubleshoot", json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
            st.stop()

        meta = data["meta"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Latency", f"{meta['latency_ms']:.0f} ms")
        c2.metric("Cache hit", "Yes" if meta["cache_hit"] else "No")
        c3.metric("Cost", f"${meta['cost_usd']:.5f}")

        if meta.get("fallback") == "no_match":
            st.warning("No viable solution found — the engine returned an empty result rather than guessing.")

        contexts = data["response"]["contexts"]
        if not contexts:
            st.info("No troubleshooting plan generated.")
        for goal in contexts:
            st.subheader(goal["title"])
            st.caption(goal["goal"])
            st.caption(f"Confidence score: {goal['score']:.2f}")
            for action in goal["actions"]:
                with st.container(border=True):
                    badge = {"auto": "🟢", "manual": "🟡", "critical": "🔴"}.get(action["category"], "")
                    st.markdown(f"**{badge} {action['actionName']}** — {action['description']}")
                    for sg in action["stepGroups"]:
                        for step in sg["steps"]:
                            st.markdown(f"- {step}")
                        dl = sg.get("actionableDeeplink")
                        if dl:
                            st.code(dl["deeplink"], language=None)
                            st.caption(dl.get("message", ""))
                        vdl = sg.get("validationDeeplink")
                        if vdl:
                            st.caption(f"Validation: {vdl['key']}")

        with st.expander("Raw response JSON"):
            st.json(data)

# ---------------------------------------------------------------------------
# Tab 2: pick one of the real 20 queries from siis_responses.json
# ---------------------------------------------------------------------------
with tab_batch:
    st.caption("Loads a real query + siis_response pair from data/siis_responses.json for a one-click demo.")
    try:
        with open("data/siis_responses.json") as f:
            rows = json.load(f)["responses"]
    except FileNotFoundError:
        rows = []
        st.error("data/siis_responses.json not found relative to the Streamlit working directory.")

    if rows:
        labels = [r["original_query"][:80] for r in rows]
        idx = st.selectbox("Pick a real query", range(len(rows)), format_func=lambda i: labels[i])
        chosen = rows[idx]
        st.text_area("Query", value=chosen["original_query"], height=60, disabled=True)
        st.text_area("siis_response.content (preview)", value=chosen["siis_response"]["content"][:500] + "...", height=150, disabled=True)

        if st.button("Run this query", type="primary"):
            payload = {"query": chosen["original_query"], "siis_response": chosen["siis_response"]}
            try:
                with st.spinner("Running pipeline..."):
                    resp = requests.post(f"{api_base}/v1/troubleshoot", json=payload, timeout=30)
                resp.raise_for_status()
                st.json(resp.json())
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

# ---------------------------------------------------------------------------
# Tab 3: debug endpoints
# ---------------------------------------------------------------------------
with tab_debug:
    try:
        stats = requests.get(f"{api_base}/debug/catalog-stats", timeout=10).json()
        st.metric("Indexed catalog entries", stats["indexed_entries"])
    except requests.RequestException as e:
        st.error(f"Could not reach {api_base}/debug/catalog-stats: {e}")

    try:
        cstats = requests.get(f"{api_base}/debug/cache-stats", timeout=10).json()
        st.metric("Semantic cache entries", cstats["entries"])
    except requests.RequestException as e:
        st.error(f"Could not reach {api_base}/debug/cache-stats: {e}")
