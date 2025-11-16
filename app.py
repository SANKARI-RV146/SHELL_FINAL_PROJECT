# app.py — Streamlit EV Smart Vehicle Assistant (OpenAI embeddings for query-time)
import os
import pickle
import pandas as pd
import numpy as np
import faiss
import openai
import streamlit as st
from datetime import datetime

# ---------- Configuration ----------
CSV_FNAME = "electric_vehicles_spec_2025.csv"
INDEX_FNAME = "ev_faiss.index"
META_FNAME = "ev_meta.pkl"
OPENAI_EMBED_MODEL = "text-embedding-3-small"  # change if needed

# Read OpenAI key from environment (Streamlit Secrets set this as an env var)
openai.api_key = os.environ.get("OPENAI_API_KEY")

# ---------- Helpers ----------
@st.cache_data
def load_ev_dataset():
    if not os.path.exists(CSV_FNAME):
        return pd.DataFrame({"_error": [f"file not found: {CSV_FNAME}"]})
    try:
        return pd.read_csv(CSV_FNAME)
    except Exception as e:
        return pd.DataFrame({"_error": [f"error reading {CSV_FNAME}: {e}"]})

@st.cache_resource
def load_faiss_and_meta(index_path=INDEX_FNAME, meta_path=META_FNAME):
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, None
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return index, meta

def embed_query_openai(query: str, model_name: str = OPENAI_EMBED_MODEL):
    if openai.api_key is None:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to environment or Streamlit Secrets.")
    resp = openai.Embedding.create(model=model_name, input=[query])
    emb = np.array(resp["data"][0]["embedding"], dtype="float32").reshape(1, -1)
    # normalize if index stored normalized vectors
    try:
        faiss.normalize_L2(emb)
    except Exception:
        pass
    return emb

def semantic_search(query: str, k: int = 5):
    index, meta = load_faiss_and_meta()
    if index is None or meta is None:
        return []
    q_emb = embed_query_openai(query)
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(meta):
            continue
        m = meta[idx].copy()
        m["_score"] = float(score)
        results.append(m)
    return results

def get_model_response(user_message: str) -> str:
    lowm = user_message.lower()
    # quick canned fallbacks
    if "charging" in lowm or "charge" in lowm:
        # still attempt semantic retrieval below but keep fallback if none found
        pass

    results = []
    try:
        results = semantic_search(user_message, k=3)
    except Exception as e:
        # if OpenAI key missing or other error, fall back
        return f"(Error performing semantic search: {e})"

    if not results:
        if "range" in lowm:
            return "Estimated remaining range: ~180 km (placeholder)."
        return "I couldn't find matching EV specs locally."

    lines = ["I found these relevant EV models and key specs:"]
    for r in results:
        brand = r.get("brand", "") or r.get("Brand", "")
        model = r.get("model", "") or r.get("Model", "")
        range_km = r.get("range_km", "") or r.get("range", "")
        batt = r.get("battery_capacity_kWh", "") or r.get("battery_capacity", "")
        score = r.get("_score", None)
        line = f"- {brand} {model} — range: {range_km} km; battery: {batt} kWh"
        if score is not None:
            line += f"  (score: {score:.3f})"
        lines.append(line)
    lines.append("\nAsk me to show full specs for any model above.")
    return "\n".join(lines)

# ---------- Load dataset & show debug info ----------
ev_df = load_ev_dataset()

st.set_page_config(page_title="EV Smart Vehicle Assistant", layout="wide")
st.title("⚡ EV Smart Vehicle Assistant")
st.markdown("A prototype Streamlit app — UI ready for chat and model integration.")

with st.sidebar:
    st.markdown("---")
    st.header("EV Dataset")
    if "_error" in ev_df.columns:
        st.error(ev_df["_error"].iloc[0])
    else:
        st.write(f"Loaded EV models: {len(ev_df)}")
        if st.checkbox("Show EV dataset"):
            st.dataframe(ev_df, height=300)

    st.markdown("---")
    st.header("Vehicle summary")
    st.write("Model: **EV Prototype 2025**")
    st.write("Battery: **75 kWh**")
    st.write("Last updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.markdown("---")
    st.header("Index status (debug)")
    idx, meta = load_faiss_and_meta()
    if idx is None:
        st.error("FAISS index / metadata not found. Make sure ev_faiss.index and ev_meta.pkl exist.")
    else:
        st.success(f"FAISS loaded: {idx.ntotal} vectors")
        if st.checkbox("Show top 5 metadata items (debug)"):
            st.dataframe(pd.DataFrame(meta[:5]))

# ---------- Chat UI ----------
st.subheader("Assistant Chat")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for entry in st.session_state.chat_history:
    role = entry["role"]
    msg = entry["message"]
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Assistant:** {msg}")

# input box
user_input = st.text_input("Type a question or command for the EV assistant", key="input_box")
send_clicked = st.button("Send")
if send_clicked or (user_input and st.session_state.get("last_input") != user_input):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        try:
            reply = get_model_response(user_input)
        except Exception as e:
            reply = f"(Error generating response: {e})"
        st.session_state.chat_history.append({"role": "assistant", "message": reply})
        st.session_state.last_input = user_input
        st.rerun()

st.markdown("---")
st.caption("Next: add model explainability, comparisons, voice or images.")
