# dataset loader with debug
import streamlit as st
import os
import pandas as pd
import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# faiss import attempt
try:
    import faiss
except Exception:
    faiss = None

@st.cache_data
def load_ev_dataset():
    fname = r"C:\Users\RAGHAV\smart_ev_vehicle_assist\electric_vehicles_spec_2025.csv"
    if not os.path.exists(fname):
        # return a small DataFrame with error information so UI shows something
        return pd.DataFrame({"_error": [f"file not found: {fname}"]})
    try:
        df = pd.read_csv(fname)
        return df
    except Exception as e:
        return pd.DataFrame({"_error": [f"error reading {fname}: {str(e)}"]})

ev_df = load_ev_dataset()

# --- FAISS + metadata loader ---
@st.cache_resource
def load_faiss_and_meta(index_path="ev_faiss.index", meta_path="ev_meta.pkl"):
    # returns (index, meta_list, embedding_model or None)
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, None, None

    if faiss is None:
        st.sidebar.error("FAISS not available in the environment.")
        return None, None, None

    # load index
    index = faiss.read_index(index_path)

    # load metadata
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # load the same embedding model used to build the index
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    return index, meta, emb_model
def semantic_search(query: str, k: int = 5):
    """
    Returns top-k metadata entries for the query.
    """
    index, meta, emb_model = load_faiss_and_meta()
    if index is None or meta is None or emb_model is None:
        return []

    # embed query
    q_emb = emb_model.encode([query], convert_to_numpy=True)
    # ensure float32
    q_emb = q_emb.astype("float32")
    # if index was built with normalized vectors for inner product, normalize query
    try:
        faiss.normalize_L2(q_emb)
    except Exception:
        pass

    # search
    D, I = index.search(q_emb, k)  # I shape: (1, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(meta):
            continue
        m = meta[idx].copy()
        # attach score
        m["_score"] = float(score)
        results.append(m)
    return results


# debug info (temporary — remove later)
if "_error" in ev_df.columns:
    st.sidebar.error(ev_df["_error"].iloc[0])
else:
    st.sidebar.success(f"Loaded EV models: {len(ev_df)}")
    # show a tiny sample always so it's obvious
    st.sidebar.dataframe(ev_df.head(5), height=180)

# app.py
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="EV Smart Vehicle Assistant", layout="wide")

def get_model_response(user_message: str) -> str:
    """
    Use semantic search to fetch relevant EV specs and produce a short reply.
    Falls back to canned responses if no index available.
    """
    # quick canned behaviour
    lowm = user_message.lower()
    if "range" in lowm and "?" not in user_message:
        # still allow semantic route below; keep canned for quick hits if desired
        pass

    # try semantic retrieval
    results = semantic_search(user_message, k=3)
    if not results:
        # fallback canned
        if "range" in lowm:
            return "Estimated remaining range: ~180 km (placeholder)."
        if "charge" in lowm or "charging" in lowm:
            return "Charging status: 45% (placeholder)."
        return "I couldn't find matching EV specs locally. (This is fallback response.)"

    # build a helpful reply summarizing top results
    reply_lines = []
    reply_lines.append("I found the following relevant EV models and key specs:")

    for r in results:
        brand = r.get("brand") or r.get("Brand") or ""
        model = r.get("model") or r.get("Model") or ""
        text = r.get("text") or ""
        score = r.get("_score", None)
        # show a compact summary (brand, model, range, battery)
        range_km = r.get("range_km") or r.get("range") or ""
        batt = r.get("battery_capacity_kWh") or r.get("battery_capacity") or ""
        line = f"- **{brand} {model}** — range: {range_km} km; battery: {batt} kWh"
        if score is not None:
            line += f"  (score: {score:.3f})"
        reply_lines.append(line)

    reply_lines.append("\nYou can ask me to show full specs for any listed model.")
    return "\n".join(reply_lines)


# --- Layout ---
st.title("⚡ EV Smart Vehicle Assistant")
st.markdown("A prototype Streamlit app — UI ready for chat and model integration.")

with st.sidebar:
    st.markdown("---")
    st.header("EV Dataset")
    st.write(f"Loaded EV models: {len(ev_df)}")

    if st.checkbox("Show EV dataset"):
        st.dataframe(ev_df, height=300)

    st.markdown("---")
    st.header("Vehicle summary")
    st.write("Model: **EV Prototype 2025**")
    st.write("Battery: **75 kWh**")
    st.write("Last updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.markdown("---")
    st.header("Quick actions")
    if st.button("Simulate low battery"):
        st.success("Simulated: battery level set to 12% (placeholder).")
    if st.button("Simulate charging"):
        st.info("Simulated: charging started at 7 kW (placeholder).")

# --- Chat area ---
st.subheader("Assistant Chat")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# show chat history
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
        # append user message
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        # use the semantic-enabled responder (falls back if index missing)
        reply = get_model_response(user_input)
        st.session_state.chat_history.append({"role": "assistant", "message": reply})
        # remember last input and rerun to update UI
        st.session_state.last_input = user_input
        st.rerun()

       


st.markdown("---")
st.caption("Next: we'll hook this UI to a real embedding store + model and replace the placeholder responses.")
