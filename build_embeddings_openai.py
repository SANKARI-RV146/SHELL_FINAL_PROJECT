# build_embeddings.py — OpenAI v1.0+ compatible
import os
import pandas as pd
import pickle
import faiss
import numpy as np
import time
from openai import OpenAI

# ---------- Configuration ----------
CSV_FNAME = "electric_vehicles_spec_2025.csv"
INDEX_FNAME = "ev_faiss.index"
META_FNAME = "ev_meta.pkl"
EMB_MODEL = "text-embedding-3-small"

# ---------- OpenAI client ----------
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
if client.api_key is None:
    raise RuntimeError("OPENAI_API_KEY not set in environment.")

# ---------- Helpers ----------
def make_text_for_row(row):
    parts = []
    if pd.notna(row.get("brand")):
        parts.append(str(row["brand"]))
    if pd.notna(row.get("model")):
        parts.append(str(row["model"]))
    spec_cols = [
        "battery_capacity_kWh", "range_km", "top_speed_kmh",
        "acceleration_0_100_s", "efficiency_wh_per_km", "torque_nm",
        "drivetrain", "car_body_type", "seats"
    ]
    specs = []
    for c in spec_cols:
        if c in row and pd.notna(row[c]):
            specs.append(f"{c.replace('_',' ')}: {row[c]}")
    if specs:
        parts.append(" | ".join(specs))
    if "source_url" in row and pd.notna(row["source_url"]):
        parts.append(f"source: {row['source_url']}")
    return " — ".join(parts)

def embed_batch_openai(texts):
    all_embs = []
    B = 50
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=EMB_MODEL, input=batch)
        embs = [item.embedding for item in resp.data]
        all_embs.extend(embs)
        time.sleep(0.1)
    arr = np.array(all_embs, dtype="float32")
    return arr

def build_embeddings():
    if not os.path.exists(CSV_FNAME):
        raise FileNotFoundError(CSV_FNAME + " not found")
    df = pd.read_csv(CSV_FNAME)
    texts = []
    meta = []
    for i, row in df.iterrows():
        t = make_text_for_row(row)
        texts.append(t)
        meta.append({
            "row_index": int(i),
            "brand": row.get("brand", ""),
            "model": row.get("model", ""),
            "range_km": row.get("range_km", ""),
            "battery_capacity_kWh": row.get("battery_capacity_kWh", ""),
            "text": t
        })

    print("Computing embeddings via OpenAI for", len(texts), "records...")
    embeddings = embed_batch_openai(texts)
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FNAME)
    with open(META_FNAME, "wb") as f:
        pickle.dump(meta, f)
    print("Saved", INDEX_FNAME, "and", META_FNAME)

if __name__ == "__main__":
    build_embeddings()
