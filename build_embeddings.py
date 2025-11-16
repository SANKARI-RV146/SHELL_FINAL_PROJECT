import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle

print("DEBUG: Script started")

def build_embeddings():
    print("DEBUG: Loading dataset...")

    # Load dataset
    df = pd.read_csv("electric_vehicles_spec_2025.csv")
    print("DEBUG: Dataset loaded:", df.shape)

    # Convert rows to texts
    texts = []
    metadata = []

    print("DEBUG: Converting rows to text lines...")
    for _, row in df.iterrows():
        item = ", ".join([f"{col}: {row[col]}" for col in df.columns])
        texts.append(item)
        metadata.append(dict(row))

    print(f"DEBUG: Total text rows = {len(texts)}")

    # Load embedding model
    print("DEBUG: Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("DEBUG: Model loaded.")

    # Encode texts
    print("DEBUG: Encoding started...")
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True)
    print("DEBUG: Encoding completed. Shape:", embeddings.shape)

    # Build FAISS index
    print("DEBUG: Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)

    index.add(embeddings)
    print("DEBUG: Index built. Number of items:", index.ntotal)

    # Save index
    faiss.write_index(index, "ev_faiss.index")
    print("DEBUG: Saved ev_faiss.index")

    # Save metadata
    with open("ev_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print("DEBUG: Saved ev_meta.pkl")

    print("DEBUG: COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    build_embeddings()
