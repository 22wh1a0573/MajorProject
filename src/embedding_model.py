import os
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_PATH = "embeddings/sbert_embeddings.npy"

def build_product_text(df):
    return (
        df["product_name"].fillna("") + " " +
        df["brand"].fillna("") + " " +
        df["product_category_tree"].fillna("")
    )

def get_embeddings(df):
    if os.path.exists(EMBEDDING_PATH):
        print("Loading cached embeddings...")
        return np.load(EMBEDDING_PATH)

    print("Generating SBERT embeddings...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = build_product_text(df).tolist()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    os.makedirs("embeddings", exist_ok=True)
    np.save(EMBEDDING_PATH, embeddings)

    return embeddings
