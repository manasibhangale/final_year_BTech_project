import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from utils.config import EMBEDDING_MODEL, FAISS_PATH, TEXT_STORE

model = SentenceTransformer(EMBEDDING_MODEL)


def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return np.array(embeddings).astype("float32")


def save_index(embeddings, texts):
    os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, FAISS_PATH)

    with open(TEXT_STORE, "wb") as f:
        pickle.dump(texts, f)


def load_index():
    if not os.path.exists(FAISS_PATH) or not os.path.exists(TEXT_STORE):
        return None, None

    index = faiss.read_index(FAISS_PATH)

    with open(TEXT_STORE, "rb") as f:
        texts = pickle.load(f)

    return index, texts


def search(query, index, texts, k=4):
    q_emb = model.encode([query]).astype("float32")

    distances, indices = index.search(q_emb, k)

    return [texts[i] for i in indices[0] if i < len(texts)]