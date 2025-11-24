# app.py (root)
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# load catalog CSV (path relative to repo root)
df = pd.read_csv("./data/shl_catalog_full_details.csv")

# load precomputed embeddings (npy)
catalog_embeddings = np.load("./embeddings/catalog_embeddings.npy")  # <-- ensure this file exists in repo

class Query(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def home():
    return {"message":"SHL API running. Use /health and /recommend"}

@app.get("/health")
def health():
    return {"status":"healthy"}

# optional: use HF API or your local method to get query embedding
def get_query_embedding_via_hf(text, hf_token):
    import requests
    url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {hf_token}"}
    r = requests.post(url, headers=headers, json={"inputs": text})
    r.raise_for_status()
    return np.array(r.json())

@app.post("/recommend")
def recommend(q: Query):
    # Option A: use HF API key from env to get query embedding
    HF_API_KEY = os.getenv("HF_API_KEY", "")
    if HF_API_KEY == "":
        return {"error":"HF_API_KEY is not set on the server. Set it in Railway variables or use local mode."}

    q_emb = get_query_embedding_via_hf(q.query, HF_API_KEY)  # shape (dim,)
    # if HF returns nested list, ensure shape is (dim,)
    if isinstance(q_emb, list):
        q_emb = np.array(q_emb[0]) if isinstance(q_emb[0], list) else np.array(q_emb)

    sims = np.dot(catalog_embeddings, q_emb) / (np.linalg.norm(catalog_embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-12)
    df["score"] = sims
    top = df.sort_values("score", ascending=False).head(q.top_k)
    return {"query": q.query, "results": top[["name","url","score"]].to_dict(orient="records")}
