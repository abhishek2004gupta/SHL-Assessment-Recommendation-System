from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
import os

HF_API_KEY = os.getenv("HF_API_KEY")

HF_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def get_embedding(text):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(HF_URL, headers=headers, json={"inputs": text})
    return np.array(response.json())

app = FastAPI()

df = pd.read_csv("./data/shl_catalog_full_details.csv")
catalog_embeddings = np.load("./embeddings/catalog_embeddings.npy")

class Query(BaseModel):
    query: str
    top_k: int = 5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(q: Query):
    q_emb = get_embedding(q.query)

    sims = np.dot(catalog_embeddings, q_emb) / (
        np.linalg.norm(catalog_embeddings, axis=1) * np.linalg.norm(q_emb)
    )

    df["score"] = sims
    top = df.sort_values("score", ascending=False).head(q.top_k)

    return {
        "query": q.query,
        "results": top[["name", "url", "score"]].to_dict(orient="records")
    }
