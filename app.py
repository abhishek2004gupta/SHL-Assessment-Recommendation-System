from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sentence_transformers_lite import SentenceTransformerLite
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load catalog
df = pd.read_csv("./data/shl_catalog_full_details.csv")

# Load lite model (no torch needed)
model = SentenceTransformerLite("all-MiniLM-L6-v2")

# Load embeddings (saved earlier in npy format)
catalog_embeddings = np.load("./embeddings/catalog_embeddings.npy")

class Query(BaseModel):
    query: str
    top_k: int = 5

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/recommend")
def recommend(q: Query):
    q_emb = model.encode([q.query])
    scores = cosine_similarity(q_emb, catalog_embeddings)[0]

    df["score"] = scores
    top = df.sort_values("score", ascending=False).head(q.top_k)

    return {
        "query": q.query,
        "results": [
            {
                "name": row["name"],
                "url": row["url"],
                "score": float(row["score"])
            }
            for _, row in top.iterrows()
        ]
    }
