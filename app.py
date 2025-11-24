from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import torch
from sentence_transformers import  util
from sentence_transformers_lite import SentenceTransformerLite

app = FastAPI()

# Load catalog
df = pd.read_csv("../data/shl_catalog_full_details.csv")

# Load model
model = SentenceTransformerLite("all-MiniLM-L6-v2")

# Load embeddings
embeddings = torch.load("./embeddings/catalog_embeddings.pt")
embeddings = embeddings.cpu()

class Query(BaseModel):
    query: str
    top_k: int = 5

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend")
def recommend(q: Query):
    query = q.query
    top_k = q.top_k

    q_emb = model.encode([query], convert_to_tensor=True)
    scores = util.cos_sim(q_emb, embeddings)[0].cpu().numpy()

    df["score"] = scores
    top = df.sort_values("score", ascending=False).head(top_k)

    results = []
    for _, row in top.iterrows():
        results.append({
            "name": row["name"],
            "url": row["url"],
            "score": float(row["score"])
        })

    return {"query": query, "results": results}
