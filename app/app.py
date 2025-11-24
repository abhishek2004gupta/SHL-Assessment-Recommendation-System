import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import gradio as gr

# Load catalog
df = pd.read_csv("../data/shl_catalog_full_details.csv")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load embeddings
catalog_embeddings = torch.load("../embeddings/catalog_embeddings.pt")
catalog_embeddings = catalog_embeddings.to(device)

# Recommendation function
def recommend(query, top_k):
    if not query.strip():
        return pd.DataFrame({"Error": ["Please type a query"]})

    q_emb = model.encode([query], convert_to_tensor=True).to(device)
    scores = util.cos_sim(q_emb, catalog_embeddings)[0].cpu().numpy()

    df["score"] = scores
    top = df.sort_values("score", ascending=False).head(top_k)[["name", "url", "score"]]

    return top

# Gradio UI
iface = gr.Interface(
    fn=recommend,
    inputs=[
        gr.Textbox(label="Enter job description or requirement"),
        gr.Number(label="Number of suggestions (Top-K)", value=5),
    ],
    outputs=gr.DataFrame(label="Recommended Assessments"),
    title="SHL Assessment Recommendation Engine",
    description="Enter a job-related query to get the best matching SHL assessments."
)

if __name__ == "__main__":
    iface.launch()
