# SHL Assessment Recommendation Engine  
*A lightweight Retrieval-Augmented recommendation system built using SHL’s Product Catalog*

---

## 📌 Overview  
This project recommends the most relevant SHL Assessments based on a user’s job description or hiring requirement.  
The system uses:

- Scraped SHL product catalog (mandatory per assignment)
- SentenceTransformer embeddings
- Cosine similarity for ranking
- A simple and clean Gradio UI

The goal is to quickly map a hiring query → suitable SHL assessments.

---

## 🛠️ Tech Stack  
- **Python**
- **BeautifulSoup4** (Web scraping)
- **SentenceTransformer (all-MiniLM-L6-v2)** (Embeddings)
- **Torch**
- **Pandas / NumPy**
- **Gradio** (Frontend UI)

---

## 📂 Project Structure  

```
project/
│
├── app/
│   └── app.py                         # Main Gradio app
│
├── data/
│   ├── Gen_AI Dataset.xlsx            # Provided SHL dataset
│   ├── shl_catalog_full_details.csv   # Scraped catalog (408 items)
│   └── shl_individual_tests_catalog.csv
│
├── embeddings/
│   └── catalog_embeddings.pt          # Model-generated embeddings
│
├── notebooks/
│   └── notebook.ipynb                 # Development & experiments
│
├── scraper/
│   └── scrape_shl_catalog.py          # Web scraper for SHL catalog
│
├── requirements.txt
└── README.md
```

---

## 🧹 Step 1 — Data Ingestion (Scraping)

SHL does not offer a public API, so the catalog is scraped **directly from the official SHL website** using BeautifulSoup.  
This satisfies the mandatory requirement:

> “Solutions built without scraping and storing SHL product catalog from the website will be rejected.”

The scraper navigates through all pagination pages for:
- **Pre-packaged Job Solutions**  
- **Individual Test Solutions**

Total collected items: **408**

Output file:  
`data/shl_catalog_full_details.csv`

---

## 🔍 Step 2 — Embedding Generation  

We encode each SHL product using the model:

**Model Used:** `all-MiniLM-L6-v2`  
(Chosen because it is small, fast, and stable for cosine similarity)

Each product name → embedding vector of shape:

```
[408, 384]
```

Saved to:  
`embeddings/catalog_embeddings.pt`

---

## 🤖 Step 3 — Query → Recommendations

Whenever the user enters a job description:

1. Query is converted to embedding  
2. Cosine similarity is computed against all catalog embeddings  
3. Top-K matching assessments are returned  

Ranking criteria: **Higher cosine score = higher similarity**

---

## 🖥️ Step 4 — Gradio Web App

A simple UI asks:

1. **Job description / requirement**
2. **Number of suggestions (Top-K)**

Output:  
A clean table showing:

| Assessment Name | URL | Score |

Run the app using:

```
python app/app.py
```

---

## 🚀 How to Run Locally

### 1. Install dependencies  
```
pip install -r requirements.txt
```

### 2. Run the scraper (optional)  
```
python scraper/scrape_shl_catalog.py
```

### 3. Regenerate embeddings (optional)  
Run the notebook once, or use:

```
model.encode(...)
torch.save(...)
```

### 4. Start the application  
```
python app/app.py
```

---

## 🧪 Testing

- All results are deterministic  
- File paths kept relative for easy SHL automated testing  
- Links remain publicly accessible  
- No external API dependencies → fully offline capable

---

## 📄 Notes

- The project meets all SHL requirements:
  ✓ Scraped data  
  ✓ Structured catalog  
  ✓ Embedding-based retrieval  
  ✓ Working recommendation engine  
  ✓ Clean UI  
  ✓ Reproducible pipeline  

- Codebase is intentionally lightweight to ensure fast runtime.

---

## 👤 Author  
Built as part of the **SHL AI Internship Assignment**, with focus on clarity, correctness, and practical retrieval performance.

