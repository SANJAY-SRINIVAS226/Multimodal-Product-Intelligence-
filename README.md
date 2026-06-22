# 🛍️ Multimodal Product Intelligence System
### Vision-Language AI Pipeline for E-Commerce Catalog Automation

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![CLIP](https://img.shields.io/badge/Model-CLIP%20ViT--B%2F32-green)
![BLIP-2](https://img.shields.io/badge/Model-BLIP--2-orange)
![FAISS](https://img.shields.io/badge/Search-FAISS-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This project builds an end-to-end AI-powered Product Intelligence System applied to a 
real-world fashion e-commerce dataset. Using foundation models — CLIP and BLIP-2 — the 
system understands products visually, generates metadata automatically, and enables 
intelligent search and recommendations without any task-specific training.

Built as part of the **Gen AI Bootcamp** at **iDThirdeye Technology Solutions**.

---

## 🎯 Problem Statement

Large e-commerce platforms like Amazon and Flipkart face three persistent challenges:

| Challenge | Scale |
|-----------|-------|
| Duplicate product listings from multiple sellers | Millions of redundant catalog entries |
| Poor product recommendations | Users shown visually similar, not complementary items |
| Keyword-dependent search | Fails when users describe products naturally |

This system solves all three using a unified multimodal embedding pipeline.



## 🧠 System Architecture
<img width="1024" height="675" alt="image" src="https://github.com/user-attachments/assets/09996647-06c4-4b6e-b7e4-80133b068442" />



## 🚀 Features

### Task 1 — Smart Product Recommendation Engine
- Assigns product category via `argmax(cosine_similarity)` against prototype embeddings
- Maps category to complements using co-purchase rules
- Retrieves top-K complementary products using FAISS vector search
- Generates product captions using BLIP-2

**Example:**
Input  : Running Shoe

Output : Sports Socks · Fitness Watch · Gym Bag

### Task 2 — Unique Product Catalog Creation
- Computes pairwise cosine similarity matrix across all product embeddings
- Clusters near-duplicates using a similarity threshold of 0.92
- Outputs one representative product per cluster
- Reduces catalog size while preserving full category coverage

**Example:**
Input  : Blue Shirt A · Blue Shirt B · Blue Shirt C · Running Shoe A · Running Shoe B

Output : Blue Shirt · Running Shoe

### Task 3 — Reverse Product Search (Text → Image)
- Encodes natural language queries using CLIP's text encoder
- Performs cross-modal search against image embeddings in FAISS
- Returns top-5 visually matching products with relevance scores
- Zero-shot — no fine-tuning required

**Example:**
Input  : "blue casual shirt"

Output : Men's Blue Casual Shirt · Blue Checked Shirt · Slim Fit Blue Shirt

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Vision-Language Embedding | CLIP ViT-B/32 (OpenAI) |
| Product Captioning | BLIP-2 OPT-2.7B (Salesforce) |
| Vector Search | FAISS (Facebook AI) |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib |
| Runtime | Python 3.10 · CUDA GPU |

---

## ⚙️ Installation

```bash
pip install faiss-cpu transformers accelerate sentencepiece Pillow matplotlib
```

> `torch`, `numpy`, and `pandas` are pre-installed on Kaggle.
> For local setup, also run: `pip install torch pandas numpy`
## ▶️ How to Run

### On Kaggle
```python
# Step 1 — Install dependencies
!pip install faiss-cpu transformers accelerate sentencepiece

# Step 2 — Run the pipeline
exec(open("/kaggle/working/day2_product_intelligence.py").read())
```

### Locally
```bash
# Clone and run
python day2_product_intelligence.py
```

Update dataset paths at the top of the file if running locally:
```python
STYLES_CSV = Path("path/to/styles.csv")
IMAGE_DIR  = Path("path/to/images")
```

---
