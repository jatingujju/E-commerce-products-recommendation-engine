# 🚀 Nexus Commerce Intelligence Platform

Advanced E-Commerce Recommendation Engine built using Machine Learning, Information Retrieval, Learning-to-Rank, FastAPI, and Modern Recommendation System Architecture.

---

## 📌 Overview

This project implements an end-to-end recommendation pipeline inspired by real-world systems used by Amazon, Flipkart, Netflix, and e-commerce platforms.

The system generates candidate products using multiple retrieval techniques and ranks them using a LightGBM LambdaMART model.

---

## ✨ Features

### Candidate Generation
- TF-IDF Content-Based Retrieval
- Item-to-Item Co-occurrence Retrieval
- Cold Start Recommendation Handling
- Candidate Fusion Pipeline

### Feature Engineering
- User Feature Store
- Item Feature Store
- Popularity Features
- Interaction-Based Features

### Ranking Engine
- LightGBM LambdaMART Ranker
- Learning-to-Rank Pipeline
- NDCG Optimization
- Top-K Ranking

### Serving Layer
- FastAPI REST APIs
- Swagger Documentation
- Recommendation Endpoint
- Product Metadata Endpoint

### Additional Components
- Product Similarity Graph
- Heap-Based Top-K Retrieval
- Recommendation Dashboard
- Model Inference Pipeline

---

## 🏗️ System Architecture

```text
User Request
      │
      ▼
Candidate Generation
 ├── TF-IDF Retrieval
 ├── Co-Occurrence Retrieval
 └── Cold Start Retrieval
      │
      ▼
Feature Engineering
 ├── User Features
 └── Item Features
      │
      ▼
LambdaMART Ranker
      │
      ▼
FastAPI Serving Layer
      │
      ▼
Recommendations
```

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-Learn
- LightGBM
- FastAPI
- NetworkX
- Streamlit
- NumPy

---

## 📂 Project Structure

```text
E-Commerce-Recommendation-Engine
│
├── data/
├── frontend/
├── screenshots/
│
├── app.py
├── candidate_generator.py
├── ranking_engine.py
├── train_ranker.py
├── predict_rank.py
├── feature_store.py
├── graph_engine.py
│
├── rank_train.parquet
├── user_features.csv
├── item_features.csv
│
└── README.md
```

---

## 🌐 API Endpoints

### Home

```http
GET /
```

### Health Check

```http
GET /health
```

### Products

```http
GET /products
```

### Recommendations

```http
GET /recommend?product_name=iPhone%2015&k=5
```

### Product Details

```http
GET /product?product_name=iPhone%2015
```

### Swagger UI

```http
/docs
```

---

## 📸 Screenshots

### FastAPI Swagger UI

Add screenshot here

### Recommendation Results

Add screenshot here

### Product Similarity Graph

Add screenshot here

### Dashboard

Add screenshot here

---

## 🧠 Algorithms Used

- TF-IDF Vectorization
- Cosine Similarity
- Co-Occurrence Modeling
- Heap-Based Top-K Retrieval
- Product Similarity Graph
- LambdaMART Ranking
- Learning-to-Rank

---

## 📈 Future Improvements

- FAISS ANN Search
- Personalized User Recommendations
- Real-Time Feature Store
- Online Learning
- Hybrid Deep Learning Recommender
- A/B Testing Framework

---

## 👨‍💻 Author

**Jatin Gujarathi**

Mechanical Engineer | Python Developer | Machine Learning Enthusiast

GitHub: https://github.com/jatingujju

---
