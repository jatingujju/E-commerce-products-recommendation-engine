# 🚀 Nexus Commerce Intelligence Platform

Advanced E-Commerce Recommendation Engine built using Machine Learning, Information Retrieval, Learning-to-Rank, FastAPI, and Modern Recommendation System Architecture.

---

## 📌 Overview

This project implements an end-to-end recommendation pipeline inspired by real-world systems used by Amazon, Flipkart, Netflix, and modern e-commerce platforms.

The system generates candidate products using multiple retrieval techniques and ranks them using a LightGBM LambdaMART model to deliver relevant recommendations.

---

## ✨ Key Features

### Candidate Generation

* TF-IDF Content-Based Retrieval
* Item-to-Item Co-Occurrence Retrieval
* Cold Start Recommendation Handling
* Multi-Source Candidate Fusion

### Feature Engineering

* User Feature Store
* Item Feature Store
* Popularity Features
* Interaction-Based Features

### Ranking Engine

* LightGBM LambdaMART Ranker
* Learning-to-Rank Pipeline
* NDCG Optimization
* Top-K Ranking

### Serving Layer

* FastAPI REST APIs
* Interactive Swagger Documentation
* Recommendation Endpoint
* Product Metadata Endpoint

### Additional Components

* Product Similarity Graph
* Heap-Based Top-K Retrieval
* Model Inference Pipeline
* Recommendation Dashboard

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

* Python
* Pandas
* NumPy
* Scikit-Learn
* LightGBM
* FastAPI
* NetworkX
* Streamlit

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

### Swagger Documentation

```http
GET /docs
```

---

## 📸 Screenshots

### FastAPI Swagger UI

<img src="screenshots/swagger_api.PNG" width="900"/>

### Recommendation API Response

<img src="screenshots/api_response.PNG" width="900"/>

### Product Similarity Graph

<img src="screenshots/product_similarity_graph.png" width="900"/>

---

## 🧠 Algorithms Used

* TF-IDF Vectorization
* Cosine Similarity
* Co-Occurrence Modeling
* Heap-Based Top-K Retrieval
* Product Similarity Graph
* LambdaMART Ranking
* Learning-to-Rank

---

## 📈 Future Improvements

* FAISS ANN Search
* Personalized User Recommendations
* Real-Time Feature Store
* Online Learning
* Hybrid Deep Learning Recommender
* A/B Testing Framework
* Real-Time User Behavior Tracking

---

## 👨‍💻 Author

**Jatin Gujarathi**

Mechanical Engineer | Python Developer | Machine Learning Enthusiast

GitHub: https://github.com/jatingujju

---

## ⭐ Project Highlights

* End-to-End Recommendation System
* Learning-to-Rank using LightGBM LambdaMART
* FastAPI Production-Ready APIs
* TF-IDF + Co-Occurrence Retrieval
* User & Item Feature Engineering
* Graph-Based Product Relationships
* Extensible Architecture for Real-World Applications
