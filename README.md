# 📱 Mobile Product Segmentation & Recommendation System

> **Capstone Project 4 | GUVI Master Data Science Program (IIT Madras Incubated)**

An end-to-end Machine Learning pipeline that segments **47,550 mobile phone reviews** into customer satisfaction clusters using **K-Means Clustering**, and recommends similar phones using **Cosine Similarity** — deployed as a premium dark-themed Streamlit web application called **MobiSense AI**.



## 📌 Problem Statement

The smartphone industry generates large volumes of structured data — product specifications, pricing, customer ratings, and review metadata — that is often unorganised and difficult to analyse for meaningful business insights.

**Two core challenges this project solves:**

1. **Segmentation** — Group 47,550 customer reviews into distinct satisfaction clusters based on structured review features (rating, sub-ratings, price, sentiment, demographics)
2. **Recommendation** — Given a phone a user selects, recommend the most similar mobile phones based on product feature similarity using Cosine Similarity

---

## 🎯 Project Objectives

- Clean and preprocess the Global Mobile Reviews dataset (47,550 records, 16 features)
- Perform EDA to uncover trends across brands, prices, ratings, countries, and sentiments
- Apply K-Means Clustering to segment reviews into customer satisfaction groups
- Analyse each cluster to generate actionable business insights
- Build a Cosine Similarity-based product recommendation engine
- Deploy a 3-tab interactive Streamlit application (MobiSense AI)

---

## 📊 Dataset Overview

| Property | Value |
|----------|-------|
| Total Reviews | **47,550** |
| Mobile Models | **22** |
| Brands | **7** (Apple, Google, Samsung, Xiaomi, Realme, Motorola, OnePlus) |
| Countries | **8** |
| Features | Price (USD), Rating, Battery / Camera / Performance / Design / Display ratings, Sentiment, Verified Purchase, Helpful Votes, Age, Country, Source |

> ⚠️ Raw dataset not included in this repo due to size. Download from the link above and place in the root folder as `Mobile Reviews Sentiment null.csv`

---

## ⚙️ Project Pipeline

### Step 1 — Data Preprocessing

| Issue | Solution |
|-------|----------|
| Missing `source` values | Filled with `'store'` |
| Missing `rating` | Filled using **sentiment-group median** — exploits the strong correlation between sentiment and rating |
| Missing `sentiment` | Derived from rating: ≤2 → Negative, 3 → Neutral, ≥4 → Positive |
| Missing `price_usd` | Rows dropped |
| `helpful_votes` outliers | IQR-based capping at upper bound = 9 |
| `price_local` column | Dropped (irrelevant) |

### Step 2 — Feature Engineering

- Log-transformed `price_usd` → `price_usd_log` to reduce right skew
- Label-encoded categorical features: `sentiment`, `verified_purchase`, `brand`, `model`, `country`
- Standardised all features using `StandardScaler`
- Built `product_data.csv` — 22 models × 14 aggregated features for the recommendation engine

### Step 3 — Exploratory Data Analysis (EDA)

- Review count by brand
- Sentiment distribution (Positive / Neutral / Negative)
- Average price by brand
- Rating distribution across 47,550 reviews
- Average rating by brand
- Reviewer age distribution
- Sub-rating profiles (Battery, Camera, Performance, Design, Display)

### Step 4 — K-Means Clustering

**Features used:** `age`, `price_usd_log`, `rating`, `battery_life_rating`, `camera_rating`, `performance_rating`, `design_rating`, `display_rating`, `helpful_votes`, `sentiment_enc`, `verified_enc`, `brand_enc`, `model_enc`, `country_enc`

**K Selection:** Elbow Method (Inertia) + Silhouette Score

| K | Inertia | Silhouette |
|---|---------|------------|
| 1 | 665,700 | — |
| 2 | 485,202 | **0.2338** ← Best |
| 3 | 451,029 | 0.1337 |
| 4 | 424,945 | 0.1302 |
| 5 | 405,010 | 0.1069 |

**Final Result: K = 2**

| Cluster | Reviews | Avg Price | Avg Rating | Dominant Sentiment | Interpretation |
|---------|---------|-----------|------------|--------------------|----------------|
| **Cluster 0** | 22,017 | $690 | **2.07 ⭐** | Neutral + Negative | Dissatisfied customers |
| **Cluster 1** | 25,533 | $690 | **4.01 ⭐** | Positive (88%) | Satisfied customers |

> 💡 **Key Insight — The Price Paradox:** Both clusters have identical average prices ($690). Price does NOT predict customer satisfaction. Product quality and brand experience are the real differentiators.

**Cluster 0 top brands:** Xiaomi (3,251), Google (3,200), Motorola (3,162)
**Cluster 1 top brands:** Google (3,703), Realme (3,692), Apple (3,692)

Clusters visualised using **3D PCA** (14 dimensions → 3 principal components)

### Step 5 — Cosine Similarity Recommendation Engine

- Aggregated review features into a **22 models × 14 features** product matrix
- Computed **pairwise Cosine Similarity** across all 22 mobile models
- Saved as `similarity_matrix.pkl` (4KB) for instant inference
- User selects a phone → system returns **top-5 most similar phones** ranked by cosine score with similarity bar visualisation

---

## 🖥️ Streamlit App — MobiSense AI

**3-tab dark-themed interactive application:**

### 📊 Tab 1 — EDA & Overview
- Live KPI metrics: Total Reviews, Avg Price, Avg Rating, Verified Purchase %
- 6 interactive Plotly charts covering brand, sentiment, price, rating, and age distributions

### 🗂️ Tab 2 — Cluster Analysis
- Cluster summary cards with interpretation and the Price Paradox insight
- 3D PCA scatter plot (3,000-point sample per cluster, interactive)
- Rating box plots by cluster
- Sentiment stacked bar chart by cluster
- Sub-rating grouped bar (Battery, Camera, Performance, Design, Display)
- Dual-axis Elbow + Silhouette chart with K=2 annotation

### 🔍 Tab 3 — Recommendations
- **Step 1** — Filter by brand, budget (USD range), minimum rating
- **Step 2** — Browse all matching phones with price, rating, review count, positive %, cluster badge
- **Step 3** — Select one phone → view top-5 cosine similarity recommendations with scores
- Feature profile grid (12 metrics) for the selected phone
- Cosine similarity bar chart

---

## 🗂️ Repository Structure

```
Mobile-Product-Segmentation/
│
├── model_files/
│   ├── kmeans_model.pkl           # Trained K-Means model (K=2, 187KB)
│   ├── similarity_matrix.pkl      # 22×22 cosine similarity matrix (4KB)
│   ├── rec_scaler.pkl             # Recommendation scaler (912B)
│   ├── scaler.pkl                 # StandardScaler for clustering features (1.1KB)
│   ├── cluster0.csv               # 22,017 Dissatisfied reviews + cluster label (3.3MB)
│   ├── cluster1.csv               # 25,533 Satisfied reviews + cluster label (3.8MB)
│   ├── mobile_clustered.csv       # Full 47,550 records with cluster assignments (3.4MB)
│   └── product_data.csv           # 22 models × 14 aggregated features (4.6KB)
│
├── app.py                         # MobiSense AI — Streamlit web application (39KB)
├── Project4.ipynb                 # Main notebook: EDA, preprocessing, clustering, recommendation
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.14 |
| Data Processing | Pandas, NumPy |
| ML — Clustering | Scikit-Learn `KMeans` |
| ML — Recommendation | Scikit-Learn `cosine_similarity` |
| Feature Scaling | `StandardScaler` |
| Dimensionality Reduction | `PCA` (3D visualisation only) |
| Visualisation | Plotly (interactive), Matplotlib, Seaborn |
| Web App | Streamlit |
| Model Persistence | Pickle |

---

## 📈 Final Results Summary

| Metric | Value |
|--------|-------|
| Dataset | 47,550 reviews · 22 models · 7 brands · 8 countries |
| Preprocessing | Null handling, outlier capping, log transform, encoding, scaling |
| Clustering Algorithm | K-Means |
| Optimal K | **2** |
| Silhouette Score | **0.2338** |
| Cluster 0 | 22,017 reviews · Avg 2.07⭐ · Dissatisfied |
| Cluster 1 | 25,533 reviews · Avg 4.01⭐ · Satisfied |
| Recommendation Engine | Cosine Similarity on 22×14 product feature matrix |
| Deployment | Streamlit (MobiSense AI) — 3-tab interactive app |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/saroonevan/Mobile-Product-Segmentation.git
cd Mobile-Product-Segmentation
```

### 2. Create and activate virtual environment
```bash
python -m venv mobile_env

# Windows
mobile_env\Scripts\activate

# Mac / Linux
source mobile_env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add dataset
Download `Mobile Reviews Sentiment null.csv` from the [Dataset Link](https://drive.google.com/file/d/12B5UDsw35QaMhqkoHIl1kgE8hKMxGpwm/view?usp=sharing) and place it in the root project folder.

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📦 requirements.txt

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
streamlit
```

---

## 🔗 Full Portfolio

| # | Project | Tech Stack | Link |
|---|---------|-----------|------|
| 1 | Sales Intelligence Hub | Python · MySQL · Streamlit · Plotly | [GitHub](https://github.com/saroonevan/Sales-Intelligence-Hub) |
| 2 | International Debt Analysis | Python · MySQL · Power BI (1.46M rows) | [GitHub](https://github.com/saroonevan/International-debt-analysis) |
| 3 | Marketing Campaign Predictor | Scikit-Learn · Streamlit · R²=99.2% | Coming Soon |
| 4 | Mobile Product Segmentation | K-Means · Cosine Similarity · Streamlit | This Repo |

---

## 👨‍💻 Author

**Saroon Narayanan** — Data Scientist | ML Engineer

- GUVI Master Data Science Program — IIT Madras Incubated (2026)
- 7+ years in US Healthcare Revenue Cycle Management

📧 saroon.akon@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/saroonnarayanan)
🐙 [GitHub](https://github.com/saroonevan)
📍 Chennai, Tamil Nadu | Open to Remote & Relocation (India · USA · UAE · UK)

---

## 🙏 Acknowledgement

Special thanks to my mentor **Vignesh P** — for guidance, feedback, and support throughout the GUVI Master Data Science Program.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
