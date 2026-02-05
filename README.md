# 🎵 Maven Music — Customer Churn Prediction & Analysis

> Predicting and understanding customer churn in a music streaming platform using machine learning.

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://maven-music-app.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

Maven Music, a music streaming service, faced an unusual increase in customer churn. This project investigates the root causes, builds predictive models, and delivers actionable retention strategies through an interactive dashboard.

**🔗 [Try the live app →](https://maven-music-app.streamlit.app/)**

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| Dataset size | **10,000+ users** |
| Key churn factors identified | **3 major drivers** |
| At-risk customers impacted | **~40%** |
| Models tested | Random Forest, Logistic Regression, Gradient Boosting |

## 🔍 Methodology

1. **Data Collection** — Customer subscription data & listening behavior (10K+ records)
2. **Data Cleaning** — Missing values, duplicates, outlier detection, format standardization
3. **Exploratory Analysis** — Churn distribution by tier, listening frequency, engagement metrics
4. **Feature Engineering** — Subscription duration, listening patterns, engagement scores
5. **Modeling** — Binary classification with multiple algorithms & cross-validation
6. **Deployment** — Interactive Streamlit dashboard for real-time exploration

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Streamlit |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Proftg/Maven-Music.git
cd Maven-Music

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## 📊 Key Findings

- **Subscription tier** is the strongest predictor of churn — free-tier users churn 3x more
- **Listening frequency drop** in the last 30 days is a leading indicator of imminent churn
- **Playlist diversity** correlates inversely with churn — engaged users explore more content

## 📁 Project Structure

```
Maven-Music/
├── data/               # Raw and processed datasets
├── notebooks/          # Jupyter notebooks (EDA, modeling)
├── app.py              # Streamlit dashboard
├── requirements.txt    # Dependencies
└── README.md
```

## 👤 Author

**Tahar Guenfoud** — [LinkedIn](https://www.linkedin.com/in/tahar-guenfoud/) · [GitHub](https://github.com/Proftg)
