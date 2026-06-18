# 📺 TV Streaming Sentiment NLP Engine
### Netflix · Disney+ · HBO Max · Amazon Prime · Apple TV+ · Peacock | 2020–2026

> ⚠️ **ACADEMIC RESEARCH DISCLAIMER:** This project is for educational and portfolio purposes only. Sentiment data is synthetically generated using real-world industry event timelines. No content constitutes financial or investment advice. The author is not liable for any investment decisions made based on this codebase.

---

## 📌 Project Overview

This NLP pipeline analyzes public sentiment toward major TV streaming platforms across 9,918 daily records spanning 2020–2026. Using a **FinBERT-calibrated sentiment framework**, it computes Net Sentiment Momentum Scores (Sₜ), detects shock events, models topic themes with LDA, and builds a content recommendation engine using TF-IDF cosine similarity.

### Sₜ Formula
```
Sₜ = (Count_Positive - Count_Negative) / (Count_Positive + Count_Negative + 1)
```

---

## 📊 Key Visualizations

### Chart 1 — Sentiment Dashboard
![Sentiment Dashboard](outputs/chart1_sentiment_dashboard.png)

### Chart 2 — NLP Topic & Similarity Analysis
![NLP Analysis](outputs/chart2_nlp_analysis.png)

---

## 📈 Key Findings

| Platform | Avg Sₜ | Positive Days | Negative Days | Regime |
|----------|--------|--------------|--------------|--------|
| Apple TV+ | **0.476** | 1,653 | 0 | Strongest bullish |
| Amazon Prime | 0.440 | 1,648 | 0 | Strong bullish |
| Netflix | 0.419 | 1,613 | 1 | Bullish |
| Disney+ | 0.377 | 1,467 | 0 | Moderate bullish |
| HBO Max | 0.348 | 1,226 | 1 | Mixed |
| Peacock | 0.310 | 904 | 0 | Weakest |

### Shock Events Detected
- 🔴 Netflix Q1 2022: subscriber loss (Sₜ = –0.028)
- 🔴 HBO Max Aug 2022: Batgirl cancellation (Sₜ = –0.095)
- 🟢 Netflix Nov 2023: password enforcement boosts subs (Sₜ = +0.917)
- 🟢 Disney+ Sep 2021: Shang-Chi streaming release (Sₜ = +0.656)

### LDA Topic Themes Discovered
1. Growth & Earnings (subscribers, revenue, quarterly)
2. Content Performance (original, hit, award, viewership)
3. Market Competition (churn, cancellation, competition)
4. Pricing & Strategy (price, bundle, ad-supported)
5. Tech & Infrastructure (platform, streaming, quality)

---

## 🔬 Methodology

**Step 1 — Data Acquisition:** Daily headline counts per platform with FinBERT-calibrated positive/negative/neutral classifications  
**Step 2 — Preprocessing:** Tokenization, URL removal, 512-token BERT limit handling  
**Step 3 — Sentiment Quantification:** Sₜ score computation, rolling averages, lagged features  
**Step 4 — Topic Modeling:** LDA (5 topics) on TF-IDF vectorized headline corpus  
**Step 5 — Content Similarity:** TF-IDF cosine similarity on show plot synopses for recommendations

**Citations:**
> [1] ProsusAI/FinBERT — https://huggingface.co/ProsusAI/finbert  
> [2] Reddit/PRAW API — r/streaming, r/cordcutters, r/Netflix  
> [3] NewsAPI — streaming industry headline aggregation  
> [4] Counterpoint Research Streaming Survey 2025

---

## 🚀 Setup

```bash
git clone https://github.com/YOUR_USERNAME/streaming-sentiment-nlp.git
cd streaming-sentiment-nlp
pip install -r requirements.txt
python src/01_nlp_pipeline.py
python src/02_visualizations.py
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C)

*Part of a multi-domain data science portfolio — companion: [Streaming Financial Analytics](https://github.com/Tommy-bit02/streaming-financial-analytics)*
