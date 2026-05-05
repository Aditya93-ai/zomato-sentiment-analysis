# zomato-sentiment-analysis
End-to-end sentiment analysis on Zomato reviews using NLP and ML to uncover customer satisfaction drivers across delivery, food quality, and pricing.

## 📌 Project Overview

This project performs end-to-end sentiment analysis on 2,000 Zomato 
restaurant reviews across 8 major Indian cities. The goal is to 
automatically classify customer reviews as Positive, Neutral, or 
Negative — and extract what specific aspects (delivery, food quality, 
packaging, price) are driving customer satisfaction or dissatisfaction.

## 🎯 Business Problem

Zomato receives thousands of reviews daily. Manually reading each one 
is impossible. This project builds an automated pipeline that:
- Classifies review sentiment with 96% accuracy
- Identifies which aspects customers complain about most
- Highlights which cities have the lowest satisfaction scores
- Gives the retention/ops team actionable signals to act on

## 🛠️ Tools & Technologies

- Python, Pandas, NumPy
- NLTK, VADER (unsupervised sentiment scoring)
- TF-IDF Vectorization (scikit-learn)
- Logistic Regression, Naive Bayes (classification)
- Matplotlib, Seaborn (visualization)

## 📊 Key Findings

- Logistic Regression achieved 96.2% accuracy on test data
- Supervised ML outperformed VADER (67.1%) significantly
- Food Quality is the most discussed aspect in reviews
- Negative reviews mention delivery time 34% more than positive ones
- Pune has the highest positive sentiment (67.1%), Chennai the lowest

## 📁 Project Structure

├── zomato_reviews.csv              # Dataset (2000 reviews, 9 features)
├── zomato_sentiment_analysis.py    # Main analysis script
├── 01_eda_dashboard.png            # Exploratory data analysis visuals
├── 02_model_performance.png        # Model accuracy & confusion matrix
├── 03_aspect_analysis.png          # Aspect-based sentiment breakdown
└── README.md

## 🚀 How to Run

pip install pandas numpy matplotlib seaborn scikit-learn vaderSentiment
python zomato_sentiment_analysis.py
