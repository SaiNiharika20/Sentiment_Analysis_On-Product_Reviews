# 🧠 Sentiment Analysis on Product Reviews

## 📌 Project Overview
This project implements a **Sentiment Analysis system** to classify product reviews as **Positive, Negative, or Neutral** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

The system is built as a **Flask web application** where users can enter a review and instantly receive sentiment predictions.

---

## 🎯 Objectives
- Analyze customer opinions from product reviews
- Perform text preprocessing and feature extraction
- Train a machine learning model for sentiment classification
- Deploy the model using a user-friendly web interface

---

## 🛠 Tech Stack

### Programming & Frameworks
- Python
- Flask

### Machine Learning & NLP
- Scikit-learn
- TF-IDF Vectorizer
- Pickle (Model Serialization)

### Frontend
- HTML
- CSS

---

## ⚙️ How It Works
1. User enters a product review
2. Text is cleaned and preprocessed
3. Features are extracted using **TF-IDF**
4. Trained ML model predicts sentiment
5. Result displayed as **Positive / Negative / Neutral**

---

## 📂 Project Structure

```text
Sentiment_Analysis_On_Product_Reviews/
├── model/                     # Saved ML models
├── static/                    # CSS and static files
├── templates/                 # HTML templates
├── venv/                      # Virtual environment (ignored in production)
├── app.py                     # Flask application
├── train_model.py             # Model training script
├── check_columns.py           # Dataset validation script
├── reviews.csv                # Sample dataset
├── sentiment_model.pkl        # Trained sentiment model
├── vectorizer.pkl             # TF-IDF vectorizer
├── requirements.txt           # Python dependencies
└── .gitignore                 # Ignored files and folders
