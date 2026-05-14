# Project Summary — Sentiment Analysis of Product Reviews

## Overview

This project builds an end-to-end NLP pipeline to classify Amazon product reviews into **Positive**, **Neutral**, and **Negative** sentiments. It compares traditional machine learning approaches with state-of-the-art transformer models to identify the best solution for large-scale review classification.

---

## Team

| Name | Reg No |
| --- | --- |
| Ardra Selin A G | XXXXXX |
| Sravana Nambiar | XXXXXX |
| Archana | XXXXXX |

---

## Dataset

- **Name:** Amazon Fine Food Reviews
- **Source:** Kaggle
- **Size:** 568,454 reviews
- **Link:** https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
- **Sentiment Labels:** Positive (4–5★), Neutral (3★), Negative (1–2★)

---

## Models Built

| Model | Type | Status |
| --- | --- | --- |
| TF-IDF + SVM | Traditional ML | ✅ Complete |
| TF-IDF + Logistic Regression | Traditional ML | ✅ Complete |
| DistilBERT | Transformer (Deep Learning) | ✅ Complete |
| BERT | Transformer (Deep Learning) | ✅ Complete |

---

## Key Steps

1. **Data Preprocessing** — Cleaning, stopword removal, lemmatization, label mapping
2. **EDA** — Class distribution, word clouds, review length analysis
3. **Feature Engineering** — TF-IDF vectorization, BERT tokenization
4. **Model Training** — Traditional ML + fine-tuned transformers
5. **Evaluation** — Accuracy, F1-score, confusion matrix per model

---

## Results Summary

| Model | Accuracy | F1-Score |
| --- | --- | --- |
| TF-IDF + SVM | TBD | TBD |
| TF-IDF + Logistic Regression | TBD | TBD |
| DistilBERT | TBD | TBD |
| BERT | TBD | TBD |

> Update this table after completing model training.

---

## Technologies Used

- Python 3.x
- Scikit-learn
- HuggingFace Transformers
- PyTorch
- NLTK / spaCy
- Pandas, NumPy
- Matplotlib, Seaborn
- Flask / FastAPI (planned deployment)

---

## Repository

https://github.com/sravananambiar20/Sentiment_Analysis_of_Product_Reviews
