# Project Title: Sentiment Analysis of Product Reviews

## Team Members

| Name | Reg No | Course |
| --- | --- | ---|
| Ardra Selin A G | 253006 | MSc. COMPUTER SCIENCE WITH SPECIALIZATION IN DATA ANALYTICS |
| Sravana Nambiar | 253212 | MSc. DATA SCIENCE AND BIO AI |
| Archana T | 253205 | MSc. DATA SCIENCE AND BIO AI |

## 👥 Team

| Member | Role |
|--------|--------|
| Ardra Selin A G | Led the end-to-end development of the project. Set up the complete repository structure including all folders (code/, models/, reports/, results/, Data/). Built and deployed the full Streamlit web application with 5 pages — Home, Single Review Prediction, Batch Prediction, Model Performance, and About. Trained and evaluated all 4 models (TF-IDF+SVM, TF-IDF+LR, DistilBERT, BERT) achieving best accuracy of 84.50% with DistilBERT. Created detailed README.md, Contributing.md, Project_Summary.md and .gitignore files. Added classification reports, app screenshots, and sample dataset files. Integrated model hosting via Google Drive for cloud deployment. |
| Sravana Nambiar |Performed data preprocessing and cleaning to improve dataset quality and prepare it for model training. Successfully deployed the application using Streamlit Streamlit for interactive predictions and user-friendly access. Also contributed to repository management and README documentation updates. Completed preprocessing tasks including handling missing values, feature preparation, and data transformation. Built and deployed the project interface using Streamlit Streamlit and assisted in maintaining the GitHub repository and project documentation.Worked on dataset preprocessing and prepared the project pipeline for deployment. Integrated the trained model into a Streamlit Streamlit web application and contributed to updating project files such as README.md and requirements.txt.Contributed to project development by performing preprocessing, organizing the repository structure, and deploying the machine learning application using Streamlit Streamlit. Also helped improve project documentation and setup instructions for easier collaboration.|
| Archana T | Contributed to the machine learning pipeline by training, evaluating, and comparing all four models — TF-IDF+SVM, TF-IDF+Logistic Regression, DistilBERT, and BERT — to analyze model performance and optimize prediction accuracy. Successfully generated and integrated the `model.pkl` file for deployment and real-time prediction purposes. Managed project dependencies by creating and updating the `requirements.txt` file to ensure reproducibility and seamless environment setup across different systems. Also contributed to improving the project documentation by updating the `README.md` file with detailed project information, installation steps, usage instructions, and workflow explanations to support collaboration and easier project understanding.



---

## Problem Statement

Sentiment Analysis has become one of the most important applications of Natural Language Processing (NLP) in today’s data-driven digital world. Every day, millions of users share their opinions, experiences, and feedback through online reviews on e-commerce platforms, social media, and discussion forums. For businesses, these reviews contain valuable insights about customer satisfaction, product quality, service efficiency, and overall user experience. However, manually analyzing such an enormous volume of textual data is time-consuming, inefficient, and often impractical. This creates the need for intelligent automated systems capable of understanding and classifying customer opinions accurately and efficiently.

This project focuses on developing an advanced Sentiment Analysis system for product reviews using the **Amazon Fine Food Reviews** dataset. The system is designed to automatically classify customer reviews into three sentiment categories: **Positive**, **Neutral**, and **Negative**. By leveraging both traditional machine learning techniques and modern transformer-based deep learning architectures, the project aims to compare different NLP methodologies and identify the most effective approach for sentiment classification tasks.

The project begins with extensive data preprocessing and text cleaning operations to transform raw textual data into a structured format suitable for machine learning models. Since real-world review datasets often contain noise such as punctuation, stopwords, HTML tags, emojis, repeated characters, and inconsistent formatting, preprocessing plays a critical role in improving model performance. Various Natural Language Processing techniques such as tokenization, lowercasing, stopword removal, stemming, and lemmatization are applied to prepare the dataset for analysis and feature extraction.

In addition to preprocessing, the project includes detailed Exploratory Data Analysis (EDA) to better understand the distribution and characteristics of the dataset. EDA helps uncover important patterns such as sentiment distribution, frequently occurring words, review lengths, and customer behavior trends. Visualization techniques including bar plots, word clouds, frequency distributions, and sentiment comparisons are used to derive meaningful insights from the data before model development.

The project implements and compares both traditional and advanced NLP models. Traditional machine learning models such as **TF-IDF + Support Vector Machine (SVM)** and **TF-IDF + Logistic Regression (LR)** are used as baseline approaches because of their simplicity, efficiency, and strong performance on text classification tasks. Alongside these models, state-of-the-art transformer architectures including **BERT (Bidirectional Encoder Representations from Transformers)** and **DistilBERT** are trained and evaluated to leverage contextual language understanding and deep semantic representation capabilities.

Transformer-based models have significantly improved NLP performance in recent years due to their ability to capture contextual relationships between words in a sentence. By comparing traditional feature-engineering-based models with transformer-based deep learning models, the project provides a comprehensive analysis of how different approaches perform in sentiment classification tasks involving real-world customer review data.

The complete workflow follows a structured data science and machine learning pipeline, including:

* Data collection and dataset preparation
* Data preprocessing and text normalization
* Exploratory Data Analysis (EDA)
* Feature extraction using TF-IDF and transformer embeddings
* Model training and hyperparameter tuning
* Performance evaluation and comparison
* Deployment preparation for real-time predictions

The performance of all models is evaluated using multiple classification metrics such as **accuracy, precision, recall, F1-score, and confusion matrices** to ensure reliable and unbiased comparison. The final goal of the project is to identify the most accurate and efficient sentiment analysis model that can be deployed for automated review classification and customer feedback analysis in real-world applications.

---

## Objectives

* Develop an automated Sentiment Analysis system capable of classifying product reviews into **Positive**, **Neutral**, and **Negative** sentiment categories.
* Perform extensive data preprocessing and text cleaning to improve the quality and consistency of textual data before model training.
* Conduct detailed Exploratory Data Analysis (EDA) to identify trends, patterns, and sentiment distributions within the dataset.
* Apply Natural Language Processing techniques such as tokenization, stopword removal, stemming, and lemmatization for effective text processing.
* Extract meaningful textual features using both traditional techniques like **TF-IDF** and advanced contextual embeddings from transformer models.
* Build and compare traditional machine learning models including **TF-IDF + Support Vector Machine (SVM)** and **TF-IDF + Logistic Regression (LR)**.
* Train and evaluate advanced transformer-based deep learning models such as **BERT** and **DistilBERT** for sentiment classification.
* Compare the performance of all models using evaluation metrics including **accuracy, precision, recall, F1-score, and confusion matrices**.
* Identify the best-performing model based on classification performance, computational efficiency, and generalization capability.
* Prepare the trained model for deployment and real-time sentiment prediction applications.
* Demonstrate how NLP and deep learning techniques can be applied to solve large-scale real-world business problems involving customer feedback analysis.


---

## Dataset

- **Source:** Amazon Fine Food Reviews (via Kaggle)
- The dataset contains **568,454 reviews** from Amazon spanning over 10 years

### Dataset Link

- https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

### Key Columns

- `Text` — Full review text
- `Summary` — Short review summary
- `Score` — Rating (1–5 stars, mapped to sentiment)

### Sentiment Mapping

| Score | Sentiment |
| --- | --- |
| 4–5 stars | Positive |
| 3 stars | Neutral |
| 1–2 stars | Negative |

---

## Methodology

### 1. Data Preprocessing

- Removed duplicate and null entries
- Mapped star ratings to sentiment labels (Positive / Neutral / Negative)
- Lowercased text, removed HTML tags, punctuation, and stopwords
- Applied lemmatization for word normalization
- Handled class imbalance via stratified sampling

---

### 2. Exploratory Data Analysis (EDA)

- Visualized sentiment class distribution
- Generated word clouds for each sentiment category
- Analysed review length distributions
- Identified most frequent words per sentiment class

---

### 3. Feature Engineering

- **TF-IDF Vectorization** for traditional ML models (top N-gram features)
- **BERT / DistilBERT tokenization** for transformer-based models
- Train-test split (80/20) with stratification

---

### 4. Model Building

The following models were implemented:

- TF-IDF + Support Vector Machine (SVM)
- TF-IDF + Logistic Regression
- DistilBERT (fine-tuned)
- BERT (fine-tuned)

---

### 5. Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Results & Comparison

| Model | Accuracy |
| --- | --- |
| TF-IDF + SVM | 83.97% |
| TF-IDF + Logistic Regression | 84.15% |
| DistilBERT | 84.50% |
| BERT | 83.70% |

## Results & Comparison

| Model | Accuracy | F1-Score |
| --- | --- | --- |
| TF-IDF + SVM | 83.97% | 0.81 |
| TF-IDF + Logistic Regression | 84.15% | 0.81 |
| DistilBERT | 84.50% | 0.82 |
| BERT | 83.70% | 0.79 |

**Best Model: DistilBERT with 84.50% accuracy and F1-score of 0.82** 🏆

---

## Model Performance Summary

### TF-IDF + SVM
1. **Accuracy:** 83.97%
2. **F1-Score:** 0.81
3. **Strengths:** Fast training, works well with high-dimensional sparse text features.

### TF-IDF + Logistic Regression
1. **Accuracy:** 84.15%
2. **F1-Score:** 0.81
3. **Strengths:** Highly interpretable, stro

> ⚠️ Results to be updated after model training is complete.

---

## Model Performance Summary

### TF-IDF + SVM

1. **Accuracy:** TBD
2. **Strengths:** Fast training, works well with high-dimensional sparse text features.
3. **Precision/Recall/F1:** TBD

### TF-IDF + Logistic Regression

1. **Accuracy:** TBD
2. **Strengths:** Highly interpretable, strong baseline for text classification tasks.
3. **Precision/Recall/F1:** TBD

### DistilBERT

1. **Accuracy:** TBD
2. **Strengths:** Lightweight transformer that retains ~97% of BERT's performance at 60% of the size. Captures contextual word meaning.
3. **Precision/Recall/F1:** TBD

### BERT

1. **Accuracy:** TBD
2. **Strengths:** State-of-the-art contextual embeddings. Best at understanding nuanced language in reviews.
3. **Precision/Recall/F1:** TBD

---

## Evaluation Matrix

Classification reports for each model will be added to `reports/classification_report/` once training is complete.

---

## Conclusion

This project develops a Sentiment Analysis system capable of classifying Amazon product reviews into three sentiment categories. By comparing traditional ML approaches (SVM, Logistic Regression) with modern transformer-based models (BERT, DistilBERT), we aim to demonstrate the trade-offs between speed, interpretability, and accuracy in NLP tasks. The complete pipeline covers data preprocessing, EDA, model development, and evaluation — serving as a comprehensive example of an NLP project lifecycle.

---

## Repository Structure

```
Sentiment_Analysis_of_Product_Reviews/
│
├── Data/                          # Dataset info and links
├── code/                          # Jupyter notebooks
│   └── sentiment_analysis.ipynb
├── models/                        # Saved model files (.pkl, .pt)
├── reports/
│   └── classification_report/     # Per-model classification reports
├── results/
│   └── plots/                     # Output visualizations
├── README.md
├── Contributing.md
├── Project_Summary.md
├── Requirements.txt
└── .gitignore
```
## Application Screenshots

### Home Page
![Sentiment Analysis App Home Page Dashboard](home%20page.png)

### Single Review Prediction
![Single Review Prediction](single_review_prediction.png)

### Model Performance
![Model Accuracy and F1-Score Comparison Charts](modelperformance.png)

### Metrics Summary
![Metrics Summary](summary.png) 

### About Page
![Project Overview and Team Members Page](abouttheapp.png)


## Live Application

🚀 **Streamlit App:** 
[Sentiment Analysis App](https://sentimentanalysisofappuctreviews-ibsbwdzbzrapjcmypchsse.streamlit.app/)
