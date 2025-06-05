# 🤖 Automated Resume Screening and Job Matching System

An AI-powered system that intelligently classifies resumes against job descriptions and computes ATS scores using NLP, machine learning, deep learning, and LLM techniques.

## 📌 Overview

This project aims to streamline the recruitment process by:
- Classifying candidates as **Good Fit**, **Potential Fit**, or **No Fit**
- Generating **ATS (Applicant Tracking System) scores** to evaluate candidate-job compatibility
- Providing **skill gap analysis** and keyword suggestions for resume improvement

## 🚀 Features

- ✅ Resume-JD compatibility classification using ML & DL models
- 📊 ATS score computation with semantic similarity
- 📌 Keyword extraction from JDs using **Gemini 1.5 Flash API**
- 🔍 Prompt-based LLM resume matching for explainable AI
- 🧠 Support for both traditional and transformer-based NLP models

## 🛠️ Technologies Used

- **Languages:** Python
- **Libraries/Frameworks:** TensorFlow, Keras, Scikit-learn, Transformers, Gensim
- **NLP:** GloVe Embeddings, TF-IDF, Word2Vec, BERT, RoBERTa, MPNet, MiniLM
- **ML Models:** Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, CatBoost
- **DL Models:** LSTM, BiLSTM, BiGRU, Hybrid (BiLSTM + BiGRU)
- **Others:** Hugging Face Datasets, Keras Tuner, Gemini API

## 🧪 Model Performance

| Model                  | Accuracy |
|------------------------|----------|
| BiLSTM (Deep Learning) | 99%      |
| LightGBM (Ensemble)    | 98.56%   |
| Logistic Regression    | 66.45%   |

## 📂 Dataset

- Source: Hugging Face  
- Contains 6,242 samples  
- Columns: `text` (resume + JD), `label` (Good Fit / No Fit / Potential Fit)

## ⚙️ How It Works

1. **Preprocessing**: Clean and split resume and job description texts
2. **Embedding**: Convert words using GloVe and transformers
3. **Classification**: Train models to classify fit category
4. **Scoring**: Generate ATS score based on semantic similarity
5. **Keyword Analysis**: Use LLM (via Gemini API) for resume feedback

## 🔧 Setup Instructions

```bash
git clone https://github.com/your-username/resume-matching-ai.git
cd resume-matching-ai
pip install -r requirements.txt
python run_pipeline.py
