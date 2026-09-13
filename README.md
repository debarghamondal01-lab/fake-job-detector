# Fake Job Posting Detector 🕵️‍♂️

An NLP-based Machine Learning model that detects fraudulent job postings to protect students from scams.

## 🚀 The Problem
Fake job postings scam thousands of students every year, stealing their money and personal data. This tool helps users instantly check if a job posting is real or fake.

## 🧠 The Approach
- **Dataset:** Kaggle Fake Job Postings (18,000+ samples)
- **Text Preprocessing:** Combined title, company profile, description, and requirements into a single text feature.
- **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency) with 5,000 features.
- **Model:** Logistic Regression with class balancing (to handle the heavily imbalanced dataset of 95% real vs 5% fake).
- **Accuracy:** ~97% on test data.

## 💻 Tech Stack
- Python, scikit-learn, pandas, NumPy
- **Trained on:** Google Colab (cloud-based, due to local hardware constraints)
- **Runs locally:** Lightweight inference script for instant predictions.

## ⚙️ How to Use
1. Install dependencies:
   ```bash
   pip install scikit-learn pandas numpy
2. Run the detector:
   ```bash
   python detector.py
3. Paste any job description. When finished, type END on a new line and press Enter. The model will tell you if it's REAL or FAKE with a confidence percentage.
