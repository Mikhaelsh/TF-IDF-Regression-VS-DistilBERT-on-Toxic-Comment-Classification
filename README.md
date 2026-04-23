# Toxic Comment Classification

This project focuses on **multi-label toxic comment classification** using Natural Language Processing (NLP) techniques.  
The goal is to identify different types of toxicity in online comments.

Demonstration Video: https://drive.google.com/drive/folders/1Ncn4iP5JZhvFFz_da6LP6gjcpaNABTL3?usp=sharing

The project implements and compares:
- A **baseline machine learning model** using TF-IDF + Logistic Regression
- A **transformer-based model** using DistilBERT

Dataset used: **Jigsaw Toxic Comment Classification Challenge**

Trained model checkpoints are not included due to GitHub size limitations. All results are reproducible by running the notebook.
---

## Labels
Each comment may belong to one or more of the following classes:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

---

## Models Implemented

### 1. TF-IDF + Logistic Regression
- Converts text into TF-IDF features
- Uses One-vs-Rest Logistic Regression for multi-label classification
- Fast, interpretable, and serves as a strong baseline

### 2. DistilBERT
- Pretrained transformer model (`distilbert-base-uncased`)
- Fine-tuned for multi-label classification
- Captures contextual and semantic meaning of text
- Evaluated using Macro ROC-AUC and Macro F1-score

---

## Dataset
The dataset is **not included** in this repository due to size limitations.

You can download it from:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

Required files:
- `train.csv`
- `test.csv`
- `test_labels.csv`
- `sample_submission.csv`

---

## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
