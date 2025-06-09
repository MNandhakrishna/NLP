# Fake and True News Detection Project

This project aims to detect fake and true news using various data analysis and machine learning techniques. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, model building, and evaluation.

## Introduction

Fake news detection is an essential task in natural language processing, helping prevent the spread of misinformation. This project focuses on predicting the authenticity of news articles using machine learning models.

## Dataset

The dataset used in this project contains news articles labeled as true or fake. It includes features such as the article's text and label.  
📌 **Source**: [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)



## Project Structure

```bash
fake-news-detection/
│
├── data/
│   ├── True.csv
│   ├── Fake.csv
│
├── notebooks/
│   └── Fake_News_Detection.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   └── evaluation.py
│
├── requirements.txt
└── README.md
```

## Data Preprocessing

The dataset underwent various preprocessing steps, including:

* Converting text to lowercase
* Removing punctuation and special characters
* Removing stopwords
* Tokenization
* Lemmatization
* Handling missing/null values

## Exploratory Data Analysis (EDA)

Performed to gain insights into the data and identify patterns using:

* Word clouds
* Distribution plots of article lengths
* Bar charts of frequent terms
* Class balance analysis

## Feature Engineering

* Applied **TF-IDF Vectorization** to convert text into numerical features
* Prepared input formats suitable for both traditional ML and deep learning models
* Optionally used embeddings like Word2Vec or GloVe for deep models

## Model Building

### Traditional Machine Learning Models:

* Logistic Regression
* Support Vector Machine (SVM)
* Naive Bayes

### Deep Learning Models:

* LSTM (Long Short-Term Memory)
* GRU (Gated Recurrent Unit)
* CNN (Convolutional Neural Network)

Models were trained and validated using stratified splits and cross-validation.

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC Score

## Results

| Model               | Accuracy  |
| ------------------- | --------- |
| Logistic Regression | 81%       |
| SVM                 | 74%       |
| Naive Bayes         | 65%       |
| LSTM                | 89%       |
| **GRU**             | **92%**       |
| CNN                 | 88%       |

 GRU achieved the best results, demonstrating strong ability in capturing context and dependencies in news articles.

## Conclusion

In conclusion, this project demonstrates the use of NLP and machine learning to effectively detect fake news. The GRU model showed superior performance in classifying news articles and can be further improved with live data and fine-tuning.


