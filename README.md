# Sentiment Analysis of Chinese Microblog Posts on Attitude Toward Vaccination

## Overview

This project implements a machine learning based sentiment analysis system for Chinese microblog posts related to COVID-19 and vaccination.

The project is inspired by the reference study:

"Sentiment Analysis of Chinese Microblog Posts on Attitude Toward Vaccination Since the Distribution of COVID-19 Vaccines in China"

The system preprocesses Chinese social media text, extracts TF-IDF features and compares multiple machine learning classifiers.

## Objectives

- Analyze sentiment in Chinese microblog posts.
- Preprocess Chinese text using Jieba tokenization.
- Convert text into numerical features using TF-IDF.
- Compare Naive Bayes, Logistic Regression and SVM.
- Evaluate models using accuracy, precision, recall and F1-score.
- Build a model capable of predicting sentiment for new Chinese text.

## Dataset

The project uses a publicly available labelled Chinese Weibo COVID-19 sentiment dataset.

The original dataset contains 21,173 labelled Chinese microblog posts with seven emotion categories:

- Fear
- Disgust
- Optimism
- Surprise
- Gratitude
- Sadness
- Anger

For binary sentiment classification:

- Optimism and Gratitude → Positive
- Fear, Disgust, Sadness and Anger → Negative
- Surprise → Neutral and excluded from binary classification

## Methodology

The project follows the pipeline:

Raw Weibo Posts
        ↓
Data Cleaning
        ↓
Chinese Word Segmentation using Jieba
        ↓
TF-IDF Feature Extraction
        ↓
Train/Test Split
        ↓
Naive Bayes / Logistic Regression / SVM
        ↓
Performance Evaluation
        ↓
Sentiment Prediction

## Models

### Naive Bayes
Used as a probabilistic baseline classifier.

### Logistic Regression
Used for linear binary sentiment classification.

### Support Vector Machine
Used to identify the best separating decision boundary between positive and negative sentiment classes.

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Naive Bayes | 84.76% | 85.23% | 84.76% | 84.14% |
| Logistic Regression | 84.30% | 85.26% | 84.30% | 83.46% |
| SVM | 84.81% | 84.68% | 84.81% | 84.56% |

SVM achieved the highest accuracy and F1-score among the evaluated models.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jieba
- Matplotlib
- Seaborn
- Jupyter / Google Colab

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt