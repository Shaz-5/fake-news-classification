# Fake News Detection

This project focuses on detecting fake news using various machine learning techniques. The dataset used in this project is sourced from [Kaggle Fake News Challenge](https://www.kaggle.com/c/fake-news/data). Multiple approaches, such as classical machine learning models, GloVe embeddings, BERT embeddings, and CNN-based models, are used to identify whether a news article is real or fake.

## Project Overview

This repository contains code for data preprocessing, model training, and evaluation on a dataset of fake and real news articles. The techniques explored include:

- **Data Preprocessing**: Text cleaning, tokenization, stop-word removal, and lemmatization.
- **Feature Extraction**: GloVe word embeddings, BERT embeddings, and classical bag-of-words models.
- **Model Training**: Naive Bayes, Decision Tree, Random Forest, K-Nearest Neighbors, and CNN models.
- **Evaluation**: Accuracy, confusion matrix, and classification reports.

## Dependencies

The following libraries are required to run the project:

```bash
pip install nltk numpy pandas seaborn matplotlib scikit-learn gensim tensorflow torch tqdm keras
```

## Data
The dataset used in this project is the Fake News dataset, which is available on Kaggle. It contains labeled news articles with the following columns:

- `id`: Unique identifier for each article.
- `title`: Title of the article.
- `author`: Author of the article.
- `text`: The content of the article.
- `label`: Target variable (1 for fake news, 0 for real news).

## Project Structure

The project has the following directory and file structure:

- `train.csv`, `test.csv`: Training and test datasets.
- `GloVe Embedding/`: Folder containing pre-trained GloVe embeddings and processed data. (not added due to size)
- `BERT Embedding/`: Folder containing tokenized data and BERT embeddings. (not added due to size)
- `Models/`: Folder containing trained models (e.g., CNN model). 
- `Training/`: Jupyter notebook containing exploratory data analysis and model training and evaluation.

## Results
The models' performances are evaluated using accuracy, precision, recall, and F1-score. The CNN-based model using Glove Embeddings achieved an accuracy of 89.28% on the test set.
