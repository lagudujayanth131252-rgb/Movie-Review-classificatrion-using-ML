# Movie-Review-classificatrion-using-ML# 🎬 Movie Review Classification using Machine Learning

Sentiment classification of IMDb movie reviews (positive vs. negative), with the whole NLP + ML pipeline implemented **from scratch in pure Python**. It uses only the standard library, with no scikit-learn, NumPy, or pandas.

## Overview

The project downloads a sample of the IMDb reviews dataset, cleans the text, converts it into TF-IDF features (unigrams + bigrams), trains several classifiers, compares them, and runs a live demo on custom reviews.

## Features

- **Automatic dataset download**: pulls the IMDb dataset CSV and saves a balanced sample (default 3,000 reviews, 50% positive / 50% negative) to `imdb_reviews.csv`
- **Text preprocessing**: HTML tag and URL removal, lowercasing, contraction expansion (keeps negations like "not"), and non-alphabet stripping
- **Custom TF-IDF vectorizer**: unigrams + bigrams, sublinear TF, smoothed IDF, L2 normalization, `max_features=6000`, `min_df=2`
- **Classifiers implemented from scratch**:
  - Multinomial Naive Bayes
  - Logistic Regression (SGD with L2 regularization)
  - Linear SVM (hinge-loss SGD)
  - Weighted Soft-Voting Ensemble (LR 0.45, SVM 0.35, NB 0.20)
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrix
- **Interpretability**: shows the top positive and negative words learned by Logistic Regression
- **Live inference demo**: predicts sentiment and confidence for sample reviews using the best model

## Project Structure

```
.
├── ai_project.py       # Full pipeline: data loading, preprocessing, models, evaluation, demo
├── imdb_reviews.csv    # Generated automatically on first run
└── README.md
```

## Requirements

- Python 3.7+
- Internet connection on the first run (to download the dataset)

No third-party packages are needed.

## Usage

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Run the project
python ai_project.py
```

The script will:

1. Download and cache the dataset (first run only)
2. Split the data 80/20 into train and test sets
3. Extract TF-IDF features
4. Train and evaluate all models
5. Print the best model, its confusion matrix, and metrics
6. Show the most influential words
7. Run the live inference demo

To change the dataset size, edit the `sample_size` argument in `load_imdb_dataset(sample_size=3000)`.

## How It Works

| Stage | Description |
|-------|-------------|
| 1. Data | IMDb reviews, balanced sample, shuffled with a fixed seed |
| 2. Preprocessing | Clean text, expand contractions, tokenize |
| 3. Features | TF-IDF over unigrams and bigrams (sparse dict vectors) |
| 4. Models | Naive Bayes, Logistic Regression, Linear SVM, Soft-Voting Ensemble |
| 5. Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix |
| 6. Inference | Predict sentiment and confidence for new reviews |

## Sample Output

Running the script prints a comparison table in this format (your numbers will vary slightly depending on the run):

```
Classifier Model             Accuracy     F1-Score   ROC-AUC    Time
--------------------------------------------------------------------
Multinomial Naive Bayes      ...          ...        ...        ...
Logistic Regression          ...          ...        ...        ...
Linear SVM                   ...          ...        ...        ...
Soft-Voting Ensemble         ...          ...        ...        -
```

> 💡 Tip: after running it, paste your actual results table here.

## Dataset

IMDb Movie Reviews Sentiment dataset (50K labeled reviews), loaded from:
[Ankit152/IMDB-sentiment-analysis](https://github.com/Ankit152/IMDB-sentiment-analysis)

## Possible Improvements

- Use the full 50K-review dataset
- Add cross-validation and hyperparameter tuning
- Add stopword handling and stemming/lemmatization
- Compare against scikit-learn implementations
- Wrap the model in a small web app (Flask or Streamlit)

## Author

**Lagudu Jayanth**
