# News Classification Model (TIDA Assignment 2)

Supervised text classification of news headlines into 8 categories
(Business, City / Local, Entertainment, India / National, Lifestyle, Sports, Technology, World),
deployed with Streamlit Community Cloud.

**Live app:** _paste your Streamlit URL here_

## Pipeline
raw text -> cleaning (lowercase, strip punctuation, stopwords) -> TF-IDF (1-2 grams) -> Logistic Regression / Linear SVM / Naive Bayes -> evaluation -> Streamlit app

## Dataset
India News Headlines (Times of India, 2001-2023). Raw `headline_category` values were mapped to 8 classes
and balanced to 30,000 headlines per class (240k total, 80/20 stratified split).

## Files
- `app.py` - Streamlit app
- `preprocess.py` - text cleaning + category mapping
- `train.py` - training script (regenerates `models/` and `figures/`)
- `News_Classification.ipynb` - full notebook
- `models/news_classifier.joblib` - trained vectorizer + model

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
