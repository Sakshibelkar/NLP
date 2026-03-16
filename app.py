import streamlit as st
import pandas as pd
import numpy as np
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------------
# Title
# -------------------------------
st.title("Sentiment Analysis App")

st.write("Upload dataset and predict sentiment from review text")

# -------------------------------
# File Upload 
# -------------------------------
uploaded_file = st.file_uploader("P652-Dataset.xlsx", type=["xlsx"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Drop missing values
    df = df.dropna(subset=['body', 'rating'])

    # -------------------------------
    # Sentiment Mapping
    # -------------------------------
    def get_sentiment(rating):
        if rating in [4,5]:
            return "Positive"
        elif rating == 3:
            return "Neutral"
        else:
            return "Negative"

    df["sentiment"] = df["rating"].apply(get_sentiment)

    # -------------------------------
    # Text Cleaning
    # -------------------------------
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):

        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))

        words = text.split()
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

        return " ".join(words)

    df["clean_text"] = df["body"].apply(clean_text)

    st.subheader("Cleaned Data")
    st.write(df[["body","clean_text","sentiment"]].head())

    # -------------------------------
    # Feature Extraction
    # -------------------------------
    tfidf = TfidfVectorizer(max_features=5000)

    X = tfidf.fit_transform(df["clean_text"])
    y = df["sentiment"]

    # -------------------------------
    # Train Test Split
    # -------------------------------
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    # -------------------------------
    # Model Training
    # -------------------------------
    model = LinearSVC()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test,y_pred)

    st.subheader("Model Accuracy")
    st.write(acc)

    st.subheader("Classification Report")
    st.text(classification_report(y_test,y_pred))

    # -------------------------------
    # User Input Prediction
    # -------------------------------
    st.subheader("Test Your Own Review")

    user_input = st.text_area("Enter Review Text")

    if st.button("Predict Sentiment"):

        cleaned = clean_text(user_input)

        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)

        st.success(f"Predicted Sentiment: {prediction[0]}")