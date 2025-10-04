import streamlit as st
import pandas as pd

# --- Load and preprocess dataset ---
st.header("📊 Dataset Summary")

st.markdown("""
The dataset is sourced from [takala/financial_phrase_book](https://huggingface.co/datasets/takala/financial_phrasebank) and contains financial news headlines labeled by sentiment.

Each headline is categorized as:
- **Positive (1)**
- **Negative (0)**
- **Neutral (2)**

The goal is to classify unseen financial news into one of these categories based on the textual content.
The dataset contains **rows of short news headlines**, which are typically more ambiguous and compact in language than product reviews or articles, 
making the classification noisy. While some headlines are clearly positive or negative, there is always subjectivity involved in labeled data.
""")

path = "data/financial-news.csv"
data = pd.read_csv(path, encoding='ISO-8859-1', names=['label', 'text'])
st.write(data.head(5))
st.write(f"Dataset contains {data.shape[0]} samples.")
