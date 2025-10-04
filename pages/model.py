import streamlit as st

import nltk
import pandas as pd

from utils.model_utils import (
    load_data, vectorize,
    train_and_evaluate, get_top_n_words_per_class,
)
from utils.diagnostics_utils import identify_noise_words

def check_nltk_setup():
    for resource, path in [
        ("punkt", "tokenizers/punkt"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet") ]:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"{resource} not found.")


check_nltk_setup()
data = load_data("data/financial-news.csv")

# ---- Model Configuration ----
st.header("⚙️ Model Configuration")

st.markdown("""
Before we train the model, choose your desired settings:
""")

vectorizer_type = st.radio(
    "Vectorization Method:",
    ["Bag of Words", "TF-IDF"],
    index=0,
    key="vectorizer_choice"
)

use_smote = st.sidebar.radio(
    "Apply SMOTE for Class Balancing?",
    ["No", "Yes"],
    index=0,
    key="smote_choice"
) == "Yes"

"---"

# # --- Train model on dataset ---

with st.spinner("Training the model..."):
    X, vectorizer = vectorize(
        data['processed'], method='bow' if vectorizer_type == "Bag of Words" else 'tfidf')
    model, report, matrix = train_and_evaluate(
        X, data['label'], apply_sampling=use_smote)

st.header("🧹 Noise Word Filter Tool")

with st.expander("#### How does Entropy and Document Frequency affect predictions?"):
    st.markdown(r"""
    Some words like "EUR" or "mn" appear in both lists.  
                
    - “Company X posts EUR 20 mn loss” → Negative
    - “Company Y secures EUR 50 mn investment” → Positive
                
    The dominance weight of "EUR" and "mn" is mitigated to some extnent from BoW ~(0.03-0.05) range to 
    TF-IDF (~0.01). 

    Their co-occurrence with strong sentiment words gives them statistical importance across classes.
    This motivates us use entropy and class variance metrics to flag such **statistically uninformative** words.
                
    Let's make an effort to clean our corpus from these.

    We define,  
        $ \text{Normalized Entropy}(w) = - \frac{\sum_{c} P(c \mid w) logP(c \mid w)}{log(\#\text{classes})} $    
        $ \text{Document Frequency}(w) = \frac{\#\text{docs with w}}{\text{total docs}}$

    - For 'mn' and 'eur' the entropies are 0.777 and 0.914. This indicates that 'eur' appears in evenly across classes 
    and is statistically insignificant. 'mn' less so but is distributed fairly evenly.
    - For 'mn' and 'eur' the document frequencies 0.058 and 0.129. This means 'mn' is not very common appearing in only 
    5.8% of the docuemnts and 'eur' in only 12.9% of the documents. 
                
    - The maximum document frequency of any word is 0.186 for 'company', followed by 'eur', 'said (0.112)' and 'finish (0.104)'
                
    Adjust the sliders below to set the minimum entropy and document frequency
    """)


threshold_entropy = st.slider(
    "Minimum Normalized Entropy", 0.65, 1.0, 0.75, 0.01)
threshold_df = st.slider("Minimum Document Frequency", 0.05, 0.20, 0.10, 0.01)

noise_stats, candidates = identify_noise_words(model, vectorizer, X, data['label'],
                                               threshold_entropy=threshold_entropy,
                                               threshold_docfreq=threshold_df, debug=False)

st.subheader("📋 Candidate Stopwords")
st.dataframe(candidates[["Word", "EntropyRatio", "DocFreq", "LogProbVar"]])


if st.button("🔁 Retrain Without These Words"):

    custom_stopwords = list(candidates["Word"])
    # Vectorize again with updated stopword list
    X, vectorizer = vectorize(data['processed'], method='tfidf' if vectorizer_type == 'TF-IDF' else 'bow',
                              stop_words=custom_stopwords)
    model, report, matrix = train_and_evaluate(
        X, data['label'], apply_sampling=use_smote)
    st.success("Retrained model with custom stopwords removed!")

"---"
st.session_state['vectorizer'] = vectorizer
st.session_state['model'] = model
st.session_state['report'] = report
st.session_state['matrix'] = matrix

# # --- Visualizations ---

st.subheader("🧠 What Words Influence Predictions?")

st.markdown(
    "The top words per class are shown below based on model feature weights:")

top_words = get_top_n_words_per_class(model, vectorizer, class_labels=[
                                      "Negative", "Positive", "Neutral"])

col_neg, col_pos, col_neu = st.columns(3)

with col_neg:
    st.subheader("🟥 Negative Words")
    df_neg = pd.DataFrame(top_words["Negative"], columns=[
                          "Word", "Estimated Probability"])
    st.dataframe(df_neg, use_container_width=True)

with col_pos:
    st.subheader("🟩 Positive Words")
    df_pos = pd.DataFrame(top_words["Positive"], columns=[
                          "Word", "Estimated Probability"])
    st.dataframe(df_pos, use_container_width=True)

with col_neu:
    st.subheader("🟦 Neutral Words")
    df_neu = pd.DataFrame(top_words["Neutral"], columns=[
                          "Word", "Estimated Probability"])
    st.dataframe(df_neu, use_container_width=True)

