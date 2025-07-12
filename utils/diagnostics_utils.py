import numpy as np
import pandas as pd
import streamlit as st

def identify_noise_words(model, vectorizer, X, y, threshold_entropy=0.95, threshold_docfreq=0.7, debug=False):
    
    feature_names = vectorizer.get_feature_names_out()
    # print(feature_names)
    num_classes = model.class_count_.shape[0]  # 3
    class_log_probs = model.feature_log_prob_  # shape: (n_classes, n_features)

    # Convert to probabilities
    class_probs = np.exp(class_log_probs)
    log_var = np.var(class_log_probs, axis=0)
    # print(class_log_probs, log_var)

    # Document frequency (fraction of docs with word)
    doc_freqs = np.array((X > 0).sum(axis=0)).flatten() / X.shape[0]

    # Estimate P(c | w) using Bayes' rule approximation
    p_c_given_w = class_probs / class_probs.sum(axis=0, keepdims=True)
    entropy = -np.sum(p_c_given_w * np.log(p_c_given_w + 1e-10), axis=0)
    max_entropy = np.log(num_classes)
    
    df = pd.DataFrame({
        "Word": feature_names,
        "Entropy": entropy,
        "EntropyRatio": entropy / max_entropy,
        "DocFreq": doc_freqs,
        "LogProbVar": log_var
    })

    if debug:
        st.subheader("🔍 Raw Word Statistics")
        st.dataframe(df.sort_values("EntropyRatio", ascending=False))
        st.write("EntropyRaio (max):", df["EntropyRatio"].max())
        st.write("DocFreq (max):", df["DocFreq"].max())
        st.write("LogProbVar (max):", df["LogProbVar"].max())

    # Filter candidates
    stop_candidates = df[
        (df["EntropyRatio"] > threshold_entropy) &
        (df["DocFreq"] > threshold_docfreq)
    ].sort_values(by="EntropyRatio", ascending=False)

    return df, stop_candidates
