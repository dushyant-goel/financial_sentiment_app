import streamlit as st

# Set page config
st.set_page_config(
    page_title="Financial News Sentiment Analyzer",
)

# --- Start of Page ---
st.title("Sentiment Analysis of Financial News")
st.subheader("""Analyze the sentiment of unseen user-input financial news.""")

st.markdown(f"""
            
Welcome to this interactive **Sentiment Analysis of Financial News** app!

We walk through the full pipeline of building and interpreting a Naïve Bayes classifier
for real-world financial text data. The focus is not just on model accuracy,
but also on interpretability, limitations, and iterative refinement.

The goal is to classify unseen financial news headlines into one of following categories based on its text.   
**Positive**, **Negative** and **Neutral** sentiments.

To do this, we process the headline to extract some useful textual features from it. This is called **vectorization**.
We then input this vector into a classifier, which gives a probability for the headline in each class.
          
We use a supervised learning approach. This mean, we train our own classifier on labeled real world data. 
            
### 🔍 What You'll Learn and Explore

- 📐 **Naïve Bayes Theory**  
  Begin with a deep dive into the mathematics behind Naïve Bayes — how we estimate class probabilities and what assumptions we make (like word independence).

- 🧾 **Text Vectorization Techniques**  
  Learn how raw financial text is converted into numerical features using methods like **Bag-of-Words (BoW)** and **TF-IDF**. We explain how each affects model learning.

- ⚖️ **Class Imbalance and SMOTE**  
  Our data is imbalanced across sentiment classes. We use **SMOTE (Synthetic Minority Over-sampling Technique)** to address this and examine the impact on model performance.

- 📊 **Model Evaluation & Diagnostics**  
  After training, we visualize metrics like the confusion matrix and F1-scores. We also extract and inspect the **top sentiment-predicting words**.

- 🧹 **Noise Word Filtering Tool**  
  Some words, may dominate predictions without carrying sentiment. We use entropy- and frequency-based metrics to identify and remove "statistical noise" from the model.

- 🔁 **Iterative Retraining**  
  See how removing noisy tokens and rebalancing the training set changes the model's predictions, performance, and top features.

Let's dive in!
 """)
