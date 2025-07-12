import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# NLTK download check
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from utils.model_utils import (
    load_data, preprocess_text, vectorize,
    train_and_evaluate, get_top_n_words_per_class
)
from utils.visual_utils import plot_confusion_matrix
from utils.diagnostics_utils import identify_noise_words

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

# --- Load and preprocess dataset ---
st.header("📊 Dataset Summary")

st.markdown("""
The dataset is sourced from [kaggle/ankurzing](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) and contains financial news headlines labeled by sentiment.

Each headline is categorized as:
- **Positive (1)**
- **Negative (0)**
- **Neutral (2)**

The goal is to classify unseen financial news into one of these categories based on the textual content.
The dataset contains **rows of short news headlines**, which are typically more ambiguous and compact in language than product reviews or articles, 
making the classification noisy. While some headlines are clearly positive or negative, there is always subjectivity involved in labeled data.
""")

data = load_data("data/financial-news.csv")
st.write(data.head(5))
st.write(f"Dataset contains {data.shape[0]} samples.")

"---"

# --- Theory ---

st.header("📐 Theory: Multinomial Naïve Bayes")

st.markdown("""
We model the probability of a class $c \in \{0, 1, 2\}$ given a document $d$ using Bayes' Theorem.
We want to find a class $\hat{c}$ = ${argmax}_{c \in \{0, 1, 2\}}P( c \mid d)$. 
From Naïve Bayes Theorme
""")

st.latex(r"""
P(c \mid d) = \frac{P(d \mid c) \cdot P(c)}{P(d)}
""")

st.markdown("""
Here:
- $P(c)$ is the prior probability of the class,
- $P(d \mid c)$ is the likelihood of the document given the class,
- $P(d)$ is the probability of document,
- We choose the class $c$ that **maximizes** this posterior.

""")

st.markdown(r"""
The term $\frac{1}{P(d)}$ is constant. Hence we maximize 
$P(d \mid c) \cdot P(c)$.

            
For some features $f_{i}$ extracted from the words of the document, 
we can write :
""")

st.latex(r"""
         P(d \mid c) = P(f_1, f_2, \cdots, f_n \mid c)
""")

st.markdown("""
Where:
- $f_i$ is the $i$-th feature in the document, and the feature vector $F$ is the 
input to the classifier model.  

Further, we use the Naïve Bayes assumption, that the features $f_i$ are independent of each
other, i.e.  

""")

st.latex(r"""
P(d \mid c) = P(f_1 \mid c) \cdot P(f_2 \mid c) \cdots P(f_n \mid c)  \\
P(d \mid c) = \prod_{i=1}^{n} P(f_i \mid c)
""")

st.header("Text Vectorization")

st.subheader("Bag of Words")

st.markdown("""
    In **Bag of Words** (BoW), the count of each word is treated as a feature. 
    The document becomes a vector of word counts. The logic behind this is that 
    words "windfall", "good", "profit" indicate a positive sentiment, irrespective
    of where they appear in the document. The more frequently they appear, the stronger the
    sentiment.
            
    From the labeled data, the classifier learns which words are associated with positive, neutral
    and negative sentiment. 
            
    Bag of Word vectorization does **not consider order**, it associates **word count** with sentiment
    class label. Mathematically,
            
""")

st.latex(r"""
    P(f_i \mid c) = \frac{{count}(w_i, c) + 1}{\sum_{w \in |V|} {count}(w_j, c) + 1}
""")

st.markdown(r"""
Where:
- ${count}(w_i, c)$, is count of w_i for documents in class $c$. 
- $\sum_{w \in |V|}{count}(w_j, c)$, is count of all words in vocubulary in class $c$.
- $|V|$ is the vocabulary size.
            
We add $1$ to both numerator and denominator, called "Laplace Smoothening", to avoid `divide by 0` error 
during computation.
""")

st.subheader("TF-IDF")

st.markdown("""
In **TF-IDF** (Term Frequency-Inverse Document Frequency), differs from Bag of Words by, 
down-weighting common words and up-weighting rare but informative ones. 
The idea is that words like "the", "market", or "shares" appear frequently 
in financial news across all sentiments and may not help in distinguishing 
between classes.

On the other hand, words like "slump", "surge", or "litigation" may occur 
less frequently but signal strong sentiment. TF-IDF captures this intuition.

The weight for each word in a document is computed as:
""")

st.latex(r"""
\text{tfidf}(w, d) = tf(w, d) \cdot \log\left(\frac{N}{df(w)}\right)
""")

st.markdown(r"""
Where:
- $tf(w, d)$ is the **term frequency** of word $w$ in document $d$.
- $df(w)$ is the **document frequency** — number of documents in which $w$ appears.
- $N$ is the total number of documents.

$tf(w, d)$, captures how important the word is **within the document**.
The second part, $\log\left(\frac{N}{df(w)}\right)$, captures how **discriminative** 
the word is across documents.

TF-IDF thus assigns high weight to words that are:
- **frequent in one document**, and
- **rare across the corpus**

The resulting feature vector is **normalized** before being passed 
into the classifier. This ensures that longer documents don't get unfair advantage
due to higher word counts.
""")

# --- SMOTE ---
st.header("⚖️ Class Imbalance & SMOTE")

st.markdown("""
Before training the classifier, let's examine the **distribution of sentiment classes** in the dataset:
""")

# Class count plot

label_names = {0: "Negative", 1: "Positive", 2: "Neutral"}
data['label_name'] = data['label'].map(label_names)

fig, ax = plt.subplots()
sns.countplot(data=data, x='label_name', palette=[
              'red', 'green', 'blue'], ax=ax)
ax.set_title("Distribution of Sentiment Labels")
st.pyplot(fig)

st.markdown("""
We observe a significant **class imbalance**:

- 🟥 Negative: 110 samples  
- 🟩 Positive: 289 samples  
- 🟦 Neutral: 571 samples  

This imbalance can **bias the classifier** toward predicting the dominant class ("Neutral"),
leading to poor recall and F1 scores for minority classes — especially Negative, which is often
critical in financial decision-making.

To address this, we can use **SMOTE** (*Synthetic Minority Over-sampling Technique*), which:
- Generates synthetic examples of minority class samples,
- Balances the dataset without simple duplication,
- Helps the classifier learn meaningful patterns in underrepresented classes.

You may toggle SMOTE below to see its effect on the performance of the classifier.
            
🔍 **How are synthetic minority samples generated?**  
Ping me on [LinkedIn](https://www.linkedin.com/in/dusdusdushyant-goel-fintech/) and I'll be happy to explain with examples.  
""")

# ---- Model Configuration ----
st.header("⚙️ Model Configuration")

st.markdown("""
Before we train the model, choose your desired settings:
""")

col1, col2 = st.columns(2)

with col1:
    vectorizer_type = st.radio(
        "Vectorization Method:",
        ["Bag of Words", "TF-IDF"],
        index=0,
        key="vectorizer_choice"
    )

with col2:
    use_smote = st.radio(
        "Apply SMOTE for Class Balancing?",
        ["No", "Yes"],
        index=0,
        key="smote_choice"
    ) == "Yes"

"---"

# --- Train model on dataset ---

with st.spinner("Training the model..."):
    X, vectorizer = vectorize(
        data['processed'], method='bow' if vectorizer_type == "Bag of Words" else 'tfidf')
    model, report, matrix = train_and_evaluate(
        X, data['label'], apply_sampling=use_smote)


# --- Visualizations ---

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


st.header("🧹 Noise Word Filter Tool")

st.markdown("#### Why do some common words dominate top predictors?")
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
    $ Normalized Entropy(w) = - \frac{\sum_{c} P(c \mid w) logP(c \mid w)}{log(\#classes)} $
    $ Document Frequency(w) = \frac{\#docs with w}{total docs}$

- For 'mn' and 'eur' the entropies are 0.777 and 0.914. This indicates that 'eur' appears in evenly across classes 
and is statistically insignificant. 'mn' less so but is distributed fairly evenly.
- For 'mn' and 'eur' the document frequencies 0.058 and 0.129. This means 'mn' is not very common appearing in only 
5.8% of the docuemnts and 'eur' in only 12.9% of the documents. 
            
- The maximum document frequency of any word is 0.186 for 'company', followed by 'eur', 'said (0.112)' and 'finish (0.104)'
            
Adjust the sliders below to set the minimum entropy and document frequency
""")

threshold_entropy = st.slider(
    "Minimum Normalized Entropy", 0.75, 1.0, 0.85, 0.01)
threshold_df = st.slider("Minimum Document Frequency", 0.10, 0.20, 0.13, 0.01)

noise_stats, candidates = identify_noise_words(model, vectorizer, X, data['label'],
                                               threshold_entropy=threshold_entropy,
                                               threshold_docfreq=threshold_df, debug=False)

st.subheader("📋 Candidate Stopwords")
st.dataframe(candidates[["Word", "EntropyRatio", "DocFreq", "LogProbVar"]])

"---"

if st.button("🔁 Retrain Without These Words"):

    custom_stopwords = list(candidates["Word"])
    # Vectorize again with updated stopword list
    X, vectorizer = vectorize(data['processed'], method='tfidf' if vectorizer_type == 'TF-IDF' else 'bow',
                              stop_words=custom_stopwords)
    model, report, matrix = train_and_evaluate(
        X, data['label'], apply_sampling=use_smote)
    st.success("Retrained model with custom stopwords removed!")
    # st.text(report_filtered)

    st.subheader("🔁 Updated Top Words After Filtering")

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


"---"

st.subheader("Metrics")

# --- Metrics Display ---


def format_metrics(report):
    metrics_df = pd.DataFrame(report).T
    return metrics_df.iloc[:-3]  # remove accuracy/macro avg rows for brevity


st.write("**Classification Report**")
st.dataframe(format_metrics(report), use_container_width=True)

st.write("**Confusion Matrix**")
plot_confusion_matrix(matrix, labels=["Negative", "Positive", "Neutral"])

# --- User Input Section ---
st.header("📰 Try It Yourself")
user_text = st.text_area("Enter your financial news sentence:", "")

if user_text:
    processed = preprocess_text(user_text)
    input_vector = vectorizer.transform([processed])
    prediction = model.predict(input_vector)[0]
    proba = model.predict_proba(input_vector)[0]

    class_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
    st.success(f"**Prediction:** {class_map[prediction]}")
    st.write("**Class Probabilities**")
    st.bar_chart(pd.DataFrame(
        proba, index=["Negative", "Positive", "Neutral"], columns=["Probability"]))

# --- Footer ---
st.markdown("---")
st.markdown("""
            Built by Dushyant Goel • [Github](https://github.com/dushyant-goel) • [LinkedIn](https://www.linkedin.com/in/dusdusdushyant-goel-fintech/)
            MSc Data Science (Financial Technology), University of Bristol🎓
            """)
