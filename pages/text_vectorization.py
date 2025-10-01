import streamlit as st

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
