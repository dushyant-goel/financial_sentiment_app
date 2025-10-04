import streamlit as st

from utils.model_utils import (
    preprocess_text, 
    vectorize,
    get_topn_words
)

st.header("Text Vectorization")
st.write('Text vectorization is a crucial step in natural language processing, to machine learning models to work with human language. ' \
'Since models can only process numerical data, it\'s necessary to convert text into numbers before analysis. ' \
'There are many approaches to representing text, but two of the most commonly used methods are Bag of Words and TF-IDF (Term Frequency-Inverse Document Frequency). ' \
'Bag of Words simply counts how often each word appears, while TF-IDF weighs words by their importance across all documents.' \
'You can click the provided buttons to explore the mathematical foundations and the intuition behind each method in more detail')


with st.expander("Bag of Words"):
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

with st.expander("TF-IDF"):

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

"---"

st.subheader("🔍 Try it Yourself")
st.markdown(r"""
Enough of Maths! Let's try this out yourself. Paste your document text in the area provided,
select a vectorization technique from the radio button, and see the calculated word frequency for the doucment.
""")

docs = st.text_area("Enter the documents, one on each line:", key="user_doc", placeholder="""Paste any document, one per line""")
vectorizer_selection = st.radio("Vectorization Method:", 
                           ["Bag of Words", "TF-IDF"], 
                           index=0,
                            key="vectorizer_choice"
    )

with st.spinner("Vectorizing the text..."):
    docs = docs.splitlines()
    docs = [preprocess_text(doc) for doc in docs]
    
    try:
        if vectorizer_selection == 'Bag of Words':
            method = 'bow'
        elif vectorizer_selection == 'TF-IDF':
            method = 'tfidf'
        
        X, vectorizer = vectorize(docs, method=method)
        top_words = get_topn_words(vectorizer, X[0], 5)

        st.write("Top 5 words in the 1st document:")
        st.write(top_words)
    except:
        st.warning("Please ensure each lines contains a valid document.")






