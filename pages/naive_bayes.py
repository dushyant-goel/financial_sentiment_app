import streamlit as st

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
