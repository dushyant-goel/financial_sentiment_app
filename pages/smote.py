import streamlit as st

import seaborn as sns
import matplotlib.pyplot as plt

from utils.model_utils import load_data

# --- SMOTE ---
st.header("⚖️ Class Imbalance & SMOTE")

st.markdown("""
Before training the classifier, let's examine the **distribution of sentiment classes** in the dataset:
""")

# Get the data
data = load_data("data/financial-news.csv")

# Class count plot
label_names = {0: "Negative", 1: "Positive", 2: "Neutral"}
data['label_name'] = data['label'].map(label_names)

fig, ax = plt.subplots()
sns.countplot(data=data, x='label_name', palette=['red', 'green', 'blue'], ax=ax)
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