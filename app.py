import streamlit as st
import os
import nltk

# Setup NLTK corpus

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Debugging punkt_tab issue
# nltk_dir = os.path.expanduser('~/nltk_data/tokenizers')
# st.write(os.listdir(nltk_dir))
# exists = os.path.isdir(nltk_dir)
# st.write(f"{nltk_dir} {exists}")

# Define the pages
main_page = st.Page('pages/main_page.py', title="Main Page")
dataset_summary_page = st.Page('pages/dataset_summary.py', title="Dataset Summary")
theory_naive_bayes_page = st.Page('pages/naive_bayes.py', title="Naive Bayes")
theory_text_vectorization = st.Page('pages/text_vectorization.py', title="Text Vectorization")
smote_page = st.Page("pages/smote.py", title="SMOTE")
model_page = st.Page("pages/model.py", title="Model")
metrics_page = st.Page("pages/metrics.py", title="Metrics")
try_it_yourself_page = st.Page("pages/try_it_yourself.py", title="Try It Yourself")

# Set up navigation
pg = st.navigation([main_page, 
    dataset_summary_page, 
    theory_naive_bayes_page, 
    theory_text_vectorization,
    smote_page,
    model_page,
    metrics_page,
    try_it_yourself_page
    ])


# Run the selected pages
pg.run()

# --- Footer ---
st.markdown("---")
st.markdown("""
    Built by Dushyant Goel • [Github](https://github.com/dushyant-goel) • [LinkedIn](https://www.linkedin.com/in/dusdusdushyant-goel-fintech/)   
    🎓 MSc Data Science (Financial Technology), University of Bristol 
""")
