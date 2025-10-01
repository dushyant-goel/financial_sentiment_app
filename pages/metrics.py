import streamlit as st

import pandas as pd

from utils.visual_utils import plot_confusion_matrix
from utils.model_utils import preprocess_text

st.subheader("Metrics")


# --- Metrics Display ---
def format_metrics(report):
    metrics_df = pd.DataFrame(report).T
    return metrics_df.iloc[:-3]  # remove accuracy/macro avg rows for brevity

if 'model' in st.session_state:
    report = st.session_state['report']
    matrix = st.session_state['matrix']
    model = st.session_state['model']
    vectorizer = st.session_state['vectorizer']

    st.write("**Classification Report**")
    st.dataframe(format_metrics(report), use_container_width=True)

    st.write("**Confusion Matrix**")
    plot_confusion_matrix(matrix, labels=["Negative", "Positive", "Neutral"])

else:
    st.warning("Please train the model first.")
