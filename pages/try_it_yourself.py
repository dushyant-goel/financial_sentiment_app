import streamlit as st 

import pandas as pd

from utils.model_utils import preprocess_text



if 'model' in st.session_state:
    vectorizer = st.session_state['vectorizer']
    model = st.session_state['model']

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
        st.bar_chart(pd.DataFrame(proba, index=["Negative", "Positive", "Neutral"], columns=["Probability"]))
    
else:
    st.warning("Please train the model first.")
