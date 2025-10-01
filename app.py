import streamlit as st
import nltk

# Setup NLTK corpus

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define the pages
main_page = st.Page('pages/main_page.py', title="Main Page")
dataset_summary_page = st.Page('pages/dataset_summary.py', title="Dataset Summary")
theory_naive_bayes_page = st.Page('pages/naive_bayes.py', title="Naive Bayes")
theory_text_vectorization = st.Page('pages/text_vectorization.py', title="Text Vectorization")
smote_page = st.Page("pages/smote.py", title="SMOTE")

# Set up navigation
pg = st.navigation([main_page, 
    dataset_summary_page, 
    theory_naive_bayes_page, 
    theory_text_vectorization,
    smote_page,
    ])

# Run the selected pages
pg.run()

# # ---- Model Configuration ----
# st.header("⚙️ Model Configuration")

# st.markdown("""
# Before we train the model, choose your desired settings:
# """)

# col1, col2 = st.columns(2)

# with col1:
#     vectorizer_type = st.radio(
#         "Vectorization Method:",
#         ["Bag of Words", "TF-IDF"],
#         index=0,
#         key="vectorizer_choice"
#     )

# with col2:
#     use_smote = st.radio(
#         "Apply SMOTE for Class Balancing?",
#         ["No", "Yes"],
#         index=0,
#         key="smote_choice"
#     ) == "Yes"

# "---"

# from utils.model_utils import (
#     load_data, preprocess_text, vectorize,
#     train_and_evaluate, get_top_n_words_per_class,
# )
# from utils.visual_utils import plot_confusion_matrix
# from utils.diagnostics_utils import identify_noise_words

# data = []
# @st.cache_resource
# def setup_nltk():
#     # NLTK download check
#     nltk.download('punkt')
#     nltk.download('stopwords')
#     nltk.download('wordnet')
#     nltk.download('punkt_tab')

#     for resource, path in [
#         ("punkt", "tokenizers/punkt"),
#         ("stopwords", "corpora/stopwords"),
#         ("wordnet", "corpora/wordnet")
#     ]:
#         try:
#             nltk.data.find(path)
#             data = load_data("data/financial-news.csv")
#         except LookupError:
#             nltk.download(resource)



# # --- Train model on dataset ---

# with st.spinner("Training the model..."):
#     X, vectorizer = vectorize(
#         data['processed'], method='bow' if vectorizer_type == "Bag of Words" else 'tfidf')
#     model, report, matrix = train_and_evaluate(
#         X, data['label'], apply_sampling=use_smote)


# # --- Visualizations ---

# st.subheader("🧠 What Words Influence Predictions?")

# st.markdown(
#     "The top words per class are shown below based on model feature weights:")

# top_words = get_top_n_words_per_class(model, vectorizer, class_labels=[
#                                       "Negative", "Positive", "Neutral"])

# col_neg, col_pos, col_neu = st.columns(3)

# with col_neg:
#     st.subheader("🟥 Negative Words")
#     df_neg = pd.DataFrame(top_words["Negative"], columns=[
#                           "Word", "Estimated Probability"])
#     st.dataframe(df_neg, use_container_width=True)

# with col_pos:
#     st.subheader("🟩 Positive Words")
#     df_pos = pd.DataFrame(top_words["Positive"], columns=[
#                           "Word", "Estimated Probability"])
#     st.dataframe(df_pos, use_container_width=True)

# with col_neu:
#     st.subheader("🟦 Neutral Words")
#     df_neu = pd.DataFrame(top_words["Neutral"], columns=[
#                           "Word", "Estimated Probability"])
#     st.dataframe(df_neu, use_container_width=True)


# st.header("🧹 Noise Word Filter Tool")

# st.markdown("#### Why do some common words dominate top predictors?")
# st.markdown(r"""
# Some words like "EUR" or "mn" appear in both lists.  
            
# - “Company X posts EUR 20 mn loss” → Negative
# - “Company Y secures EUR 50 mn investment” → Positive
            
# The dominance weight of "EUR" and "mn" is mitigated to some extnent from BoW ~(0.03-0.05) range to 
# TF-IDF (~0.01). 

# Their co-occurrence with strong sentiment words gives them statistical importance across classes.
# This motivates us use entropy and class variance metrics to flag such **statistically uninformative** words.
            
# Let's make an effort to clean our corpus from these.

# We define,  
#     $ \text{Normalized Entropy}(w) = - \frac{\sum_{c} P(c \mid w) logP(c \mid w)}{log(\#\text{classes})} $    
#     $ \text{Document Frequency}(w) = \frac{\#\text{docs with w}}{\text{total docs}}$

# - For 'mn' and 'eur' the entropies are 0.777 and 0.914. This indicates that 'eur' appears in evenly across classes 
# and is statistically insignificant. 'mn' less so but is distributed fairly evenly.
# - For 'mn' and 'eur' the document frequencies 0.058 and 0.129. This means 'mn' is not very common appearing in only 
# 5.8% of the docuemnts and 'eur' in only 12.9% of the documents. 
            
# - The maximum document frequency of any word is 0.186 for 'company', followed by 'eur', 'said (0.112)' and 'finish (0.104)'
            
# Adjust the sliders below to set the minimum entropy and document frequency
# """)

# threshold_entropy = st.slider(
#     "Minimum Normalized Entropy", 0.75, 1.0, 0.85, 0.01)
# threshold_df = st.slider("Minimum Document Frequency", 0.10, 0.20, 0.13, 0.01)

# noise_stats, candidates = identify_noise_words(model, vectorizer, X, data['label'],
#                                                threshold_entropy=threshold_entropy,
#                                                threshold_docfreq=threshold_df, debug=False)

# st.subheader("📋 Candidate Stopwords")
# st.dataframe(candidates[["Word", "EntropyRatio", "DocFreq", "LogProbVar"]])

# "---"

# if st.button("🔁 Retrain Without These Words"):

#     custom_stopwords = list(candidates["Word"])
#     # Vectorize again with updated stopword list
#     X, vectorizer = vectorize(data['processed'], method='tfidf' if vectorizer_type == 'TF-IDF' else 'bow',
#                               stop_words=custom_stopwords)
#     model, report, matrix = train_and_evaluate(
#         X, data['label'], apply_sampling=use_smote)
#     st.success("Retrained model with custom stopwords removed!")
#     # st.text(report_filtered)

#     st.subheader("🔁 Updated Top Words After Filtering")

#     top_words = get_top_n_words_per_class(model, vectorizer, class_labels=[
#                                           "Negative", "Positive", "Neutral"])

#     col_neg, col_pos, col_neu = st.columns(3)

#     with col_neg:
#         st.subheader("🟥 Negative Words")
#         df_neg = pd.DataFrame(top_words["Negative"], columns=[
#                               "Word", "Estimated Probability"])
#         st.dataframe(df_neg, use_container_width=True)

#     with col_pos:
#         st.subheader("🟩 Positive Words")
#         df_pos = pd.DataFrame(top_words["Positive"], columns=[
#                               "Word", "Estimated Probability"])
#         st.dataframe(df_pos, use_container_width=True)

#     with col_neu:
#         st.subheader("🟦 Neutral Words")
#         df_neu = pd.DataFrame(top_words["Neutral"], columns=[
#                               "Word", "Estimated Probability"])
#         st.dataframe(df_neu, use_container_width=True)


# "---"

# st.subheader("Metrics")

# # --- Metrics Display ---


# def format_metrics(report):
#     metrics_df = pd.DataFrame(report).T
#     return metrics_df.iloc[:-3]  # remove accuracy/macro avg rows for brevity


# st.write("**Classification Report**")
# st.dataframe(format_metrics(report), use_container_width=True)

# st.write("**Confusion Matrix**")
# plot_confusion_matrix(matrix, labels=["Negative", "Positive", "Neutral"])

# # --- User Input Section ---
# st.header("📰 Try It Yourself")
# user_text = st.text_area("Enter your financial news sentence:", "")

# if user_text:
#     processed = preprocess_text(user_text)
#     input_vector = vectorizer.transform([processed])
#     prediction = model.predict(input_vector)[0]
#     proba = model.predict_proba(input_vector)[0]

#     class_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
#     st.success(f"**Prediction:** {class_map[prediction]}")
#     st.write("**Class Probabilities**")
#     st.bar_chart(pd.DataFrame(
#         proba, index=["Negative", "Positive", "Neutral"], columns=["Probability"]))

# # --- Footer ---
# st.markdown("---")
# st.markdown("""
#             Built by Dushyant Goel • [Github](https://github.com/dushyant-goel) • [LinkedIn](https://www.linkedin.com/in/dusdusdushyant-goel-fintech/)   
#             🎓 MSc Data Science (Financial Technology), University of Bristol 
#             """)
