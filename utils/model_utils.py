import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# NLTK download check
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w.isalnum() and w not in stop_words]
    return ' '.join(filtered)

def load_data(path):
    df = pd.read_csv(path, encoding='ISO-8859-1', names=['label', 'text'])
    label_map = {'negative': 0, 'positive': 1, 'neutral': 2}
    df['label'] = df['label'].map(label_map)
    df['processed'] = df['text'].apply(preprocess_text)
    return df

def vectorize(corpus, method='bow', stop_words=None):
    if method == 'bow':
        vectorizer = CountVectorizer(stop_words=stop_words)
    else:
        vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def apply_smote(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def train_and_evaluate(X, y, apply_sampling=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if apply_sampling:
        X_train, y_train = apply_smote(X_train, y_train)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)
    return model, report, matrix

def get_top_n_words_per_class(classifier, vectorizer, class_labels, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    log_probs = classifier.feature_log_prob_
    
    top_words = {}
    for i, class_label in enumerate(class_labels):
        top_indices = np.argsort(log_probs[i])[::-1][:top_n]
        top_words[class_label] = [(feature_names[j], np.exp(log_probs[i][j])) for j in top_indices]
    return top_words
