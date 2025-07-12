# 📰 Financial News Sentiment Analysis App

This interactive Streamlit app classifies financial news into **positive**, **negative**, or **neutral** sentiment using a **Naïve Bayes classifier** trained on real-world headlines.

Learn how text is converted to features using **Bag-of-Words** and **TF-IDF**, tackle class imbalance with **SMOTE**, and experiment with **feature noise removal** using entropy metrics.

---

## 🔍 Features

- 📐 Mathematical foundation of Naïve Bayes
- 🔢 Text vectorization: BoW and TF-IDF
- ⚖️ SMOTE-based class rebalancing
- 📊 Top words by class, confusion matrix, performance metrics
- 🧹 Interactive filtering of statistically noisy tokens
- 🔁 Retraining and comparison after token pruning

---

## 🚀 Try It Online

👉 [Launch the app on Streamlit Cloud](https://share.streamlit.io/your-username/sentiment-finance-app/main/app.py)

---

## 🛠️ Run Locally

```bash
# Clone repo and install dependencies
git clone https://github.com/your-username/sentiment-finance-app.git
cd sentiment-finance-app
pip install -r requirements.txt

# Start the app
streamlit run app.py

---

## 📚 Dataset

Source: [kaggle/ankurzing](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)