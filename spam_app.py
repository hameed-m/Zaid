import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_message(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)

# Load model and vectorizer
with open("nb_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)


# --------- Streamlit App UI ---------
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

st.title("📧 Spam Message Classifier")
st.write("Welcome! This app detects whether a message is SPAM or VALID using a Naive Bayes model.")

st.markdown("---")

# Input text box
user_input = st.text_area("✍️ Write your message below:", height=150, placeholder="e.g. Congratulations! You’ve won a $1,000 gift card...")

if st.button("🔍 Check Message"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message to classify.")
    else:
        cleaned_text = clean_message(user_input)
        transformed_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(transformed_text)[0]

        st.markdown("### 📊 Result:")
        if prediction.lower() == "spam":
            st.error("🔴 This message is SPAM! 🚫")
        else:
            st.success("🟢 This message is VALID! ✅")

        st.markdown("---")
        st.markdown("### 🤖 Model Explanation")
        st.info("""
        This model is based on Multinomial Naive Bayes, a simple and effective classifier for spam detection.

        - Preprocessing includes text cleaning, tokenization, stop word removal, and TF-IDF vectorization.
        - Naive Bayes then uses learned word patterns to predict the category.
        - It’s a fast, lightweight approach commonly used in email spam filters.
        """)

        st.markdown("---")
        st.caption("Built for MIS 542 — Spam Classifier Project by [Your Name]")