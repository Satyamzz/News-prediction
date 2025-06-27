#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Optional: define the same clean_text function you used during training
def clean_text(text):
    import string
    from nltk.corpus import stopwords
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# App UI
st.title("📰 Fake News Detection App")
user_input = st.text_area("Paste the news content here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
        label = "Fake" if prediction == 0 else "Real"
        st.success(f"🧠 This news seems: **{label}**")


# In[ ]:




