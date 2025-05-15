import streamlit as st
import json  # Ensure JSON is imported
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load Data and Model
dataset_path = "medquads.json"
with open(dataset_path, "r", encoding="utf-8") as file:
    data = json.load(file)

df = pd.DataFrame({"question": [item["question"] for item in data], "answer": [item["answer"] for item in data]})

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("tfidf_matrix.pkl", "rb") as matrix_file:
    X = pickle.load(matrix_file)

# Function to Retrieve Answers
def get_answer(user_query, threshold=0.3):
    user_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_vec, X).flatten()
    
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]
    
    if best_score < threshold:
        return "Sorry, I couldn't find an exact answer. Please try rephrasing your question."
    
    return df.iloc[best_match_idx]["answer"]

# Streamlit UI
st.title("Medical Q&A Chatbot 🤖")
st.write("Ask me medical questions based on the MedQuAD dataset!")

user_question = st.text_input("Enter your medical question:")

if st.button("Get Answer"):
    if user_question:
        response = get_answer(user_question)
        st.write("**Answer:**", response)
    else:
        st.write("Please enter a question.")
