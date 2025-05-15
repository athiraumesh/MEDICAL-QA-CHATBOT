# 🧠 Medical Q&A Chatbot

A simple TF-IDF based chatbot that answers medical questions using the MedQuAD dataset.

---

## 🚀 Features

- Ask medical questions.
- Get answers from a trusted dataset (MedQuAD).
- Built with Python, Scikit-learn, and Streamlit.
- Lightweight and offline-compatible.

---

## 🛠️ Project Structure

```
medical_qa_model/
├── app.py                  # Streamlit UI
├── model_training.ipynb    # Jupyter notebook to train TF-IDF model
├── load_data.py            # Loads and processes MedQuAD data
├── medquads.json           # Dataset used (MedQuAD)
├── tfidf_matrix.pkl        # Trained TF-IDF matrix
├── tfidf_vectorizer.pkl    # Trained vectorizer
├── requirements.txt        # Required packages
└── README.md               # Project documentation
```

---

## 📦 Installation

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/medical_qa_model.git
cd medical_qa_model
```

### ✅ Step 2: Create and Activate Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate
```

### ✅ Step 3: Install Requirements

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Chatbot

```bash
streamlit run app.py
```

---

## 📊 Dataset Used

- [MedQuAD Dataset](https://www.nlm.nih.gov/databases/download/medquad.html) - Trusted medical Q&A dataset.

---

## 🙋‍♀️ Example Questions

- What are the symptoms of diabetes?
- How can asthma be treated?

---

## 📌 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

Athira V Umesh, Thrissur 🌴"# medical_qa_model" 
