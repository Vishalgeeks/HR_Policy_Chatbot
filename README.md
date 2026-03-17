# 🤖 HR Policy Chatbot (RAG Based)

This project is an AI-powered chatbot that answers questions about HR policies using a PDF document.
It uses Retrieval-Augmented Generation (RAG) to retrieve relevant policy sections and generate accurate responses.

## 🚀 Features

* Chat with HR policy documents
* Vector search using ChromaDB
* Sentence-transformer embeddings
* Gemini LLM for response generation
* Conversation memory support
* Built with Streamlit UI

## 🛠 Tech Stack

* Python
* Streamlit
* LangChain
* ChromaDB
* Sentence Transformers
* Google Gemini API

## 📂 Project Structure

```
hr_chatbot/
│
├── main.py                # Streamlit chatbot UI
├── create_vectorstore.py  # Builds vector database
│
├── utils/                 # Helper modules
│   ├── embeddings.py
│   ├── retriever.py
│   ├── llm.py
│   └── prompts.py
│
├── data/
│   └── HR Policy Manual 2023.pdf
│
├── chroma_db/             # Vector database (ignored in git)
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation

1. Clone the repository

```bash
git clone https://github.com/Vishalgeeks/HR_Policy_Chatbot.git
cd HR_Policy_Chatbot
```

2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Add API key in `.env`

```
GOOGLE_API_KEY=your_api_key
```

## 📊 Build Vector Database

Run once to create embeddings:

```bash
python create_vectorstore.py
```

## ▶️ Run the Chatbot

```bash
streamlit run main.py
```

The chatbot will start at:

http://localhost:8501

## 💡 Example Questions

* What types of leaves are available?
* What is the maternity leave policy?
* What are employee benefits?

## 📌 Future Improvements

* Add streaming responses
* Hybrid search (BM25 + vector)
* Better UI with sidebar history
* Multi-document support


