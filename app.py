import streamlit as st
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ----------------- Load API key -----------------
load_dotenv()

# ----------------- Chat memory -----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_cache" not in st.session_state:
    st.session_state.qa_cache = {}

# ----------------- Vector Store -----------------
@st.cache_resource
def load_vectorstore():
    loader = PyPDFLoader("HR Policy Manual 2023.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    if vectorstore._collection.count() == 0:
        vectorstore.add_documents(chunks)
        vectorstore.persist()

    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# ----------------- LLM -----------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

llm = load_llm()

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="HR Policy Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 HR Policy Chatbot")

# 1️⃣ Search box always at the top
query = st.chat_input("Ask your HR question here...")

# 2️⃣ Process query if user submits
if query:
    query = query.strip()
    if query:
        # store user input
        st.session_state.chat_history.append({"role": "user", "content": query})

        # retrieve relevant chunks
        docs = retriever.invoke(query)
        context = "\n".join([d.page_content for d in docs]) if docs else ""

        recent_history = st.session_state.chat_history[-6:]
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])

        prompt = f"""
You are an HR assistant.

Previous conversation:
{history_text}

Answer ONLY using the HR policy context below.
If the answer is not present in the policy, say:
"The HR policy does not specify this."

Context:
{context}

Question:
{query}
"""

        # Use cache if available
        if query in st.session_state.qa_cache:
            answer = st.session_state.qa_cache[query]
        else:
            response = llm.invoke(prompt)
            answer = response.content
            st.session_state.qa_cache[query] = answer

        # store assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# 3️⃣ Display the full conversation in chat bubbles (no duplication)
for msg in st.session_state.chat_history:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.write(msg["content"])