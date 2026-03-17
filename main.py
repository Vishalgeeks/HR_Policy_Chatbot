import streamlit as st
from utils.llm import get_llm
from utils.retriever import get_retriever
from utils.prompts import hr_prompt
from utils.embedding import get_embeddings

# ---------------- UI ----------------

st.set_page_config(page_title="HR Chatbot")
st.title(" HR Policy Chatbot")

# ---------------- Memory ----------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- Load models ----------------

embeddings = get_embeddings()
retriever = get_retriever(embeddings)
llm = get_llm()

# ---------------- User input ----------------

query = st.chat_input("Ask HR question")

if query:

    # save user message
    st.session_state.chat_history.append(
        {"role":"user","content":query}
    )

    # retrieve documents
    docs = retriever.invoke(query)

    context = "\n".join([d.page_content for d in docs])

    prompt = hr_prompt(
        context=context,
        query=query,
        chat_history=st.session_state.chat_history
    )

    response = llm.invoke(prompt)

    answer = response.content

    # store answer
    st.session_state.chat_history.append(
        {"role":"assistant","content":answer}
    )

# ---------------- Display chat ----------------

for msg in st.session_state.chat_history:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])