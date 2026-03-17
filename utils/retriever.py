from langchain_community.vectorstores import Chroma

def get_retriever(embedding_function, persist_dir="chroma_db", k=5):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_function)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever