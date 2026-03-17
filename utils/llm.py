import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(model_name="gemini-2.5-flash"):
    from dotenv import load_dotenv
    load_dotenv()
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=os.getenv("GOOGLE_API_KEY"))
    return llm