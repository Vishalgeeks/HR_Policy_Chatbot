def hr_prompt(context, query, chat_history=None):
    history_text = ""
    if chat_history:
        recent_history = chat_history[-6:]
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
    
    return f"""
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