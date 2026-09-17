import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq


# Explicitly load .env from rag directory and current directory
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)
load_dotenv()


def get_llm(api_key: str = None, model: str = "openai/gpt-oss-20b", temperature: float = 0.2):
    """
    Initialize and return ChatGroq client.
    """
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        return None

    return ChatGroq(
        model=model,
        api_key=key,
        temperature=temperature
    )


def generate_answer(query, results, api_key: str = None):
    """
    Generate an answer using context retrieved from the RAG knowledge base.
    """
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        return (
            "⚠️ **Groq API key is missing.**\n\n"
            "Please add your `GROQ_API_KEY` to the `rag/.env` file or enter it in the sidebar settings."
        )

    try:
        llm = get_llm(api_key=key)
    except Exception as e:
        return f"⚠️ **Error initializing LLM client:** {e}"

    context = "\n\n".join(
        result["text"]
        for result in results
    )

    prompt = f"""
You are an AI customer support assistant.

Your job is to answer the user's support question
using the provided knowledge base.

IMPORTANT RULES:

1. Use the knowledge base as the primary source
   for support-related questions.

2. Do not invent policies, prices, procedures,
   or company information.

3. If the requested support information is not
   available in the context, politely explain that
   the information is not available.

4. Give a clear, natural and helpful answer.

5. Do not mention "retrieved chunks", "FAISS",
   "embeddings", or internal technical details.

Knowledge Base Context:
------------------------
{context}
------------------------

User Question:
{query}

Answer:
"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"⚠️ **Error generating response from Groq:** {e}"