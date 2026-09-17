import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Load .env explicitly
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)
load_dotenv()

from retriever import semantic_search
from llm import generate_answer


def handle_conversation(query):
    """
    Handle simple conversational messages
    without querying the RAG knowledge base.
    """

    message = query.lower().strip()

    greetings = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "hi there",
        "hello there"
    ]

    thanks = [
        "thanks",
        "thank you",
        "thanks a lot",
        "thank you so much",
        "thx"
    ]

    goodbye = [
        "bye",
        "goodbye",
        "see you",
        "see you later",
        "talk to you later"
    ]

    if message in greetings:
        return "Hello! 👋 How can I help you with your support question?"

    if message in thanks:
        return "You're very welcome! 😊 I'm happy to help."

    if message in goodbye:
        return "You're welcome! 👋 Have a great day!"

    return None


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Support Assistant",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")
    
    current_key = os.getenv("GROQ_API_KEY", "")
    api_key_input = st.text_input(
        "Groq API Key",
        value=current_key,
        type="password",
        help="Get your key at https://console.groq.com/keys"
    )
    
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input
        st.success("API Key set! ✅")
    else:
        st.info("💡 Enter your Groq API key here or define it in `rag/.env`.")

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")
    st.caption("Documents: Cancellation, Privacy, Refund policies, & Troubleshooting.")


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🤖 AI Support Assistant")

st.write(
    "Ask questions about our policies, payments, "
    "accounts, and troubleshooting."
)


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# USER QUESTION
# --------------------------------------------------

query = st.chat_input("Ask your support question...")

if query:
    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Check for conversational message
    conversational_response = handle_conversation(query)

    # If conversational → don't use RAG
    if conversational_response:
        answer = conversational_response
        with st.chat_message("assistant"):
            st.markdown(answer)

    # Otherwise → use RAG
    else:
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching knowledge base..."):
                    results = semantic_search(
                        query,
                        top_k=3
                    )

                if not results:
                    answer = (
                        "I'm an AI customer support "
                        "assistant, so I can help with "
                        "questions related to our support "
                        "knowledge base. 😊"
                    )
                    st.markdown(answer)

                else:
                    with st.spinner("Generating answer..."):
                        answer = generate_answer(
                            query,
                            results,
                            api_key=api_key_input or os.getenv("GROQ_API_KEY")
                        )

                    st.markdown(answer)

                    # Sources
                    with st.expander("📚 Sources"):
                        for i, result in enumerate(results, start=1):
                            metadata = result.get("metadata", {})
                            source = metadata.get("source", "Unknown")
                            page = metadata.get("page", "N/A")
                            score = result.get("score", 0)

                            st.write(f"**{i}. {source}**")
                            st.caption(f"Page: {page} | Similarity: {score:.4f}")

            except Exception as e:
                answer = "⚠️ Something went wrong while processing your question."
                st.error(answer)
                st.exception(e)

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )