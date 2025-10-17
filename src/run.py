import streamlit as st
from app import chatbot_response




# Streamlit Page Configuration

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)


# Title and Description

st.title("🤖 Context-Aware RAG Chatbot")
st.caption("Ask questions based on your uploaded documents or preloaded dataset.")



# Session State Initialization

if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores chat history



# Chat Display

chat_placeholder = st.container()
with chat_placeholder:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])




# User Input

user_query = st.chat_input("Ask your question here...")

# When user sends a message
if user_query:
    # Show and store the user's message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... Retrieving and generating answer..."):
            try:
                # Call the chatbot response function
                answer = chatbot_response(user_query)

                # If chatbot_res returns None or empty
                if not answer:
                    answer = "Hmm... I couldn't find relevant information in the document."

                # Display answer
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})




# Sidebar Information

with st.sidebar:
    st.header("🧠 About")
    st.markdown(
        """
        This is a **Retrieval-Augmented Generation (RAG)** chatbot built with **LangChain**, **ChromaDB**, 
        and **Streamlit**. It:
        - Loads your local documents (PDF, TXT, MD).
        - Retrieves the most relevant chunks.
        - Queries available LLMs (OpenAI, Groq, Gemini).
        - Responds with context-based answers.
        """
    )
    st.divider()
    st.caption("Built by Richmond's AI Assistant 💡")
