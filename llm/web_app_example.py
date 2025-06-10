"""
An example of how llm_core.py can be used. In general, take note at the code
under ATTENTION 
Source: https://github.com/y-pred/Langchain/blob/main/Langchain%202.0/RAG_Conversational_Chatbot.ipynb
"""

import os
from dotenv import load_dotenv
# For UI
import streamlit as st
# For Chatbot
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# For RAGChatbot
from llm_core import RAGChatbot

# Import Gemini API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAIN_URL = "https://www.tum.de/en/studies/degree-programs"

# TOOLS NEEDED (ATTENTION)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)
llm = ChatGoogleGenerativeAI(
        model='gemini-2.0-flash',
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
)
rag_class = RAGChatbot(embedding_model, llm)
chunks_program_dict = rag_class.chunk_tum_program()
vector_store = rag_class.get_vector_store(chunks_program_dict.get("chunks_list"))

# Add header
st.header("TUM Application AI Chatbot")
st.write("Welcome to TUM applicant chatbot, where you can ask questions " \
"about TUM programs")
# Create variables to be stored in streamlit session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, how can I help you?")
    ]
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = vector_store
# Create text field for user to enter their questions
user_input = st.chat_input("Write your question here ...")
# Once user entered the question
if user_input and user_input.strip() != "":
    # Store questions into chat_history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("Human"):
        st.markdown(user_input)
    with st.chat_message("AI"):
        with st.spinner("Thinking ..."):
            # Generate response (ATTENTION)
            response = rag_class.get_response(
                user_input,
                st.session_state.vector_store,
                st.session_state.chat_history
            )
            st.markdown(response)
    # Store response into chat_history
    st.session_state.chat_history.append(AIMessage(content=response))

