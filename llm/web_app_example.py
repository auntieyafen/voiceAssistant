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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAIN_URL = "https://www.tum.de/en/studies/degree-programs"

# TOOLS NEEDED
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)
llm = ChatGoogleGenerativeAI(
        model='gemini-2.0-flash',
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
)
rag_class = RAGChatbot(embedding_model, llm)
chunks_program_dict = rag_class.chunk_tum_program()
vector_store = rag_class.get_vector_store(chunks_program_dict.get("chunks_list"))

# Add header
st.header("TUM Application AI Chatbot")
st.subheader("Welcome to TUM applicant chatbot, where you can ask questions" \
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
    # Generate response
    response = rag_class.get_response(
        user_input,
        st.session_state.vector_store,
        st.session_state.chat_history
    )
    # Store questions and response into chat_history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    else:
        with st.chat_message("Human"):
            st.markdown(message.content)
