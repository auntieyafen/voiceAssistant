"""
Contains core functionalities of LLM, including extract information from website,
add to LLM using RAG and generate response.
Source: https://github.com/y-pred/Langchain/blob/main/Langchain%202.0/RAG_Conversational_Chatbot.ipynb
"""
import os
from typing import List
import requests
import json
# For visualization
import streamlit as st
# For web scraping
import bs4
from langchain_community.document_loaders import WebBaseLoader
# For RAG
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

MAIN_URL = "https://www.tum.de/en/studies/degree-programs"

class WebScraper:
    """Class for web scraping and text processing.
    NOTE: this just works for TUM website, for others, you might need to modify
    the location in HTML that store information you want to extract.
    """    
    def __init__(self, main_url: str):
        """Initialize the WebScraper class.
        Args:
            main_url (str): The URL of the main page to scrape.
        """
        self.main_url = main_url
    
    def get_raw_page_content(self, url: str, interested_class_name: str) -> str:
        """Get raw content of the interested area in the page
        Args:
            url (str): The URL of the page to scrape.
            interested_class_name (str): the name of the class in HTML that
            contains the information that we are interested in
        """
        # Limit parsing to the specified class only
        bs4_strainer = bs4.SoupStrainer(class_=interested_class_name)
        # Load the page content
        loader = WebBaseLoader(
            web_path=url,
            bs_kwargs={"parse_only": bs4_strainer},
        )
        docs = loader.load()
        # Transfer the list of document into text
        text = ""
        for doc in docs:
            text += doc.page_content + "\n"
        text = text.strip()
        text = text.replace('\n', ' ').replace('\t', ' ')
        return text

    def get_main_info(self) -> dict:
        """Get interested information in main url, including the list of
        programs and their urls.
        Returns:
            dict: dictionary where keys are program name, values are dictionary
            contains program information
        """
        # Extract html file of the mail_url
        page = requests.get(self.main_url, timeout=10)
        soup = bs4.BeautifulSoup(page.text, 'html')
        # Find <option> item in html, which include TUM program's name and url
        options_list = soup.find_all('option')
        programs_dict = {}
        for option in options_list:
            # Find info in <option ... data-url=program_url>program_name</option>
            program_name = option.text.strip()
            program_url = option.get('data-url')
            if program_url:
                programs_dict[program_name] = {"url": f"https://www.tum.de{program_url}"}
        return programs_dict
    
    def get_program_info(self, url: str) -> dict:
        """Get interested information from the url of a program, including the
        program summary in the beginning, and Key Data
        Args:
            url (str): the url of the program
        Return:
            dict: the dictionary part in the tum program dictionary that
            corresponds to program need update
        """
        # Dictionary stores that part that need to be added to tum program dict
        update_info = {}
        # Extract html file of the mail_url
        page = requests.get(url, timeout=10)
        soup = bs4.BeautifulSoup(page.text, 'html')
        # Find program description in <p class="lead-text">program_description</p>
        program_description = soup.find("p", class_="lead-text")
        # Find key datas in <div class="flex__md-6 flex__xl-4">
        # <strong>data_name</strong><ul><li>data_values</li></ul></div>
        divs_list = soup.find_all("div", class_="flex__md-6 flex__xl-4")
        for div in divs_list:
            strong_tag = div.find("strong")
            ul_tag = div.find("ul")
            # Extract data_name
            if strong_tag:
                data_name = strong_tag.get_text(strip=True)
                data_name = data_name.replace('\n', ' ').replace('\t', ' ')
            else:
                data_name = ""
            # Extract data_values
            data_values = []
            if ul_tag:
                for li in ul_tag.find_all("li"):
                    value = li.get_text(strip=True)
                    value = value.replace('\n', ' ').replace('\t', ' ')
                    data_values.append(value)
            update_info[f"{data_name}"] = data_values
        # Add program description into updated tum program dictionary
        update_info["program description"] = program_description.text.strip()
        return update_info
    
    def get_tum_programs_dict(self) -> dict:
        """Load from or create tum_programs_dict.json"""
        file_path = "./fixed_data/tum_programs_dict.json"
        # Check if file exists and has valid content
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    if isinstance(data, dict) and data:  # Optional: make sure it's not empty
                        print("Loaded existing tum_programs_dict.json")
                        return data
            except (OSError, IOError, json.JSONDecodeError) as e:
                print(f"Error reading existing JSON file, will regenerate: {e}")
        # Otherwise, create the file
        print("Creating tum_programs_dict.json")
        programs_dict = self.get_main_info()
        for key, value in programs_dict.items():
            page_content = self.get_raw_page_content(value["url"], "flex__lg-8")
            programs_dict[key].update({"program description": page_content})
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(programs_dict, file, ensure_ascii=False, indent=4)
            print("Stored new data in ./fixed_data/tum_programs_dict.json")
        except (OSError, IOError, json.JSONDecodeError) as e:
            print(f"Error while saving the TUM programs data: {e}")
        return programs_dict
    
class RAGChatbot:
    """Prepare functions for an AI Chatbot with RAG"""
    def __init__(self, embedding_model: GoogleGenerativeAIEmbeddings,
                 llm: ChatGoogleGenerativeAI) -> None:
        """Init class RAG"""
        self.embedding_model = embedding_model
        self.llm = llm
        self.web_scrapper_class = WebScraper(MAIN_URL)
        os.makedirs("./fixed_data", exist_ok=True)

    def chunk_tum_program(self, chunk_size: int = 1000, chunk_overlap: int = 100) -> List:
        """Split the dictionary in tum_programs_dict into chunks and save/load
        from JSON.
        Args:
            chunk_size (int): the size of each chunk
            chunk_overlap (int): how much characters in the previous chunk can be
                included in the chunk (for better understanding the context)
        """
        file_path = "./fixed_data/chunked_tum_programs.json"
        # Check if file already exists and is valid
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    if (isinstance(data, dict) and "chunks_list" in data
                        and isinstance(data["chunks_list"], list) and data["chunks_list"]):
                        print("Loaded existing chunked_tum_programs.json")
                        return data
            except (OSError, IOError, json.JSONDecodeError) as e:
                print(f"Error reading existing chunked file, will regenerate: {e}")
        # Otherwise, regenerate the chunks
        print("Chunking the tum_programs_dict (the detailed version)...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        # Create tum_programs_dict if not exist
        tum_programs_dict = self.web_scrapper_class.get_tum_programs_dict()
        # Combine all program info into a single string
        programs_str = ""
        for key, value in tum_programs_dict.items():
            programs_str += f"Program: {key}\nProgram description:\n{value}\n"
        # Chunk the string
        program_chunks = text_splitter.split_text(programs_str)
        # Save to a dictionary
        program_chunks_dict = {"chunks_list": program_chunks}
        # Save to JSON
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(program_chunks_dict, file, ensure_ascii=False, indent=4)
            print("Stored new chunks in ./fixed_data/chunked_tum_programs.json")
        except (OSError, IOError, json.JSONDecodeError) as e:
            print(f"Error while saving chunked TUM programs data: {e}")
        return program_chunks_dict
    
    def get_vector_store(self, chunks_list: List) -> FAISS:
        """Create a vector store from a list of text chunks
        Args:
            chunks_list (list): = tum_program_dict.get("chunks_list")
        """
        vector_store = FAISS.from_texts(
            chunks_list, embedding=self.embedding_model)
        return vector_store
    
    def get_history_retriever(self, vector_store: FAISS) -> object:
        """Prepare the chain for chat history
        Args:
            vector_store: results of vectorizing the chunks list
        """
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Given the context and chat history, generate a search"
            "query to find out the best answer"),
            ("user", "{input}")
        ])
        retriever = vector_store.as_retriever()
        history_retriever = create_history_aware_retriever(
            self.llm, retriever, prompt)
        return history_retriever
    
    def get_conversation_rag(self, history_retriever: object) -> object:
        """Prepare the chain for conversation
        Args:
            history_retriever: result of get_history_retriever(...)
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a RAG chatbot providing advices for applicants"
            " who want to apply to TUM programs. If the question from user is"
            " too ambigious, as follow-up questions to better understand what"
            " they need from you."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        conversational_retrieval_chain = create_retrieval_chain(
            history_retriever, question_answer_chain
        )
        return conversational_retrieval_chain

    def get_response(self, user_input: str, vector_store: FAISS,
                     chat_history: list) -> str:
        """MAIN function to interact with user input
        Args:
            user_input (str): question from user
            vector_store (FAISS): the vector store of the chunks list
            chat_history (list): previous chats
        """
        history_retriever_chain = self.get_history_retriever(vector_store)
        conversation_rag_chain = self.get_conversation_rag(history_retriever_chain)
        response = conversation_rag_chain.invoke({
            "chat_history": chat_history,
            "input": user_input
        })
        return response["answer"]
