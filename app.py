import os
from dotenv import load_dotenv
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st

from src.prompt import *
from src.help import load_pdfs,recursive_text_split,vectordb


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


#Basic_operations

pdf_data = load_pdfs(os.path.join(os.getcwd(), "data"))
text_chunks = recursive_text_split(pdf_data)
documents = [Document(page_content=chunk) for chunk in text_chunks]
embeddings_model = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings_model)

#Retriver
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

#chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

import streamlit as st

# Streamlit UI Setup
st.set_page_config(page_title="AI-Powered RAG Chatbot", page_icon="🤖")
st.title("🤖 AI-Powered RAG Chatbot")
st.write("**Enhance your search with Retrieval-Augmented Generation (RAG). Ask questions with real-time knowledge retrieval.**")

# Query input box
st.subheader("💬 Ask a Question")
query = st.text_input("Type your question here...")

# Handle query submission
if st.button("🔍 Get Answer"):
    response = rag_chain.invoke({"input": query})  # Ensure rag_chain is initialized before this line
    st.subheader("📝 Response:")
    st.write(response["answer"])