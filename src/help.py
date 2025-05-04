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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")



#Document load

def load_pdfs(directory):
    pdf_texts = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            loader = PyPDFLoader(pdf_path)  # ✅ Use LangChain's PyPDFLoader
            documents = loader.load()
            text = "\n".join([doc.page_content for doc in documents])  # Extract text from pages
    
    return text


#Chunking

def recursive_text_split(text, chunk_size=500, chunk_overlap=50):
    """
    Recursively splits text into chunks while preserving hierarchical structure.

    Args:
    - text (str): The full text to be split.
    - chunk_size (int): Max size of each chunk.
    - chunk_overlap (int): Overlapping portion for context retention.

    Returns:
    - List of structured text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],  # Breaks at paragraph, sentence, word levels
    )

    return splitter.split_text(text)


#vector Database

embeddings_model = OpenAIEmbeddings()
def vectordb(documents):
    db = FAISS.from_documents(documents, embeddings_model)
    return db