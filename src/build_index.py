from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

DATA_PATH = "Data/attention-is-all-you-need-Paper.pdf"
DB_FAISS_PATH = 'faiss_index'

def create_vector_store():
    """Creates and saves the FAISS vector store from the PDF."""
    try:
        loader = PyPDFLoader(file_path=DATA_PATH)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250,
            chunk_overlap=50
        )
        docs_splitter = splitter.split_documents(docs)
        
        print("Creating vector store...")
        vectors = FAISS.from_documents(
            documents=docs_splitter,
            embedding=HuggingFaceEmbeddings()
        )

        vectors.save_local(DB_FAISS_PATH)
        print(f"Vector store saved successfully at '{DB_FAISS_PATH}'.")
    except Exception as e:
        print(f"Error creating vector store: {e}")

if __name__ == "__main__":
    create_vector_store()