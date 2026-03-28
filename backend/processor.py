import os
from dotenv import load_dotenv   
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
from langchain_community.vectorstores import FAISS
# 2. Load the API Key from .env file

def process_pdf(file_path: str):
    # Load the PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    # Shred into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)
    
    # THE VAULT: Save chunks into a folder called 'db'
    vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings)  # ← changed
    vector_db.save_local("./backend/db")
    
    print(f"Success! 'db' folder created in backend with {len(chunks)} chunks.")
    return chunks

# Example usage
if __name__ == "__main__":
    # Ensure you have a PDF in your uploads folder
    # Example: if your file is named 'data.pdf'
    process_pdf("./backend/uploads/Narayan_Tamrakar.pdf")