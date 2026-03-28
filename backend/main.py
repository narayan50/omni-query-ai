from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from dotenv import load_dotenv
\
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

app = FastAPI()

# Initialize embeddings once and reuse
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "backend/uploads"
DB_DIR = "backend/db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "Hey Narayan! Omni-Query-AI is fully operational!"}

@app.post("/upload")
async def upload_and_process(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)

    vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vector_db.save_local(DB_DIR)

    return {"filename": file.filename, "msg": "File indexed and ready for querying!"}
# Note: The /query endpoint is designed to retrieve relevant chunks from the FAISS vector store and use them as context for the ChatGroq model to generate an answer. Make sure to have the FAISS index created before querying.
@app.post("/query")
def ask_ai(request: QueryRequest):
    vector_db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

    llm = ChatGroq(model="llama-3.3-70b-versatile")

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant. Use the following context to answer the question.
    If you don't know the answer, just say you don't know.

    Context: {context}
    Question: {question}

    Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": vector_db.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(request.query)
    return {"answer": answer}
