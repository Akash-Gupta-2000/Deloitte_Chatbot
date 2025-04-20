import os
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Load the GROQ and OpenAI API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define the file path to save/load the FAISS index
FAISS_INDEX_PATH = "faiss_index"

# Create a FastAPI app instance
app = FastAPI()

# # Add CORS middleware to allow requests from the React app
# origins = ["http://localhost:3000"]  # Change this if your React app runs on a different port
# app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

# Define a model for the request body
class QuestionRequest(BaseModel):
    question: str

# Define the LLM and prompt
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Hardcoded list of URLs to load content from
url_list = [
]

def crawl_urls(urls):
    """Fetch and extract text from the given URLs."""
    documents = []
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator='\n')  # Extract text from the HTML
            documents.append(Document(page_content=text, metadata={"source": url}))
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
    return documents

def vector_embedding():
    """Handles vector embedding. If a stored FAISS index exists, load it, otherwise create and save one."""
    if not os.path.exists(FAISS_INDEX_PATH):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        loader = PyPDFDirectoryLoader("Documents_New/")  # Load PDFs from the specified directory
        docs = loader.load()  # Document loading
        if not docs:
            print("No documents loaded from the directory.")
        else:
            print(f"Loaded {len(docs)} PDF documents.")
            for doc in docs:
                # Print the full file path to help identify duplicates
                pdf_path = doc.metadata.get('source', 'Unknown')
                print(f"Loaded PDF from path: {pdf_path}")

        # Load and process URLs
        crawled_docs = crawl_urls(url_list)  # Fetch content from the URLs
        docs.extend(crawled_docs)  # Add crawled docs to the existing ones
        
        # Chunk creation
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)  # Split all documents
        
        # Create FAISS vector store and save to disk
        vectors = FAISS.from_documents(final_documents, embeddings)
        vectors.save_local(FAISS_INDEX_PATH)
        print("Created and saved FAISS index to disk.")

# Ensure the vector embedding is created or loaded when the app starts
vector_embedding()

def create_retrieval_chain_func():
    retriever = FAISS.load_local(
        FAISS_INDEX_PATH,
        GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
        # HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
        allow_dangerous_deserialization=True  # Allow deserialization
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Directly return the retrieval chain without using with_config
    return create_retrieval_chain(retriever.as_retriever(), document_chain)


retrieval_chain = create_retrieval_chain_func()

# Define an API endpoint for receiving questions
@app.post("/api/question")
async def answer_question(request: QuestionRequest):
    user_question = request.question
    if not user_question:
        raise HTTPException(status_code=400, detail="No question provided.")

    try:
        # Get answer from the retrieval chain
        response = retrieval_chain.invoke({'input': user_question})
        answer = response['answer']
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
