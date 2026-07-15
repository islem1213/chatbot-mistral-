import os
import json
import glob
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_github_docs(github_dir: str) -> List[Document]:
    """
    Load raw GitHub JSON files and convert to LangChain Documents.
    """
    documents = []
    pattern = os.path.join(github_dir, "*.json")
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            name = data.get("name", "")
            description = data.get("description", "")
            stars = data.get("stars", 0)
            last_activity = data.get("last_activity", "")
            url = data.get("url", "")
            readme = data.get("readme", "")
            
            # Format document content with metadata header
            content = (
                f"Repository: {name}\n"
                f"Description: {description}\n"
                f"Stars: {stars} | Last activity: {last_activity}\n"
                f"URL: {url}\n\n"
                f"README Content:\n{readme}"
            )
            
            metadata = {
                "source": url,
                "title": name,
                "type": "github",
                "stars": stars
            }
            
            # Skip if readme is empty
            if readme.strip():
                documents.append(Document(page_content=content, metadata=metadata))
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            
    print(f"Loaded {len(documents)} GitHub repository documents.")
    return documents

def load_scrape_docs(docs_dir: str) -> List[Document]:
    """
    Load scraped documentation JSON files and convert to LangChain Documents.
    """
    documents = []
    pattern = os.path.join(docs_dir, "*.json")
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            title = data.get("title", "")
            url = data.get("url", "")
            content = data.get("content", "")
            
            doc_content = (
                f"Documentation Page: {title}\n"
                f"URL: {url}\n\n"
                f"{content}"
            )
            
            metadata = {
                "source": url,
                "title": title,
                "type": "documentation"
            }
            
            if content.strip():
                documents.append(Document(page_content=doc_content, metadata=metadata))
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            
    print(f"Loaded {len(documents)} documentation page documents.")
    return documents

def build_vector_store(
    github_dir: str = "data/github",
    docs_dir: str = "data/docs",
    persist_directory: str = "chroma_db",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Chunck documents, compute embeddings, and build the local Chroma DB.
    """
    # 1. Load documents
    github_docs = load_github_docs(github_dir) if os.path.exists(github_dir) else []
    scrape_docs = load_scrape_docs(docs_dir) if os.path.exists(docs_dir) else []
    all_docs = github_docs + scrape_docs
    
    if not all_docs:
        print("No documents found to index! Please run ingestion first.")
        return
        
    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(all_docs)
    print(f"Split {len(all_docs)} documents into {len(chunks)} chunks.")
    
    # 3. Create/load Embeddings model
    print(f"Loading embedding model: {embedding_model_name}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'} # Force cpu to avoid CUDA issues
    )
    
    # 4. Build Chroma Store
    print(f"Initializing local Chroma DB at: {persist_directory}...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Successfully built and persisted Chroma DB with {len(chunks)} chunks.")

if __name__ == "__main__":
    build_vector_store()
