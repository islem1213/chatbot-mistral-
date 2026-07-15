import os
from typing import Dict, Any, List
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

class MistralRAGAgent:
    def __init__(
        self,
        persist_directory: str = "chroma_db",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "mistral-large-latest"
    ):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.llm_model = llm_model
        
        # Verify MISTRAL_API_KEY
        if not os.getenv("MISTRAL_API_KEY"):
            print("WARNING: MISTRAL_API_KEY environment variable is not set. API calls will fail.")
            
        print("Loading embedding model for retrieval...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'}
        )
        
        if os.path.exists(self.persist_directory):
            print(f"Loading Chroma DB from {self.persist_directory}...")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print(f"Chroma DB directory {self.persist_directory} not found! You must build the index first.")
            self.vector_store = None
            
        print(f"Initializing Mistral LLM ({self.llm_model})...")
        # ChatMistralAI automatically reads MISTRAL_API_KEY from environment
        self.llm = ChatMistralAI(
            model=self.llm_model,
            temperature=0.2
        )
        
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Query the RAG agent.
        Returns:
            Dict containing 'answer' and a list of 'sources'.
        """
        if not self.vector_store:
            return {
                "answer": "Erreur : La base vectorielle n'est pas initialisée. Veuillez d'abord ingérer et indexer les données.",
                "sources": []
            }
            
        # 1. Retrieve relevant chunks
        try:
            retrieved_docs = self.vector_store.similarity_search(question, k=k)
        except Exception as e:
            return {
                "answer": f"Erreur lors de la recherche vectorielle : {e}",
                "sources": []
            }
            
        if not retrieved_docs:
            return {
                "answer": "Aucun document pertinent n'a été trouvé dans la base vectorielle locale pour répondre à votre question.",
                "sources": []
            }
            
        # 2. Format Context
        context_str = ""
        for i, doc in enumerate(retrieved_docs):
            context_str += f"--- Document {i+1} ---\n"
            context_str += f"Title: {doc.metadata.get('title', 'Unknown')}\n"
            context_str += f"Source URL: {doc.metadata.get('source', '')}\n"
            context_str += f"Content:\n{doc.page_content}\n\n"
            
        # 3. Prepare sources metadata
        sources = []
        seen_urls = set()
        for doc in retrieved_docs:
            url = doc.metadata.get("source", "")
            title = doc.metadata.get("title", "")
            doc_type = doc.metadata.get("type", "unknown")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "title": title,
                    "url": url,
                    "type": doc_type
                })
                
        # 4. Formulate System & Human Prompt
        system_prompt = (
            "Vous êtes un agent de Q&A (RAG) spécialisé et expert sur l'écosystème Mistral AI.\n"
            "Votre rôle est d'aider les développeurs et utilisateurs en répondant de manière précise et professionnelle à leurs questions.\n"
            "Répondez à la question en vous basant UNIQUEMENT sur le contexte fourni ci-dessous. Ne faites aucune supposition et n'inventez rien.\n"
            "Si les documents fournis ne contiennent pas la réponse à la question, dites explicitement que vous n'avez pas l'information dans votre base de connaissances.\n"
            "Pour chaque fait important mentionné dans votre réponse, citez la source (le nom du dépôt ou le titre du document) entre parenthèses.\n\n"
            "CONTEXTE DE RECHERCHE :\n"
            "======================\n"
            "{context}\n"
            "======================"
        )
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        
        # 5. Execute LLM chain
        chain = prompt_template | self.llm
        try:
            print("Sending request to Mistral API...")
            response = chain.invoke({
                "context": context_str,
                "question": question
            })
            answer = response.content
        except Exception as e:
            answer = f"Une erreur s'est produite lors de l'appel à l'API Mistral : {e}"
            
        return {
            "answer": answer,
            "sources": sources
        }

if __name__ == "__main__":
    # Test simple (if API key is present)
    agent = MistralRAGAgent()
    if agent.vector_store:
        res = agent.query("Comment utiliser mistral-finetune ?")
        print("REPONSE :", res["answer"])
        print("SOURCES :", res["sources"])
