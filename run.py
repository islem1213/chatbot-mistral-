import sys
import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Pipeline RAG pour l'écosystème Mistral AI (POC)")
    parser.add_argument(
        "action",
        choices=["ingest", "index", "graph-index", "graph-query", "app", "all"],
        help="Action à exécuter : ingest (récupérer les données), index (créer le vector store), graph-index (construire le graphe de connaissances GraphRAG), graph-query (interroger le graphe en ligne de commande), app (lancer Gradio) ou all (exécuter le tout)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Nombre maximal de dépôts GitHub / pages de doc à ingérer (par défaut: 15)"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Quel dépôt permet de fine-tuner un modèle Mistral ?",
        help="Question pour l'action graph-query"
    )
    parser.add_argument(
        "--with-graph",
        action="store_true",
        help="Avec 'all' : construire aussi le graphe GraphRAG (plus long, appelle le LLM pour chaque chunk)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Partager l'application Gradio publiquement via un lien temporaire"
    )
    
    args = parser.parse_args()
    
    if args.action == "ingest":
        run_ingest(args.limit)
    elif args.action == "index":
        run_index()
    elif args.action == "graph-index":
        run_graph_index()
    elif args.action == "graph-query":
        run_graph_query(args.question)
    elif args.action == "app":
        run_app(args.share)
    elif args.action == "all":
        print("=== ÉTAPE 1 : INGESTION ===")
        run_ingest(args.limit)
        print("\n=== ÉTAPE 2 : INDEXATION ===")
        run_index()
        if args.with_graph:
            print("\n=== ÉTAPE 2b : GRAPHE DE CONNAISSANCES (GraphRAG) ===")
            run_graph_index()
        print("\n=== ÉTAPE 3 : LANCEMENT DE L'APPLICATION ===")
        run_app(args.share)

def run_ingest(limit: int):
    print("Début de l'ingestion...")
    from ingestion import github_scraper, doc_scraper
    
    print("\n--- Ingestion des dépôts GitHub ---")
    github_scraper.run_ingestion(output_dir="data/github")
    
    print("\n--- Ingestion de la documentation docs.mistral.ai ---")
    doc_scraper.run_ingestion(output_dir="data/docs", limit=limit)
    
    print("Ingestion terminée avec succès.")

def run_index():
    print("Début de l'indexation...")
    from indexing import vector_store
    
    vector_store.build_vector_store(
        github_dir="data/github",
        docs_dir="data/docs",
        persist_directory="chroma_db",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Indexation terminée.")

def run_graph_index():
    print("Construction du graphe de connaissances (GraphRAG, LlamaIndex)...")
    from graphrag import graph_index

    graph_index.build_graph_index(
        github_dir="data/github",
        docs_dir="data/docs",
        persist_dir="graph_store"
    )

def run_graph_query(question: str):
    from graphrag.graph_agent import GraphRAGAgent

    agent = GraphRAGAgent(persist_dir="graph_store")
    res = agent.query(question)
    print("\nQUESTION :", question)
    print("\nRÉPONSE :\n", res["answer"])
    print("\nRELATIONS DU GRAPHE UTILISÉES :")
    for t in res["triplets"][:15]:
        print("  -", t)
    print("\nSOURCES :")
    for s in res["sources"]:
        print(f"  - {s['title']} : {s['url']}")

def run_app(share: bool):
    print("Lancement de l'interface Gradio...")
    from app import interface
    interface.launch_app(share=share)

if __name__ == "__main__":
    main()
