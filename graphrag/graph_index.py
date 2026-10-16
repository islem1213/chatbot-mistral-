"""
Construction du graphe de connaissances (GraphRAG) avec LlamaIndex.

Pipeline :
    documents JSON (data/docs, data/github)
      -> découpage en chunks (SentenceSplitter)
      -> extraction de triplets (sujet, relation, objet) par le LLM Mistral
      -> PropertyGraphIndex (graphe + embeddings des entités)
      -> persistance dans graph_store/ + export HTML interactif du graphe
"""
import glob
import json
import os
from typing import List, Optional

from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.core.node_parser import SentenceSplitter

DEFAULT_LLM_MODEL = "mistral-large-latest"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Prompt d'extraction adapté au domaine (écosystème Mistral AI).
KG_EXTRACT_PROMPT = (
    "Voici un extrait de documentation technique sur l'écosystème Mistral AI.\n"
    "Extrais au maximum {max_knowledge_triplets} triplets de connaissances sous la forme "
    "(sujet, relation, objet).\n"
    "Privilégie les entités utiles : modèles, dépôts GitHub, SDK, API, fonctionnalités, "
    "formats, outils, licences, plateformes de déploiement.\n"
    "Utilise des relations courtes et explicites (ex. : 'fournit', 'dépend de', 'supporte', "
    "'est un', 'permet', 'se déploie sur').\n"
    "N'invente rien : chaque triplet doit être justifié par le texte.\n"
    "---------------------\n"
    "Exemple :\n"
    "Texte : mistral-finetune permet le fine-tuning LoRA des modèles Mistral.\n"
    "Triplets :\n"
    "(mistral-finetune, permet, fine-tuning LoRA)\n"
    "(fine-tuning LoRA, s'applique à, modèles Mistral)\n"
    "---------------------\n"
    "Texte : {text}\n"
    "Triplets :\n"
)


def get_llm(model: str = DEFAULT_LLM_MODEL):
    """LLM Mistral via LlamaIndex (lit MISTRAL_API_KEY dans l'environnement)."""
    from llama_index.llms.mistralai import MistralAI

    if not os.getenv("MISTRAL_API_KEY"):
        print("ATTENTION : MISTRAL_API_KEY n'est pas définie, les appels au LLM échoueront.")
    return MistralAI(model=model, temperature=0.0)


def get_embed_model(model_name: str = DEFAULT_EMBED_MODEL):
    """Même modèle d'embeddings que l'index vectoriel Chroma, pour rester cohérent."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=model_name, device="cpu")


def load_documents(github_dir: str = "data/github", docs_dir: str = "data/docs") -> List[Document]:
    """Charge les JSON produits par l'ingestion en Documents LlamaIndex."""
    documents: List[Document] = []

    for filepath in sorted(glob.glob(os.path.join(github_dir, "*.json"))):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:  # fichier corrompu : on continue
            print(f"Erreur de lecture {filepath} : {e}")
            continue
        readme = (data.get("readme") or "").strip()
        if not readme:
            continue
        name = data.get("name", "")
        text = (
            f"Dépôt GitHub : {name}\n"
            f"Description : {data.get('description', '')}\n\n"
            f"{readme}"
        )
        documents.append(
            Document(
                text=text,
                metadata={"source": data.get("url", ""), "title": name, "type": "github"},
                excluded_embed_metadata_keys=["source", "type"],
                excluded_llm_metadata_keys=["source", "type"],
            )
        )

    for filepath in sorted(glob.glob(os.path.join(docs_dir, "*.json"))):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Erreur de lecture {filepath} : {e}")
            continue
        content = (data.get("content") or "").strip()
        if not content:
            continue
        title = data.get("title", "")
        documents.append(
            Document(
                text=f"Page de documentation : {title}\n\n{content}",
                metadata={"source": data.get("url", ""), "title": title, "type": "documentation"},
                excluded_embed_metadata_keys=["source", "type"],
                excluded_llm_metadata_keys=["source", "type"],
            )
        )

    print(f"{len(documents)} documents chargés pour le graphe.")
    return documents


def build_graph_index(
    github_dir: str = "data/github",
    docs_dir: str = "data/docs",
    persist_dir: str = "graph_store",
    llm=None,
    embed_model=None,
    max_paths_per_chunk: int = 10,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    num_workers: int = 2,
) -> Optional[PropertyGraphIndex]:
    """Construit et persiste le PropertyGraphIndex. Retourne l'index (ou None si aucun document)."""
    documents = load_documents(github_dir, docs_dir)
    if not documents:
        print("Aucun document à indexer. Lancez d'abord : python run.py ingest")
        return None

    llm = llm or get_llm()
    embed_model = embed_model or get_embed_model()
    Settings.llm = llm
    Settings.embed_model = embed_model

    extractors = [
        # Triplets sémantiques extraits par le LLM.
        SimpleLLMPathExtractor(
            llm=llm,
            extract_prompt=KG_EXTRACT_PROMPT,
            max_paths_per_chunk=max_paths_per_chunk,
            num_workers=num_workers,
        ),
        # Liens structurels entre chunks voisins (sans appel LLM).
        ImplicitPathExtractor(),
    ]

    index = PropertyGraphIndex.from_documents(
        documents,
        llm=llm,
        embed_model=embed_model,
        kg_extractors=extractors,
        property_graph_store=SimplePropertyGraphStore(),
        transformations=[SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)],
        embed_kg_nodes=True,
        show_progress=True,
    )

    os.makedirs(persist_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_dir)

    store = index.property_graph_store
    n_triplets = len(store.graph.triplets)
    print(f"Graphe persisté dans '{persist_dir}' : {n_triplets} relations.")

    export_graph_html(index, os.path.join(persist_dir, "knowledge_graph.html"))
    return index


def export_graph_html(index: PropertyGraphIndex, path: str) -> None:
    """Export du graphe en page HTML interactive (carte mentale navigable)."""
    try:
        index.property_graph_store.save_networkx_graph(name=path)
        print(f"Visualisation du graphe : {path}")
    except ImportError:
        print("Installez networkx et pyvis pour exporter la visualisation du graphe.")
