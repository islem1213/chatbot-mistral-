"""
Agent GraphRAG : répond aux questions en combinant
  - les relations du graphe de connaissances (entités liées à la question),
  - les passages sources associés,
  - et, en mode hybride, les chunks de l'index vectoriel Chroma existant.
"""
import os
from typing import Any, Dict, List, Optional

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
)
from llama_index.core.llms import ChatMessage

from graphrag.graph_index import get_embed_model, get_llm

SYSTEM_PROMPT = (
    "Vous êtes un agent de Q&A spécialisé sur l'écosystème Mistral AI.\n"
    "Le contexte contient des RELATIONS issues d'un graphe de connaissances "
    "(format : sujet -> relation -> objet) et des PASSAGES issus de la documentation.\n"
    "Répondez UNIQUEMENT à partir de ce contexte. Utilisez les relations pour relier les "
    "concepts entre eux (ex. : quel dépôt fournit quelle fonctionnalité).\n"
    "Si le contexte ne permet pas de répondre, dites-le explicitement.\n"
    "Citez la source (titre du document ou nom du dépôt) entre parenthèses pour chaque fait important."
)


class GraphRAGAgent:
    def __init__(
        self,
        persist_dir: str = "graph_store",
        llm=None,
        embed_model=None,
        vector_agent=None,
    ):
        self.persist_dir = persist_dir
        self.llm = llm or get_llm()
        self.embed_model = embed_model or get_embed_model()
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        # Agent vectoriel (agent/rag_agent.py) optionnel pour le mode hybride.
        self.vector_agent = vector_agent

        self.index = None
        if os.path.exists(os.path.join(persist_dir, "docstore.json")):
            storage = StorageContext.from_defaults(persist_dir=persist_dir)
            self.index = load_index_from_storage(
                storage, llm=self.llm, embed_model=self.embed_model
            )
        else:
            print(f"Graphe introuvable dans '{persist_dir}'. Lancez : python run.py graph-index")

    def _retriever(self, similarity_top_k: int = 4, path_depth: int = 1):
        store = self.index.property_graph_store
        sub_retrievers = [
            # Le LLM propose des mots-clés / synonymes, puis on cherche ces entités dans le graphe.
            LLMSynonymRetriever(store, llm=self.llm, include_text=True, path_depth=path_depth),
            # Recherche des entités proches de la question par embeddings, puis voisinage dans le graphe.
            VectorContextRetriever(
                store,
                embed_model=self.embed_model,
                # SimplePropertyGraphStore ne gère pas les vecteurs : on passe le vector store de l'index.
                vector_store=self.index.vector_store,
                include_text=True,
                similarity_top_k=similarity_top_k,
                path_depth=path_depth,
            ),
        ]
        return self.index.as_retriever(sub_retrievers=sub_retrievers, include_text=True)

    def retrieve(self, question: str, similarity_top_k: int = 4, path_depth: int = 1):
        """Retourne (relations, passages) pertinents pour la question."""
        nodes = self._retriever(similarity_top_k, path_depth).retrieve(question)
        triplets: List[str] = []
        passages: List[Dict[str, Any]] = []
        seen_passages = set()
        for nws in nodes:
            node = nws.node
            text = node.get_content()
            # Les retrievers de graphe préfixent le passage par les relations trouvées.
            for line in text.splitlines():
                if " -> " in line and line.strip() not in triplets:
                    triplets.append(line.strip())
            key = node.metadata.get("source", "") + text[:80]
            if key not in seen_passages:
                seen_passages.add(key)
                passages.append(
                    {
                        "text": text,
                        "title": node.metadata.get("title", ""),
                        "url": node.metadata.get("source", ""),
                        "type": node.metadata.get("type", "unknown"),
                    }
                )
        return triplets, passages

    def query(self, question: str, hybrid: bool = False, k: int = 4) -> Dict[str, Any]:
        if self.index is None:
            return {
                "answer": "Erreur : le graphe de connaissances n'est pas construit. "
                "Lancez `python run.py graph-index`.",
                "sources": [],
                "triplets": [],
            }
        try:
            triplets, passages = self.retrieve(question, similarity_top_k=k)
        except Exception as e:
            return {"answer": f"Erreur lors de la recherche dans le graphe : {e}", "sources": [], "triplets": []}

        context = "RELATIONS DU GRAPHE :\n" + ("\n".join(triplets) if triplets else "(aucune)") + "\n\n"
        context += "PASSAGES :\n"
        for i, p in enumerate(passages, 1):
            context += f"--- Passage {i} ({p['title']}) ---\n{p['text']}\n\n"

        # Mode hybride : on ajoute les chunks de l'index vectoriel Chroma existant.
        if hybrid and self.vector_agent is not None and self.vector_agent.vector_store is not None:
            for j, doc in enumerate(self.vector_agent.vector_store.similarity_search(question, k=k), 1):
                title = doc.metadata.get("title", "")
                context += f"--- Passage vectoriel {j} ({title}) ---\n{doc.page_content}\n\n"
                passages.append(
                    {
                        "text": doc.page_content,
                        "title": title,
                        "url": doc.metadata.get("source", ""),
                        "type": doc.metadata.get("type", "unknown"),
                    }
                )

        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=f"CONTEXTE :\n{context}\nQUESTION : {question}"),
        ]
        try:
            answer = self.llm.chat(messages).message.content
        except Exception as e:
            answer = f"Une erreur s'est produite lors de l'appel au LLM : {e}"

        sources, seen = [], set()
        for p in passages:
            if p["url"] and p["url"] not in seen:
                seen.add(p["url"])
                sources.append({"title": p["title"], "url": p["url"], "type": p["type"]})

        return {"answer": answer, "sources": sources, "triplets": triplets}


if __name__ == "__main__":
    agent = GraphRAGAgent()
    res = agent.query("Quel dépôt permet de fine-tuner un modèle Mistral ?")
    print("RÉPONSE :", res["answer"])
    print("RELATIONS :", res["triplets"][:10])
    print("SOURCES :", res["sources"])
