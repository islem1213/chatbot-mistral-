# Mistral AI RAG Agent POC

Prototype de moteur RAG (Retrieval-Augmented Generation) pour l'écosystème Mistral AI.

## Objectif

Ce projet vise à fournir un assistant Q&A capable de répondre à des questions sur Mistral AI en s'appuyant sur :
- la documentation officielle de Mistral (`docs.mistral.ai`)
- les README et informations de plusieurs dépôts GitHub officiels
- un index vectoriel local Chroma
- un modèle Mistral via l'API
- un graphe de connaissances (GraphRAG) construit avec LlamaIndex

## Contenu

- `run.py` : point d'entrée principal pour ingérer, indexer et lancer l'application.
- `ingestion/` : scripts pour récupérer la documentation Mistral et les données GitHub.
- `indexing/` : construction d'un index vectoriel Chroma à partir des documents ingérés.
- `graphrag/` : GraphRAG avec LlamaIndex (`PropertyGraphIndex`) : extraction de triplets par Mistral, graphe persisté dans `graph_store/`, agent de requête graphe et mode hybride.
- `app/` : interface Gradio pour poser des questions et afficher les réponses avec leurs sources.
- `implementation_plan.md` : plan d'implémentation détaillé, architecture, roadmap et risques.

## Installation

1. Activez l'environnement Python :
```powershell
& .\.venv\Scripts\Activate.ps1
```

2. Installez les dépendances :
```powershell
pip install -r requirements.txt
```

3. Copiez le fichier d'exemple d'environnement :
```powershell
copy .env.example .env
```

4. Ouvrez `.env` et ajoutez votre clé Mistral :
```env
MISTRAL_API_KEY="votre_cle_mistral_valide"
```

5. (Optionnel) Ajoutez un token GitHub pour améliorer l'ingestion :
```env
GITHUB_TOKEN=votre_token_github
```

## Usage

- Ingestion des données :
```powershell
python run.py ingest --limit 15
```

- Construction de l'index :
```powershell
python run.py index
```

- Lancement de l'application Gradio :
```powershell
python run.py app
```

- Exécution complète :
```powershell
python run.py all
```

## GraphRAG (LlamaIndex)

En plus du RAG vectoriel (Chroma), l'agent peut s'appuyer sur un **graphe de connaissances** :

1. Les documents sont découpés en chunks, puis le LLM Mistral extrait des triplets `(sujet, relation, objet)`
   (ex. `mistral-finetune -> permet -> fine-tuning LoRA`) avec un prompt adapté au domaine.
2. Les triplets et les entités (avec leurs embeddings) sont stockés dans un `PropertyGraphIndex` LlamaIndex, persisté dans `graph_store/`.
3. À la question, deux retrievers sont combinés :
   - `LLMSynonymRetriever` : le LLM propose des mots-clés et synonymes, recherchés comme entités dans le graphe ;
   - `VectorContextRetriever` : recherche des entités proches par embeddings, puis exploration de leur voisinage.
4. Le LLM répond à partir des relations du graphe et des passages sources, en citant ses sources.

Trois modes sont disponibles dans l'interface : **RAG vectoriel**, **GraphRAG** et **Hybride** (graphe + chunks Chroma).

- Construction du graphe (après `ingest`) :
```powershell
python run.py graph-index
```

- Interrogation en ligne de commande :
```powershell
python run.py graph-query --question "Quel dépôt permet de fine-tuner un modèle Mistral ?"
```

- Pipeline complet avec graphe :
```powershell
python run.py all --with-graph
```

Une visualisation interactive du graphe est générée dans `graph_store/knowledge_graph.html` (à ouvrir dans un navigateur).

> La construction du graphe appelle le LLM pour chaque chunk : elle est plus longue et consomme des crédits API. Utilisez `--limit` à l'ingestion pour tester sur un petit corpus.

## Notes importantes

- `MISTRAL_API_KEY` doit être valide pour que les appels LLM fonctionnent.
- Si l'API retourne `401 Unauthorized`, vérifiez que la clé est correctement définie et qu'elle n'est pas expirée.
- Le projet utilise un index local `chroma_db`, donc l'ingestion et l'indexation sont nécessaires avant de lancer l'application.

## Bonnes pratiques

- Renouvelez l'index si le contenu de `data/docs` ou `data/github` change.
- Inspectez les documents ingérés pour vous assurer qu'ils sont pertinents et structurés.
- Ajustez le modèle de génération Mistral dans `agent/rag_agent.py` si vous souhaitez tester d'autres variantes.

## Plan d'implémentation

Le plan détaillé se trouve dans `implementation_plan.md` et couvre :
- besoin métier
- architecture technique
- benchmarks et sources
- roadmap de mise en œuvre
- ressources et risques
