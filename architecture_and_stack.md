# Architecture et Stack Technologique

## 1. Architecture fonctionnelle

### Vue d'ensemble
Le projet est un prototype RAG (Retrieval-Augmented Generation) permettant de répondre à des questions sur l'écosystème Mistral AI.

### Composants principaux

1. **Ingestion**
   - `ingestion/github_scraper.py` : collecte les métadonnées et README des dépôts GitHub Mistral.
   - `ingestion/doc_scraper.py` : récupère et scrape les pages officielles `docs.mistral.ai`.
   - Données produites : fichiers JSON stockés dans `data/github/` et `data/docs/`.

2. **Indexation**
   - `indexing/vector_store.py` : transforme les documents en chunks, calcule les embeddings et crée un index Chroma.
   - Stockage local : `chroma_db/`.

3. **Recherche & génération**
   - `agent/rag_agent.py` : effectue la recherche de similarité dans Chroma puis construit un prompt pour l'API Mistral.
   - Le modèle Mistral génère une réponse basée sur le contexte récupéré.

4. **Interface utilisateur**
   - `app/interface.py` : application Gradio pour saisir des questions et afficher les réponses avec les sources.

5. **Orchestration**
   - `run.py` : point d'entrée unique pour exécuter l'ingestion, l'indexation et l'application.

### Diagramme conceptuel

- Utilisateur → Gradio (`app/interface.py`)
- Requête utilisateur → `agent/rag_agent.py`
- Recherche vectorielle → Chroma (`chroma_db`)
- Contexte récupéré → API Mistral (`langchain_mistralai`)
- Réponse générée + sources → Interface Gradio

## 2. Architecture technique détaillée

### Flux de données

1. La documentation et les README GitHub sont ingérés et sauvegardés sous forme JSON.
2. Les documents sont découpés en chunks textuels avec `RecursiveCharacterTextSplitter`.
3. Chaque chunk est converti en vecteur via `HuggingFaceEmbeddings`.
4. Les vecteurs sont stockés dans `Chroma` pour la recherche par similarité.
5. Lors d'une requête utilisateur, les chunks pertinents sont récupérés depuis l'index.
6. Le prompt est formé en combinant ces chunks avec la question utilisateur.
7. L'API Mistral retourne une réponse naturelle, fiabilisée par le contexte.

### Stockage

- Fichiers de données : `data/docs/`, `data/github/`
- Index hybride local : `chroma_db/`
- Variables d'environnement : `.env`

### Sécurité et configuration

- `MISTRAL_API_KEY` : clé API Mistral requise pour les appels LLM.
- `GITHUB_TOKEN` : optionnel pour améliorer l'ingestion GitHub et éviter les limites de taux.
- Le fichier `.env.example` documente les variables attendues.

## 3. Stack technologique

### Langage

- Python 3.x (3.13 recommandé)

### Bibliothèques principales

- `langchain` : orchestration RAG
- `langchain-chroma` : intégration Chroma vector store
- `langchain_community` : connecteurs d'embeddings
- `langchain_mistralai` : génération via l'API Mistral
- `sentence-transformers/all-MiniLM-L6-v2` : embeddings textuels
- `gradio` : interface web rapide
- `requests`, `beautifulsoup4` : scraping web
- `python-dotenv` : gestion des variables d'environnement

### Infrastructure minimale

- Environnement local Windows
- Stockage local de l'index (`chroma_db`)
- API externe Mistral

### Fichiers clés

- `run.py` : orchestration et commandes CLI
- `ingestion/github_scraper.py` : collecte GitHub
- `ingestion/doc_scraper.py` : scrape documentation
- `indexing/vector_store.py` : génération et stockage de vecteurs
- `agent/rag_agent.py` : logique RAG et génération de réponse
- `app/interface.py` : interface utilisateur Gradio
- `.env.example` : configuration requise
- `requirements.txt` : dépendances Python

## 4. Recommandations de stack

### Moteur de recherche vectorielle

- Chroma est adapté pour un prototype local.
- Pour production, envisager un service vectoriel managé ou Redis/FAISS selon le besoin de scalabilité.

### Embeddings

- `all-MiniLM-L6-v2` est un bon choix pour un prototype performant sur CPU.
- Pour une meilleure qualité, migrer vers `all-mpnet-base-v2` ou un embedding plus récent.

### Modèle IA

- `mistral-large-latest` est le modèle recommandé pour ce projet, aligné avec l'écosystème Mistral.
- En production, il est possible de choisir un modèle plus petit ou un modèle optimisé selon le coût/latence.

### Interface

- Gradio est idéal pour un POC.
- Pour un produit final, migrer vers une application web plus robuste (Flask/FastAPI + React) si nécessaire.

## 5. Améliorations futures

- Ajouter la gestion des métadonnées de qualité pour les documents GitHub (README, licence, activité).
- Ajouter un composant de scoring des sources pour prioriser les docs les plus récentes et officielles.
- Introduire des tests d'intégration pour valider les réponses et la cohérence du RAG.
- Prévoir une mise à jour automatique du corpus de documentation et des dépôts.
