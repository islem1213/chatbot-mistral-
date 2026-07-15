# Plan d'implémentation : Agent RAG pour l'écosystème Mistral AI

## 1. Besoin métier

Objectif : fournir un assistant Q&A basé sur des données officielles Mistral AI et des dépôts GitHub associés, afin d'aider les développeurs à comprendre l'écosystème Mistral, la fine-tuning, les SDK et les cas d'usage.

Pourquoi :
- Les informations Mistral sont réparties entre la documentation Docusaurus et plusieurs dépôts GitHub.
- Les utilisateurs ont besoin d'une réponse synthétique, cohérente et contextualisée sans devoir parcourir manuellement plusieurs pages.
- Le projet vise à réduire le temps de recherche et augmenter la productivité des équipes IA/ML.

## 2. Sources officielles et récentes

Sources citées dans le projet :
- `https://docs.mistral.ai/` et pages associées de la documentation officielle
- Repos GitHub officiels de `mistralai` tels que `client-python`, `mistral-common`, `mistral-finetune`, `client-ts`, `cookbook`

Justification :
- La documentation `docs.mistral.ai` est la source primaire officielle des offres, des capacités de modèles, et des guides.
- Les dépôts GitHub de `mistralai` contiennent le code source, les SDK, les exemples et les instructions opérationnelles.

## 3. Analyse des benchmarks Mistral

Benchmark Mistral attendu :
- Mistral 7B et Mistral Large sont publiés comme modèles de pointe pour les charges de travail open-source.
- Mistral 7B vise une efficacité comparable à des modèles plus grands avec une latence et un coût moindres.

Comparaison :
- Mistral est positionné devant des concurrents open-source comme LLaMA 2 et Falcon en termes de score de compréhension et de vitesse (selon les communiqués officiels).
- Le choix de `mistral-large-latest` dans le code indique une volonté de privilégier un modèle récent et officiel.

Sources et preuves :
- La documentation officielle Mistral publie des benchmarks sur la page `getting-started/models/` et les pages de capacités.
- Les pages `capabilities/completion/` et `capabilities/finetuning/` justifient l'utilisation de Mistral pour la génération et l'adaptation.

## 4. Évaluation des dépôts GitHub

Repos analysés dans le projet :
- `client-python`
- `client-ts`
- `mistral-common`
- `mistral-finetune`
- `cookbook`
- `mistral-inference`
- `mistral-vibe`
- `search-starter-app`
- `platform-docs-public`
- `mistral-evals`
- `megablocks-public`
- `llm-d-router-public`
- `FLTEval`
- `LeanstralSafeVerify`
- `zed-extensions`

Critères d’évaluation :
- Documentation : présence de README, guides d'installation et exemples pratiques.
- Maintenance : fréquence des commits et activité récente sur les repos.
- Licence : utilisation de licences open-source compatibles avec le projet.
- Qualité du code : structure claire, exemples et bonnes pratiques de SDK.

Hypothèse actuelle :
- Le projet utilise ces dépôts comme source documentaire, mais une évaluation plus détaillée doit vérifier la licence exacte et l'état de maintenance via les métadonnées GitHub.

## 5. Architecture technique proposée

Architecture globale :

1. Ingestion
   - Scraper la documentation officielle Mistral (`docs.mistral.ai`) et les README GitHub.
   - Stocker les pages JSON dans `data/docs` et `data/github`.

2. Indexation
   - Convertir ces documents en chunks avec `RecursiveCharacterTextSplitter`.
   - Calculer des embeddings à l'aide d'une couche HuggingFace (`all-MiniLM-L6-v2`).
   - Construire un store Chroma local dans `chroma_db`.

3. Recherche et génération
   - Recherche de similarité dans Chroma.
   - Construction d'un prompt contextualisé avec les documents récupérés.
   - Appel à l'API Mistral pour générer la réponse.

4. Interface utilisateur
   - Gradio pour un prototype web interactif.
   - Affichage de la réponse et des sources associées.

Justification technique :
- RAG permet d’ancrer les réponses dans des documents réels et de limiter les hallucinations.
- Un store local Chroma est adapté à un POC et garantit des temps de recherche rapides.
- Un modèle Mistral hébergé via API est cohérent avec l'écosystème ciblé.

## 6. Technologies et modèles IA

Stack proposé :
- Python 3.13
- `langchain` pour orchestrer retrieval + LLM
- `langchain-chroma` pour le vector store
- `sentence-transformers/all-MiniLM-L6-v2` pour les embeddings
- `mistral-large-latest` pour la génération via `langchain_mistralai`
- `gradio` pour l’interface utilisateur

Explication des choix :
- `MiniLM-L6-v2` est un embedding efficace pour des prototypes et des recherches textuelles.
- Mistral Large est un modèle récent axé sur la génération et la qualité conversationnelle.
- Gradio permet une mise en œuvre rapide d’un prototype interactif.

## 7. Roadmap de mise en œuvre

Phase 1 : Prototype technique (1-2 semaines)
- Mettre en place l’ingestion des docs officiels et des README GitHub.
- Construire le store Chroma et vérifier les recherches.
- Implémenter le front-end Gradio.
- Tester les questions basiques et vérifier la citation des sources.

Phase 2 : Validation de contenu et qualité (1-2 semaines)
- Vérifier la qualité des documents ingérés et la couverture des sujets Mistral.
- Ajouter une couche de nettoyage des pages web et de filtrage des contenus non informatifs.
- Améliorer la formulation de prompts pour éviter les hallucinations.

Phase 3 : Production et optimisation (2-3 semaines)
- Passer en production API Mistral avec gestion des clés et erreurs.
- Ajouter un cache local ou un moteur de mise à jour régulière des docs.
- Étendre l’analyse GitHub : licence, maintenance, versionnage.

## 8. Estimation des ressources

Humaines :
- 1 ingénieur IA/ML pour l’architecture de données et l’intégration RAG
- 1 développeur backend pour l’ingestion, l’indexation et la persistance
- 1 développeur frontend léger pour l’interface Gradio ou un UI plus complète

Matérielles :
- Environnement de développement Python local ou serveur cloud léger
- Stockage : quelques centaines de Mo pour les JSON, embeddings et Chroma
- API Mistral (clé payante selon usage)

Financières :
- Coût de l’API Mistral selon volume de requêtes et taille du modèle
- Coût éventuel de calcul si l’on migre l’ingestion vers un runtime cloud
- Volume estimé pour un POC : 100-300€ selon utilisation et fréquence de liaisons API

## 9. Risques, limites et alternatives

Risques identifiés :
- Authentification API Mistral échouée (`401 Unauthorized`)
- Hallucinations si le LLM n’utilise pas correctement le contexte
- Données périmées si la documentation n’est pas régulièrement mise à jour
- Requêtes GitHub limitées par le rate limit sans token valide

Limites :
- Prototype basé sur un modèle API et un store local, pas sur un déploiement industriel
- La couverture dépend des pages sélectionnées pour l’ingestion
- La qualité des réponses est liée au nettoyage et à la granularité des chunks

Alternatives :
- Utiliser un modèle local open-source à la place de l’API Mistral pour un déploiement on-premise
- Remplacer `all-MiniLM-L6-v2` par un embedding plus robuste (`sentence-transformers/all-mpnet-base-v2`) si nécessaire
- Déployer Chroma dans un service persistant plutôt qu’un répertoire local

## 10. Justification par preuve

Chaque recommandation est appuyée par :
- la documentation Mistral : pages officielles `docs.mistral.ai`
- les dépôts GitHub officiels `mistralai` (exemple `client-python`, `mistral-finetune`)
- les meilleures pratiques RAG : ingestion, chunking, retrieval + prompt
- les retours de la structure existante du projet (ingestion + indexation + Gradio)

### Conclusion
Ce projet a une base solide pour un POC RAG Mistral AI. Il manque aujourd’hui un document de plan détaillé : ce fichier comble cette lacune en fournissant la vision produit, l’architecture, la roadmap, l’évaluation des sources et les risques.
