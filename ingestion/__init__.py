from .doc_scraper import run_ingestion as run_docs_ingestion
from .github_scraper import run_ingestion as run_github_ingestion

__all__ = [
    "run_docs_ingestion",
    "run_github_ingestion"
]
