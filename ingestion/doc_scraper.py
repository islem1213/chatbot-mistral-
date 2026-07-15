import os
import json
import requests
import time
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from urllib.parse import urlparse

# Curated fallback list of important documentation URLs
FALLBACK_DOCS_URLS = [
    "https://docs.mistral.ai/",
    "https://docs.mistral.ai/getting-started/quickstart/",
    "https://docs.mistral.ai/getting-started/models/",
    "https://docs.mistral.ai/capabilities/completion/",
    "https://docs.mistral.ai/capabilities/embeddings/",
    "https://docs.mistral.ai/capabilities/rag/",
    "https://docs.mistral.ai/capabilities/finetuning/",
    "https://docs.mistral.ai/capabilities/guardrailing/",
    "https://docs.mistral.ai/capabilities/agents/",
    "https://docs.mistral.ai/capabilities/function_calling/",
    "https://docs.mistral.ai/deployment/laceworks/",
    "https://docs.mistral.ai/deployment/cloud/azure/"
]

def fetch_sitemap_urls(sitemap_url: str = "https://docs.mistral.ai/sitemap.xml") -> List[str]:
    """
    Fetch URLs from the website sitemap.
    """
    try:
        print(f"Fetching sitemap from {sitemap_url}...")
        headers = {"User-Agent": "MistralFAQBot/1.0 (Educational RAG POC)"}
        response = requests.get(sitemap_url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Sitemap returned status {response.status_code}. Using fallback URLs.")
            return FALLBACK_DOCS_URLS
            
        soup = BeautifulSoup(response.content, "xml")
        urls = [loc.text for loc in soup.find_all("loc")]
        
        # Filter for relevant documentation pages
        # Exclude assets, pdfs, or search pages if any
        filtered_urls = []
        for url in urls:
            parsed = urlparse(url)
            # Make sure it's on docs.mistral.ai and doesn't end with media extensions
            if parsed.netloc == "docs.mistral.ai" and not re.search(r"\.(pdf|png|jpg|jpeg|gif|css|js)$", parsed.path):
                # Clean URL (remove query params/hashes if any)
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                filtered_urls.append(clean_url)
                
        # Remove duplicates
        filtered_urls = list(set(filtered_urls))
        print(f"Found {len(filtered_urls)} URLs in sitemap.")
        return filtered_urls if filtered_urls else FALLBACK_DOCS_URLS
    except Exception as e:
        print(f"Failed to fetch sitemap: {e}. Using fallback URLs.")
        return FALLBACK_DOCS_URLS

def scrape_doc_page(url: str) -> Dict[str, Any]:
    """
    Scrape and extract structured text from a Docusaurus documentation page.
    """
    headers = {"User-Agent": "MistralFAQBot/1.0 (Educational RAG POC)"}
    try:
        print(f"Scraping {url}...")
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Failed to scrape {url} (Status: {response.status_code})")
            return {}
            
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Docusaurus typically places main content in <article> or main
        main_content = soup.find("article")
        if not main_content:
            main_content = soup.find("main")
        if not main_content:
            main_content = soup.find("body")
            
        if not main_content:
            return {}
            
        # Get title
        title_tag = soup.find("h1")
        title = title_tag.text.strip() if title_tag else url
        
        # Extract headings and paragraphs to maintain some structure
        # We can reconstruct a simple markdown-like text
        text_blocks = []
        
        # Clean up some elements we don't want (navigation, footer, toc)
        for element in main_content.find_all(["nav", "footer", "aside", "script", "style"]):
            element.decompose()
            
        for child in main_content.descendants:
            if child.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                text_blocks.append(f"\n\n# {child.get_text().strip()}\n")
            elif child.name == "p":
                text_blocks.append(f"\n{child.get_text().strip()}\n")
            elif child.name == "li":
                text_blocks.append(f"\n- {child.get_text().strip()}")
            elif child.name == "pre":
                # Code blocks
                text_blocks.append(f"\n```\n{child.get_text().strip()}\n```\n")
                
        # Merge consecutive tags nicely, remove duplicate spaces
        content_text = "".join(text_blocks)
        # fallback to raw text if parsing failed to extract anything
        if not content_text.strip():
            content_text = main_content.get_text(separator="\n")
            
        # Clean up duplicate empty lines
        content_text = re.sub(r'\n{3,}', '\n\n', content_text)
        
        # Slug from URL for filename
        parsed_url = urlparse(url)
        slug = parsed_url.path.strip("/").replace("/", "_")
        if not slug:
            slug = "index"
            
        return {
            "title": title,
            "url": url,
            "content": content_text.strip(),
            "slug": slug
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return {}

def run_ingestion(output_dir: str = "data/docs", limit: int = 15):
    os.makedirs(output_dir, exist_ok=True)
    urls = fetch_sitemap_urls()
    
    # Sort and slice to respect a limit of 10-15 main pages for Phase 1
    # We want to prioritize core pages: quickstart, models, capabilities
    core_urls = [url for url in urls if "getting-started" in url or "capabilities" in url or url == "https://docs.mistral.ai/"]
    other_urls = [url for url in urls if url not in core_urls]
    
    selected_urls = (core_urls + other_urls)[:limit]
    print(f"Selected {len(selected_urls)} pages for ingestion.")
    
    for url in selected_urls:
        doc_data = scrape_doc_page(url)
        if doc_data and doc_data.get("content"):
            filename = os.path.join(output_dir, f"{doc_data['slug']}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
            print(f"Saved doc page to {filename}")
            time.sleep(1) # Polite sleep

if __name__ == "__main__":
    run_ingestion()
