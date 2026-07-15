import os
import json
import requests
import time
from typing import List, Dict, Any

# Repositories to fetch if dynamic fetching fails or as a core set
DEFAULT_REPOS = [
    "client-python",
    "mistral-common",
    "mistral-finetune",
    "mistral-src",
    "cookbook",
    "codestral-mamba",
    "mimi",
    "inference-reference",
    "client-ts",
    "helm-charts",
    "mistral-cli"
]

def fetch_repositories(org: str = "mistralai", limit: int = 15) -> List[Dict[str, Any]]:
    """
    Fetch repository list from GitHub API or use default ones if rate limited.
    """
    headers = {}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    
    url = f"https://api.github.com/orgs/{org}/repos?per_page={limit}&sort=updated"
    try:
        print(f"Fetching repository list from GitHub for org {org}...")
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            repos_data = response.json()
            return [
                {
                    "name": repo["name"],
                    "full_name": repo["full_name"],
                    "description": repo.get("description", ""),
                    "stargazers_count": repo.get("stargazers_count", 0),
                    "pushed_at": repo.get("pushed_at", ""),
                    "html_url": repo.get("html_url", "")
                }
                for repo in repos_data
            ]
        else:
            print(f"Failed to fetch from API (Status: {response.status_code}). Using default list.")
    except Exception as e:
        print(f"Error fetching repo list: {e}. Using default list.")
        
    # Fallback to default list
    fallback_list = []
    for repo_name in DEFAULT_REPOS:
        # We will fetch individual repo info in get_repo_details
        fallback_list.append({
            "name": repo_name,
            "full_name": f"{org}/{repo_name}",
            "description": "",
            "stargazers_count": 0,
            "pushed_at": "",
            "html_url": f"https://github.com/{org}/{repo_name}"
        })
    return fallback_list

def get_repo_details(repo: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch detailed repo info and README.
    """
    headers = {}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
        
    full_name = repo["full_name"]
    name = repo["name"]
    
    # 1. Update/Fetch general info if stars or description is missing (from fallback)
    if not repo["description"]:
        url = f"https://api.github.com/repos/{full_name}"
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                repo["description"] = data.get("description", "")
                repo["stargazers_count"] = data.get("stargazers_count", 0)
                repo["pushed_at"] = data.get("pushed_at", "")
        except Exception as e:
            print(f"Error updating repo details for {name}: {e}")

    # 2. Fetch README
    readme_content = ""
    readme_url = f"https://api.github.com/repos/{full_name}/readme"
    headers["Accept"] = "application/vnd.github.v3.raw" # Request raw content
    
    try:
        print(f"Fetching README for {full_name}...")
        res = requests.get(readme_url, headers=headers, timeout=10)
        if res.status_code == 200:
            readme_content = res.text
        else:
            print(f"README not found or API rate limited for {name} (Status: {res.status_code})")
    except Exception as e:
        print(f"Error fetching README for {name}: {e}")
        
    return {
        "name": name,
        "full_name": full_name,
        "description": repo["description"],
        "stars": repo["stargazers_count"],
        "last_activity": repo["pushed_at"],
        "url": repo["html_url"],
        "readme": readme_content
    }

def run_ingestion(output_dir: str = "data/github"):
    os.makedirs(output_dir, exist_ok=True)
    repos = fetch_repositories()
    
    for repo in repos:
        details = get_repo_details(repo)
        filename = os.path.join(output_dir, f"{details['name']}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        print(f"Saved {details['name']} data to {filename}")
        time.sleep(1) # Polite sleep between requests

if __name__ == "__main__":
    run_ingestion()
