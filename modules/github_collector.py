# ============================================================
# modules/github_collector.py
# ============================================================
# Responsibility: Fetch repository data from the GitHub API.
#
# Functions exported:
#   fetch_repositories(keyword, max_repos) → pd.DataFrame
#   fetch_readme(repo_full_name)           → str
# ============================================================

import time
import requests
import pandas as pd
from config import GITHUB_TOKEN, MAX_REPOS, PER_PAGE


# ------------------------------------------------------------------
# Build the HTTP headers for every GitHub API call
# ------------------------------------------------------------------
def _get_headers() -> dict:
    """Return authorization headers if a token is configured."""
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


# ------------------------------------------------------------------
# Core collector
# ------------------------------------------------------------------
def fetch_repositories(keyword: str, max_repos: int = MAX_REPOS) -> pd.DataFrame:
    """
    Search GitHub for repositories matching *keyword* and return a
    tidy DataFrame with one row per repository.

    Columns returned
    ----------------
    name, full_name, description, url, stars, forks,
    language, topics, created_at, updated_at
    """
    headers   = _get_headers()
    repos     = []           # accumulate raw dicts here
    page      = 1
    per_page  = min(PER_PAGE, max_repos)

    print(f"[collector] Searching GitHub for: '{keyword}'  (max {max_repos} repos)")

    while len(repos) < max_repos:
        # GitHub Search API endpoint
        url    = "https://api.github.com/search/repositories"
        params = {
            "q":        keyword,
            "sort":     "stars",
            "order":    "desc",
            "per_page": per_page,
            "page":     page,
        }

        response = requests.get(url, headers=headers, params=params, timeout=15)

        # Respect rate-limit headers
        if response.status_code == 403:
            reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait       = max(reset_time - int(time.time()), 5)
            print(f"[collector] Rate limit hit. Waiting {wait}s …")
            time.sleep(wait)
            continue

        if response.status_code != 200:
            print(f"[collector] API error {response.status_code}: {response.text[:200]}")
            break

        data  = response.json()
        items = data.get("items", [])

        if not items:
            break   # No more results

        for item in items:
            repos.append({
                "name":        item.get("name", ""),
                "full_name":   item.get("full_name", ""),
                "description": item.get("description") or "",
                "url":         item.get("html_url", ""),
                "stars":       item.get("stargazers_count", 0),
                "forks":       item.get("forks_count", 0),
                "language":    item.get("language") or "Unknown",
                "topics":      item.get("topics", []),      # list of strings
                "created_at":  item.get("created_at", ""),
                "updated_at":  item.get("updated_at", ""),
            })

            if len(repos) >= max_repos:
                break

        page += 1
        # Be polite to the GitHub API (avoid secondary rate limits)
        time.sleep(0.5)

    print(f"[collector] Fetched {len(repos)} repositories.")
    return pd.DataFrame(repos)


# ------------------------------------------------------------------
# README fetcher
# ------------------------------------------------------------------
def fetch_readme(full_name: str) -> str:
    """
    Download the raw README text for a repository.

    Parameters
    ----------
    full_name : str
        Repository identifier in 'owner/repo' format.

    Returns
    -------
    str
        Plain-text README content, or empty string on failure.
    """
    headers  = _get_headers()
    api_url  = f"https://api.github.com/repos/{full_name}/readme"
    response = requests.get(api_url, headers=headers, timeout=10)

    if response.status_code != 200:
        return ""

    import base64
    content_b64 = response.json().get("content", "")
    try:
        return base64.b64decode(content_b64).decode("utf-8", errors="ignore")
    except Exception:
        return ""
