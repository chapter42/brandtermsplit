"""Google Search Console API: list properties and fetch query data with a user's access token.

Plain REST calls via requests, so the only extra dependency for the Google login is
Streamlit's auth extra (Authlib). The token comes from ``st.user.tokens["access"]``.
"""

from urllib.parse import quote

import pandas as pd
import requests

API = "https://www.googleapis.com/webmasters/v3"
SCOPE = "https://www.googleapis.com/auth/webmasters.readonly"
PAGE_SIZE = 25_000  # maximum rows per searchAnalytics request


class GSCError(Exception):
    """API error with a message that can be shown to the user."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _call(method: str, url: str, token: str, **kwargs) -> dict:
    response = requests.request(method, url, headers={"Authorization": f"Bearer {token}"}, timeout=120, **kwargs)
    if response.status_code == 401:
        raise GSCError("Je Google-sessie is verlopen. Log uit en opnieuw in.", 401)
    if response.status_code == 403:
        raise GSCError(
            "Geen toegang: dit account heeft geen rechten op deze property, "
            "of de login mist de Search Console-toestemming.",
            403,
        )
    if not response.ok:
        try:
            detail = response.json()["error"]["message"]
        except (ValueError, KeyError):
            detail = response.text[:300]
        raise GSCError(f"Search Console API-fout ({response.status_code}): {detail}", response.status_code)
    return response.json()


def list_sites(token: str) -> list[str]:
    """Properties the user can read, domain properties first."""
    entries = _call("GET", f"{API}/sites", token).get("siteEntry", [])
    sites = [e["siteUrl"] for e in entries if e.get("permissionLevel") != "siteUnverifiedUser"]
    return sorted(sites, key=lambda s: (not s.startswith("sc-domain:"), s))


def brand_regex(terms: list[str]) -> str:
    """RE2 filter that keeps queries containing any brand word (glued forms included)."""
    words = set()
    for term in terms:
        term = term.strip().lower()
        if not term:
            continue
        words.add(term.replace(" ", ""))
        words.update(w for w in term.split() if len(w) >= 2)
    escaped = sorted((_re2_escape(w) for w in words), key=len, reverse=True)
    return "|".join(escaped)


def _re2_escape(text: str) -> str:
    return "".join("\\" + c if c in r".^$*+?()[]{}|\\" else c for c in text)


def fetch_queries(
    token: str,
    site: str,
    start: str,
    end: str,
    search_type: str = "web",
    regex: str | None = None,
    max_rows: int = 250_000,
    progress=None,
) -> pd.DataFrame:
    """All rows for dimension ``query`` between ``start`` and ``end`` (YYYY-MM-DD), paginated.

    Returns query, clicks, impressions, avg_position. ``progress`` is called with the
    number of rows fetched so far.
    """
    url = f"{API}/sites/{quote(site, safe='')}/searchAnalytics/query"
    body = {
        "startDate": start,
        "endDate": end,
        "dimensions": ["query"],
        "type": search_type,
        "rowLimit": PAGE_SIZE,
        "dataState": "final",
    }
    if regex:
        body["dimensionFilterGroups"] = [
            {"filters": [{"dimension": "query", "operator": "includingRegex", "expression": regex}]}
        ]

    rows: list[dict] = []
    while len(rows) < max_rows:
        body["startRow"] = len(rows)
        page = _call("POST", url, token, json=body).get("rows", [])
        rows.extend(page)
        if progress:
            progress(len(rows))
        if len(page) < PAGE_SIZE:
            break

    if not rows:
        return pd.DataFrame(columns=["query", "clicks", "impressions", "avg_position"])
    return pd.DataFrame(
        {
            "query": [r["keys"][0] for r in rows],
            "clicks": [r["clicks"] for r in rows],
            "impressions": [r["impressions"] for r in rows],
            "avg_position": [r["position"] for r in rows],
        }
    )
