"""
news_tools.py
─────────────
Fetch financial news from multiple Indian financial news sources:
  • Economic Times Markets
  • LiveMint
  • Business Standard
  • SEBI announcements

All tools use requests + BeautifulSoup (no API keys required).
"""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from typing import Optional

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_TIMEOUT = 15


def get_et_markets_news(topic: str = "mutual fund") -> list[dict]:
    """
    Fetch latest mutual fund / market news from Economic Times Markets.

    Args:
        topic: Search topic (e.g. "mutual fund", "SIP", "SEBI", "NFO", "Nifty")

    Returns:
        List of articles with title, url, and date.
    """
    try:
        url  = f"https://economictimes.indiatimes.com/topic/{topic.replace(' ', '-')}"
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        articles = []
        for item in soup.select("div.eachStory, li.clearfix, .story-box")[:10]:
            title_tag = item.select_one("h3 a, h2 a, a.title, a")
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            href  = title_tag.get("href", "")
            if not href.startswith("http"):
                href = "https://economictimes.indiatimes.com" + href
            date_tag = item.select_one("time, .date, .time-txt")
            articles.append({
                "source": "Economic Times",
                "title":  title,
                "url":    href,
                "date":   date_tag.get_text(strip=True) if date_tag else "Recent",
            })

        return articles if articles else [{"message": f"No ET articles found for '{topic}'"}]
    except Exception as exc:
        return [{"error": f"Economic Times fetch failed: {exc}"}]


def get_livemint_news(topic: str = "mutual funds") -> list[dict]:
    """
    Fetch mutual fund related news from LiveMint.

    Args:
        topic: News topic (e.g. "mutual funds", "SIP returns", "NFO")

    Returns:
        List of news articles.
    """
    try:
        url  = f"https://www.livemint.com/mutual-fund"
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        articles = []
        for item in soup.select("h2.headline a, .listingNew h2 a, article h2 a, .headlineSec h2 a")[:10]:
            title = item.get_text(strip=True)
            href  = item.get("href", "")
            if not href.startswith("http"):
                href = "https://www.livemint.com" + href
            if title:
                articles.append({
                    "source": "LiveMint",
                    "title":  title,
                    "url":    href,
                })

        return articles if articles else [{"message": "No LiveMint articles found"}]
    except Exception as exc:
        return [{"error": f"LiveMint fetch failed: {exc}"}]


def get_sebi_latest_circulars() -> list[dict]:
    """
    Fetch latest SEBI circulars and press releases relevant to mutual funds.

    Returns:
        List of SEBI circulars with title, date, and URL.
    """
    try:
        url  = "https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doLatest=yes"
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        circulars = []
        for row in soup.select("table tr")[1:10]:
            cols = row.select("td")
            if len(cols) < 2:
                continue
            link_tag = cols[0].select_one("a") or cols[1].select_one("a")
            if not link_tag:
                continue
            title = link_tag.get_text(strip=True)
            href  = link_tag.get("href", "")
            if href and not href.startswith("http"):
                href = "https://www.sebi.gov.in" + href
            date = cols[-1].get_text(strip=True) if len(cols) > 2 else "N/A"
            circulars.append({
                "source": "SEBI",
                "title":  title,
                "date":   date,
                "url":    href,
            })

        return circulars if circulars else [{"message": "No SEBI circulars parsed. Visit https://www.sebi.gov.in directly."}]
    except Exception as exc:
        return [{"error": f"SEBI circulars fetch failed: {exc}"}]


def get_rbi_policy_rates() -> dict:
    """
    Fetch current RBI policy rates (Repo Rate, Reverse Repo, CRR, SLR).
    These impact debt fund returns significantly.

    Returns:
        Dict with current RBI rates.
    """
    try:
        url  = "https://www.rbi.org.in/scripts/bs_viewcontent.aspx?Id=2009"
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        rates = {}
        for row in soup.select("table tr"):
            cols = row.select("td")
            if len(cols) >= 2:
                key = cols[0].get_text(strip=True)
                val = cols[-1].get_text(strip=True)
                if key and val and len(key) < 60:
                    rates[key] = val

        return {
            "source":  "RBI",
            "url":     url,
            "rates":   rates if rates else "Could not parse rates — visit RBI website directly",
            "note":    "Higher repo rate → debt funds may see lower returns in short term",
        }
    except Exception as exc:
        return {"error": f"RBI rates fetch failed: {exc}"}


def get_combined_market_news(topics: Optional[list[str]] = None) -> dict:
    """
    Fetch a combined market news digest from multiple sources.

    Args:
        topics: List of topics to search (default: ["mutual fund", "SIP", "NFO"])

    Returns:
        Dict with news from ET, LiveMint, and SEBI combined.
    """
    if not topics:
        topics = ["mutual fund", "SIP", "NFO"]

    all_news: dict = {}

    # ET news for first topic
    all_news["economic_times"] = get_et_markets_news(topics[0])

    # LiveMint
    all_news["livemint"] = get_livemint_news(topics[0])

    # SEBI circulars
    all_news["sebi_circulars"] = get_sebi_latest_circulars()

    return all_news
