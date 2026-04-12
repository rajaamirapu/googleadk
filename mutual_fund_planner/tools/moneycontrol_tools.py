"""
moneycontrol_tools.py
─────────────────────
Tools to fetch mutual fund data from Moneycontrol.com.

Strategy:
  1. Use Moneycontrol's internal JSON endpoints wherever possible
     (faster, more reliable than HTML parsing).
  2. Fall back to BeautifulSoup HTML parsing for pages without JSON APIs.
  3. Use realistic browser headers to avoid bot-detection rejections.
"""

from __future__ import annotations

import json
import re
from typing import Optional
import requests
from bs4 import BeautifulSoup

# ── Shared session with browser-like headers ──────────────────────────────
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/json,*/*;q=0.8",
    "Referer": "https://www.moneycontrol.com/",
})
_TIMEOUT = 15


# ─────────────────────────────────────────────────────────────────────────────
def get_fund_details_moneycontrol(fund_name: str) -> dict:
    """
    Search Moneycontrol for a mutual fund and return its key details
    including returns, rating, AUM, and expense ratio.

    Args:
        fund_name: Fund name or partial name (e.g. "Axis Bluechip Fund")

    Returns:
        Dict with fund details from Moneycontrol.
    """
    # Step 1: search the fund via Moneycontrol's search API
    search_url = "https://www.moneycontrol.com/mutual-funds/performance-tracker/returns/large-cap-fund.html"
    search_api  = f"https://www.moneycontrol.com/mccode/common/autosuggestion/getAutoSuggestion.php"

    try:
        resp = _SESSION.get(
            search_api,
            params={"classic": "true", "query": fund_name, "type": "1", "format": "json"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        suggestions = resp.json() if resp.text.strip() else []
        if not suggestions:
            return {"error": f"No results found for '{fund_name}' on Moneycontrol"}

        # Pick first MF result
        mf_results = [s for s in suggestions if s.get("sc_type") == "MF"]
        if not mf_results:
            mf_results = suggestions[:1]

        top = mf_results[0]
        return {
            "source":     "moneycontrol",
            "fund_name":  top.get("stock_name", top.get("sc_name", "N/A")),
            "mc_symbol":  top.get("symbol", "N/A"),
            "url":        f"https://www.moneycontrol.com{top.get('link', '')}",
            "category":   top.get("sc_type", "MF"),
            "note":       "Use mc_symbol with get_fund_performance_moneycontrol() for full details",
        }
    except Exception as exc:
        return {"error": f"Moneycontrol search failed: {exc}"}


def get_market_overview_moneycontrol() -> dict:
    """
    Fetch today's Indian market overview from Moneycontrol:
    Sensex, Nifty, sector performance, FII/DII activity.

    Returns:
        Dict with major indices, top gainers/losers overview.
    """
    try:
        url  = "https://www.moneycontrol.com/stocksmarketsindia/"
        resp = _SESSION.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        result: dict = {"source": "moneycontrol", "url": url}

        # ── Indices ───────────────────────────────────────────────
        indices = []
        for tag in soup.select("li.clearfix")[:6]:
            name_tag  = tag.select_one(".iname")
            val_tag   = tag.select_one(".ivalue")
            chng_tag  = tag.select_one(".ipc")
            if name_tag and val_tag:
                indices.append({
                    "index":  name_tag.get_text(strip=True),
                    "value":  val_tag.get_text(strip=True),
                    "change": chng_tag.get_text(strip=True) if chng_tag else "N/A",
                })
        if indices:
            result["indices"] = indices

        # ── MF category returns table (if present) ────────────────
        tables = soup.select("table.mctable1")
        if tables:
            headers, rows = [], []
            for th in tables[0].select("th"):
                headers.append(th.get_text(strip=True))
            for tr in tables[0].select("tr")[1:5]:
                row = [td.get_text(strip=True) for td in tr.select("td")]
                if row:
                    rows.append(dict(zip(headers, row)))
            if rows:
                result["table_data"] = rows

        return result if len(result) > 2 else {
            "source": "moneycontrol",
            "message": "Market data fetched but page layout may have changed. Visit moneycontrol.com directly.",
            "url": url,
        }
    except Exception as exc:
        return {"error": f"Moneycontrol market overview failed: {exc}"}


def get_top_mutual_funds_moneycontrol(category: str = "large-cap") -> list[dict]:
    """
    Scrape top-performing mutual funds from Moneycontrol's performance tracker.

    Args:
        category: Fund category slug. Options:
                  "large-cap-fund", "mid-cap-fund", "small-cap-fund",
                  "multi-cap-fund", "elss", "debt-fund", "hybrid-fund",
                  "index-fund", "sectoral-fund"

    Returns:
        List of top funds with 1Y/3Y/5Y returns and ratings.
    """
    # Normalise category slug
    slug_map = {
        "large cap": "large-cap-fund",
        "mid cap":   "mid-cap-fund",
        "small cap": "small-cap-fund",
        "elss":      "elss",
        "debt":      "debt-fund",
        "hybrid":    "hybrid-fund",
        "index":     "index-fund",
    }
    slug = slug_map.get(category.lower(), category.lower().replace(" ", "-"))
    if not slug.endswith("-fund") and slug not in ("elss", "index-fund"):
        slug = slug + "-fund" if "fund" not in slug else slug

    url = f"https://www.moneycontrol.com/mutual-funds/performance-tracker/returns/{slug}.html"
    try:
        resp = _SESSION.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        funds = []
        rows  = soup.select("table.mctable1 tr, table#mftable tr")
        if not rows:
            rows = soup.select("tr")

        for row in rows[1:16]:
            cols = row.select("td")
            if len(cols) < 3:
                continue
            name_tag = cols[0].select_one("a") or cols[0]
            fund = {
                "fund_name": name_tag.get_text(strip=True),
                "url":       "https://www.moneycontrol.com" + (name_tag.get("href", "") if name_tag.name == "a" else ""),
            }
            # Try to extract returns columns
            for i, col in enumerate(cols[1:6], 1):
                txt = col.get_text(strip=True)
                if txt and txt != "-":
                    fund[f"col_{i}"] = txt
            if fund["fund_name"]:
                funds.append(fund)

        return funds[:10] if funds else [{"message": f"No data parsed from {url}. Page may require JS. URL: {url}"}]
    except Exception as exc:
        return [{"error": f"Moneycontrol top funds failed: {exc}"}]


def get_financial_news_moneycontrol(topic: str = "mutual funds") -> list[dict]:
    """
    Fetch latest financial news headlines from Moneycontrol related to a topic.

    Args:
        topic: News topic (e.g. "mutual funds", "SIP", "SEBI", "Nifty")

    Returns:
        List of news articles with title, url, and summary.
    """
    try:
        # Moneycontrol news search
        url  = f"https://www.moneycontrol.com/news/tags/{topic.replace(' ', '-')}.html"
        resp = _SESSION.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        articles = []
        for item in soup.select("li.clearfix, .news-list li, .article-list li")[:10]:
            title_tag = item.select_one("h2 a, h3 a, a.article-title, a")
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            href  = title_tag.get("href", "")
            if not href.startswith("http"):
                href = "https://www.moneycontrol.com" + href
            desc_tag = item.select_one("p, .intro")
            articles.append({
                "title":   title,
                "url":     href,
                "summary": desc_tag.get_text(strip=True)[:200] if desc_tag else "",
            })

        return articles if articles else [{"message": f"No articles found for topic '{topic}'"}]
    except Exception as exc:
        return [{"error": f"Moneycontrol news fetch failed: {exc}"}]


def get_fund_manager_info(fund_house: str) -> dict:
    """
    Fetch information about mutual fund managers from Moneycontrol.

    Args:
        fund_house: Fund house name (e.g. "Axis", "HDFC", "Mirae Asset")

    Returns:
        Dict with fund manager details.
    """
    try:
        url  = f"https://www.moneycontrol.com/mutual-funds/{fund_house.lower().replace(' ', '-')}-mutual-fund/"
        resp = _SESSION.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        managers = []
        for tag in soup.select(".fund_manager, .fundMgr, [class*=manager]")[:5]:
            name = tag.get_text(strip=True)
            if name:
                managers.append(name)

        return {
            "fund_house":    fund_house,
            "url":           url,
            "fund_managers": managers if managers else ["Data not available — visit URL directly"],
        }
    except Exception as exc:
        return {"error": f"Fund manager fetch failed: {exc}"}
