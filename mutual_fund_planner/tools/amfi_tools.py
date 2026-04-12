"""
amfi_tools.py
─────────────
Tools that pull official mutual fund data from AMFI (Association of Mutual
Funds in India) via the `mftool` library and direct AMFI API calls.

Data is authoritative and free — no scraping needed.
"""

from __future__ import annotations
import re
from typing import Optional
import requests

# ── mftool: wraps AMFI official data feed (lazy init — no network at import) ─
try:
    from mftool import Mftool as _MftoolClass
    MFTOOL_AVAILABLE = True
except ImportError:
    MFTOOL_AVAILABLE = False
    _MftoolClass = None  # type: ignore

_mf_instance = None   # created on first use


def _require_mftool():
    """Return a shared Mftool instance, creating it lazily on first call."""
    global _mf_instance
    if not MFTOOL_AVAILABLE:
        raise RuntimeError("mftool is not installed. Run: pip install mftool")
    if _mf_instance is None:
        _mf_instance = _MftoolClass()
    return _mf_instance


# ─────────────────────────────────────────────────────────────────────────────
def get_fund_nav(scheme_code: str) -> dict:
    """
    Fetch the latest NAV for a mutual fund by its AMFI scheme code.

    Args:
        scheme_code: AMFI scheme code (e.g. "119598" for Axis Bluechip Fund)

    Returns:
        Dict with fund name, date, nav, and scheme_code.
    """
    mf = _require_mftool()
    try:
        data = mf.get_scheme_quote(scheme_code)
        if not data:
            return {"error": f"No data found for scheme code {scheme_code}"}
        return {
            "scheme_code": scheme_code,
            "fund_name":   data.get("scheme_name", "N/A"),
            "nav":         data.get("nav", "N/A"),
            "date":        data.get("last_updated", "N/A"),
        }
    except Exception as exc:
        return {"error": str(exc)}


def search_mutual_funds(query: str) -> list[dict]:
    """
    Search for mutual funds by name keyword.

    Args:
        query: Fund name keyword (e.g. "Axis Bluechip", "HDFC Mid Cap")

    Returns:
        List of matching funds with scheme_code and fund_name.
    """
    mf = _require_mftool()
    try:
        all_schemes = mf.get_scheme_codes(as_json=False)
        query_lower = query.lower()
        matches = [
            {"scheme_code": str(code), "fund_name": name}
            for code, name in all_schemes.items()
            if query_lower in name.lower()
        ]
        return matches[:20]  # cap at 20 results
    except Exception as exc:
        return [{"error": str(exc)}]


def get_fund_historical_nav(scheme_code: str, days: int = 365) -> dict:
    """
    Fetch historical NAV data for a fund (up to the last N days).

    Args:
        scheme_code: AMFI scheme code
        days:        Number of past days to fetch (default 365)

    Returns:
        Dict with fund_name and list of {date, nav} records.
    """
    mf = _require_mftool()
    try:
        data = mf.get_scheme_historical_nav(scheme_code, as_json=False)
        if not data or "data" not in data:
            return {"error": "No historical data found"}
        records = data["data"][:days]
        # Compute simple stats
        navs = []
        for r in records:
            try:
                navs.append(float(r["nav"]))
            except (ValueError, KeyError):
                pass
        stats = {}
        if navs:
            stats["current_nav"] = navs[0]
            stats["nav_1y_ago"]  = navs[-1] if len(navs) >= 365 else navs[-1]
            stats["1y_return_pct"] = round((navs[0] - navs[-1]) / navs[-1] * 100, 2) if navs[-1] else None
            stats["nav_high_1y"] = round(max(navs), 4)
            stats["nav_low_1y"]  = round(min(navs), 4)
        return {
            "scheme_code": scheme_code,
            "fund_name":   data.get("fund_house", "") + " " + data.get("scheme_name", ""),
            "records_fetched": len(records),
            "stats": stats,
            "recent_10_days": records[:10],
        }
    except Exception as exc:
        return {"error": str(exc)}


def get_top_funds_by_category(category: str, top_n: int = 10) -> list[dict]:
    """
    Get top-performing mutual funds by category using AMFI data.

    Args:
        category: Fund category keyword — e.g. "large cap", "mid cap",
                  "small cap", "ELSS", "debt", "hybrid", "index"
        top_n:    Number of funds to return (default 10)

    Returns:
        List of funds matching the category with their NAVs.
    """
    mf = _require_mftool()
    try:
        all_schemes = mf.get_scheme_codes(as_json=False)
        cat_lower = category.lower()
        matched = [
            {"scheme_code": str(code), "fund_name": name}
            for code, name in all_schemes.items()
            if cat_lower in name.lower()
        ]
        return matched[:top_n]
    except Exception as exc:
        return [{"error": str(exc)}]


def compare_funds(scheme_codes: list[str]) -> list[dict]:
    """
    Compare multiple funds by fetching their current NAVs.

    Args:
        scheme_codes: List of AMFI scheme codes to compare (max 5)

    Returns:
        List of {scheme_code, fund_name, nav, date} for each fund.
    """
    results = []
    for code in scheme_codes[:5]:
        results.append(get_fund_nav(code))
    return results


def get_all_fund_houses() -> list[str]:
    """
    Get a list of all mutual fund houses registered with AMFI.

    Returns:
        List of fund house names.
    """
    try:
        resp = requests.get("https://api.mfapi.in/mf", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        houses = sorted({f.get("schemeName", "").split(" ")[0] for f in data if f.get("schemeName")})
        return houses[:50]
    except Exception as exc:
        return [f"Error fetching fund houses: {exc}"]
