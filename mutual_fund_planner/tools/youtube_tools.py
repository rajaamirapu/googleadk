"""
youtube_tools.py
────────────────
Tools to fetch and analyse YouTube video transcripts.

Primary use-case: pull CNBC TV18 / ET Now / Zee Business market discussion
transcripts and extract investment insights.

Libraries used:
  - youtube_transcript_api  (pip install youtube-transcript-api)
  - yt_dlp                  (pip install yt-dlp)   ← for video metadata search
"""

from __future__ import annotations

import re
from typing import Optional
import requests


# ── YouTube Transcript API ─────────────────────────────────────────────────
try:
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
    YT_TRANSCRIPT_AVAILABLE = True
except ImportError:
    YT_TRANSCRIPT_AVAILABLE = False


def _extract_video_id(url_or_id: str) -> Optional[str]:
    """Extract YouTube video ID from a URL or return the ID as-is."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for pat in patterns:
        m = re.search(pat, url_or_id)
        if m:
            return m.group(1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
def get_youtube_transcript(video_url: str, max_chars: int = 8000) -> dict:
    """
    Fetch the transcript of a YouTube video (auto-generated or manual).

    Works with any YouTube URL including CNBC TV18, ET Now, Zee Business.

    Args:
        video_url: Full YouTube URL or video ID
                   (e.g. "https://www.youtube.com/watch?v=abcd1234567")
        max_chars: Maximum characters to return (default 8000 to fit LLM context)

    Returns:
        Dict with video_id, full_transcript (truncated), and segment count.
    """
    if not YT_TRANSCRIPT_AVAILABLE:
        return {"error": "youtube_transcript_api not installed. Run: pip install youtube-transcript-api"}

    video_id = _extract_video_id(video_url)
    if not video_id:
        return {"error": f"Could not extract video ID from: {video_url}"}

    try:
        # Try English first, then Hindi, then auto-generated
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        for lang in ("en", "en-IN", "hi"):
            try:
                transcript = transcript_list.find_transcript([lang])
                break
            except Exception:
                pass
        if transcript is None:
            # Fall back to any available
            transcript = transcript_list.find_generated_transcript(["en", "en-IN", "hi"])

        segments = transcript.fetch()
        full_text = " ".join(seg["text"] for seg in segments)

        # Truncate if too long
        truncated = len(full_text) > max_chars
        if truncated:
            full_text = full_text[:max_chars] + "... [truncated]"

        return {
            "video_id":       video_id,
            "video_url":      f"https://www.youtube.com/watch?v={video_id}",
            "language":       transcript.language_code,
            "segment_count":  len(segments),
            "char_count":     len(full_text),
            "truncated":      truncated,
            "transcript":     full_text,
        }

    except TranscriptsDisabled:
        return {
            "video_id": video_id,
            "error":    "Transcripts are disabled for this video.",
        }
    except NoTranscriptFound:
        return {
            "video_id": video_id,
            "error":    "No transcript found. Try a different video.",
        }
    except Exception as exc:
        return {"video_id": video_id, "error": str(exc)}


def get_transcript_summary_data(video_url: str) -> dict:
    """
    Fetch a YouTube transcript and extract key financial topics mentioned.
    Returns the transcript ready for LLM analysis.

    Args:
        video_url: YouTube URL (CNBC TV18, ET Now, or any financial channel)

    Returns:
        Dict with transcript and extracted financial keywords.
    """
    result = get_youtube_transcript(video_url, max_chars=10000)
    if "error" in result:
        return result

    transcript = result.get("transcript", "")

    # Extract financial keywords (basic pattern matching)
    keywords = {
        "funds_mentioned":   _extract_matches(transcript, r"\b(?:fund|scheme|SIP|NAV|ELSS|NFO)\b"),
        "indices_mentioned": _extract_matches(transcript, r"\b(?:Nifty|Sensex|BSE|NSE|Nifty50|BankNifty)\b"),
        "stocks_mentioned":  _extract_matches(transcript, r"\b[A-Z]{2,8}\b"),  # crude uppercase ticker detection
        "numbers_mentioned": re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:%|cr|lakh|crore|%)\b", transcript, re.I)[:10],
    }

    result["financial_keywords"] = keywords
    return result


def _extract_matches(text: str, pattern: str) -> list[str]:
    """Extract unique matches of a pattern from text."""
    matches = re.findall(pattern, text, re.I)
    seen = set()
    unique = []
    for m in matches:
        if m.lower() not in seen:
            seen.add(m.lower())
            unique.append(m)
    return unique[:15]


def search_cnbc_videos_on_youtube(topic: str, max_results: int = 5) -> list[dict]:
    """
    Search for CNBC TV18 or ET Now YouTube videos on a specific topic.
    Uses YouTube's public search (no API key required for basic search).

    Args:
        topic:       Topic to search (e.g. "mutual fund SIP 2024", "Nifty outlook")
        max_results: Max number of results (default 5)

    Returns:
        List of videos with title, url, and channel.
    """
    # Use YouTube's suggestion/search API (no key needed for basic results)
    query = f"CNBC TV18 {topic}"
    try:
        resp = requests.get(
            "https://www.youtube.com/results",
            params={"search_query": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        resp.raise_for_status()

        # Extract video IDs and titles from the page source
        video_ids    = re.findall(r'"videoId":"([A-Za-z0-9_-]{11})"', resp.text)
        video_titles = re.findall(r'"title":\{"runs":\[\{"text":"([^"]+)"', resp.text)

        results = []
        seen_ids = set()
        for vid_id, title in zip(video_ids, video_titles):
            if vid_id in seen_ids:
                continue
            seen_ids.add(vid_id)
            results.append({
                "title":   title,
                "url":     f"https://www.youtube.com/watch?v={vid_id}",
                "video_id": vid_id,
                "channel": "CNBC TV18 / ET Now (unverified)",
            })
            if len(results) >= max_results:
                break

        return results if results else [{"message": f"No results found. Try get_youtube_transcript() with a direct URL."}]
    except Exception as exc:
        return [{"error": f"YouTube search failed: {exc}"}]


def get_multiple_transcripts(video_urls: list[str]) -> list[dict]:
    """
    Fetch transcripts for multiple YouTube videos at once.

    Args:
        video_urls: List of YouTube URLs (max 5)

    Returns:
        List of transcript results for each video.
    """
    return [get_youtube_transcript(url) for url in video_urls[:5]]
