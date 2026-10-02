import xml.etree.ElementTree as ET
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

DEFAULT_RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.theguardian.com/technology/rss",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "https://feeds.reuters.com/Reuters/worldNews",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://rsshub.app/bbc/zhongwen/trad",
    "https://rsshub.app/zhihu/hot",
    "https://rsshub.app/people/opinions",
    "https://rsshub.app/weibo/search/hot",
]


def _parse_pubdate(text: str) -> Optional[datetime]:
    if not text:
        return None
    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def fetch_rss(url: str, timeout: int = 10) -> List[Dict[str, Any]]:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    items = []
    for item in root.iter("item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        desc = item.findtext("description") or ""
        pubdate = item.findtext("pubDate") or item.findtext("updated") or ""
        pub_dt = _parse_pubdate(pubdate)
        ts = pub_dt.timestamp() if pub_dt else None
        items.append({
            "source": "rss",
            "title": title.strip(),
            "content": desc.strip(),
            "url": link.strip(),
            "published_at": pub_dt.isoformat() if pub_dt else "",
            "timestamp": ts,
        })
    return items


def fetch_news(query: str, max_items: int = 150, recency_hours: int = 72) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).timestamp()
    for feed in DEFAULT_RSS_FEEDS:
        try:
            results.extend(fetch_rss(feed))
        except Exception:
            continue
    cutoff = now - recency_hours * 3600
    filtered = []
    for it in results:
        ts = it.get("timestamp")
        if ts is None or ts >= cutoff:
            filtered.append(it)
    return filtered[:max_items]
