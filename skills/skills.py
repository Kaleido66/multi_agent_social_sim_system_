import json
from typing import Dict, Any

# Each skill returns a structured dict

def search(query: str) -> Dict[str, Any]:
    # Placeholder local search with a simple scenario
    if "past week" in query or "international" in query:
        results = [
            {"id": "news1", "title": "Global Summit Concludes with New Climate Pledge", "snippet": "Leaders agree on emissions targets..."},
            {"id": "news2", "title": "Regional Conflict De-escalation Talks Resume", "snippet": "Diplomatic channels reopen..."},
        ]
    else:
        results = [
            {"id": "doc1", "title": "Urban Safety Report", "snippet": "Crime rate trends..."},
            {"id": "doc2", "title": "Youth Employment Study", "snippet": "Unemployment impacts..."},
        ]
    return {"skill": "search", "query": query, "results": results}


def read(doc_id: str) -> Dict[str, Any]:
    # Placeholder read
    content = f"Content of {doc_id}: This is a synthetic document used for simulation."
    return {"skill": "read", "doc_id": doc_id, "content": content}


def extract(content: str) -> Dict[str, Any]:
    # Simple extraction
    facts = ["fact_a", "fact_b"] if content else []
    return {"skill": "extract", "facts": facts}


def summarize(content: str) -> Dict[str, Any]:
    summary = (content[:120] + "...") if len(content) > 120 else content
    return {"skill": "summarize", "summary": summary}


def store(memory, item: Dict[str, Any], mem_type: str = "episodic", meta: Dict[str, Any] = None) -> Dict[str, Any]:
    payload = json.dumps(item, ensure_ascii=False)
    memory.add_memory(payload, mem_type, meta=meta or {})
    return {"skill": "store", "stored": True, "type": mem_type}
