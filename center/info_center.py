import json
import os
import time
from typing import Any, Dict, List
from center.news_fetcher import fetch_news
from utils.runtime_store import runtime_store

class InformationCenter:
    """
    Information center: crawl -> process -> store -> distribute.
    """
    def __init__(self, db_path: str = "data/events.jsonl", max_db_size: int = 200, recency_hours: int = 72, max_fetch_items: int = 150):
        self.db_path = db_path
        self.max_db_size = max_db_size
        self.recency_hours = recency_hours
        self.max_fetch_items = max_fetch_items
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.ensure_case_event()

    def crawl(self, query: str) -> List[Dict[str, Any]]:
        results = fetch_news(query, max_items=self.max_fetch_items, recency_hours=self.recency_hours)
        if results:
            return results
        return [
            {
                "source": "mock_news",
                "title": "Global Summit Concludes with New Climate Pledge",
                "content": "Leaders agree on emissions targets and funding mechanisms.",
                "url": "https://example.com/news1",
                "timestamp": time.time(),
            },
        ]

    def _dynamic_event_schema(self, item: Dict[str, Any], llm=None) -> Dict[str, Any]:
        if llm is None:
            return {}
        prompt = (
            "你是事件建模助手。请根据新闻内容生成事件结构，输出JSON，包含字段："
            "tags(数组), environment(字符串), resources(对象，数值型), scene_rules(数组)。"
            "要求标签与内容相关，资源数值0-20。只输出JSON。\n"
            f"标题：{item.get('title','')}\n内容：{item.get('content','')}"
        )
        try:
            raw = llm.generate(prompt)
            try:
                runtime_store.log_agent("info_center", "dynamic_event_schema", prompt, raw)
            except Exception:
                pass
            if isinstance(raw, dict):
                return raw
            if not isinstance(raw, str):
                return {}
            text = raw.strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text.replace("json", "", 1).strip()
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def process(self, raw_items: List[Dict[str, Any]], llm=None) -> List[Dict[str, Any]]:
        seen = set()
        events = []
        for item in raw_items:
            key = (item.get("title", ""), item.get("url", ""))
            if key in seen:
                continue
            seen.add(key)
            title = item.get("title", "")
            if not title and llm is not None:
                prompt = f"请根据内容生成标题：{item.get('content', '')}"
                title = llm.generate(prompt)
                try:
                    runtime_store.log_agent("info_center", "generate_title", prompt, title)
                except Exception:
                    pass

            schema = self._dynamic_event_schema(item, llm=llm)
            if not isinstance(schema, dict):
                schema = {}
            event_ts = item.get("timestamp") or time.time()
            event = {
                "id": f"evt_{int(time.time()*1000)}",
                "title": title or "未命名事件",
                "content": item.get("content", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "timestamp": event_ts,
                "published_at": item.get("published_at", ""),
                "type": "news_event",
                "tags": schema.get("tags") or ["public", "media"],
                "world_state": {
                    "resources": schema.get("resources") or {"attention": 10, "trust": 5},
                    "environment": schema.get("environment") or "公共讨论场景",
                    "scene_rules": schema.get("scene_rules") or [],
                },
            }
            events.append(event)
        return events

    def ensure_case_event(self) -> None:
        case_id = "evt_case_multiparty_china"
        case_title = "多党交替执政是否为中国政体最优解的跨国讨论"
        existing = self._load_existing()
        event = {
            "id": case_id,
            "title": case_title,
            "content": "来自不同国家与政治背景的社会智能体围绕“对中国来说效仿多党交替执政是否是中国政体的最优解”展开多轮讨论。",
            "source": "case_study",
            "url": "",
            "timestamp": time.time(),
            "published_at": "",
            "type": "case_event",
            "tags": ["policy", "public", "media", "debate"],
            "participant_ids": [
                "agent_007",
                "agent_036",
                "agent_037",
                "agent_038",
                "agent_039",
                "agent_040",
                "agent_042",
                "agent_014",
                "agent_016",
                "agent_033",
            ],
            "agent_goals": {
                "记者": "追问制度变迁的可行性与风险，强调信息透明与多方声音。",
                "政策制定者": "评估制度稳定性与社会成本，强调治理连续性。",
                "企业高管": "关注经济稳定与投资预期，评估制度对营商环境影响。",
                "市政官员": "从治理效率和公共服务连续性角度分析。",
                "投资人": "关注制度更替带来的市场不确定性与机会。",
                "地方议员": "强调多元代表性与问责机制的作用。",
                "法学研究者": "从宪制与法治框架讨论制度路径。",
                "教师": "关注教育公平与社会参与度的变化。",
                "多党倡议者": "积极宣传多党交替的必要性，强化程序正义与权力制衡。",
            },
            "world_state": {
                "resources": {"attention": 12, "trust": 6},
                "environment": "跨国政治制度讨论场景",
                "scene_rules": ["允许观点影响", "记录立场变化"],
            },
        }
        for e in existing:
            if e.get("id") == case_id:
                # merge existing with latest spec, but preserve user-edited participants
                merged = e.copy()
                merged.update(event)
                if e.get("participant_ids"):
                    # always honor user-edited participant_ids
                    merged["participant_ids"] = e.get("participant_ids")
                self.store([merged])
                return
        self.store([event])

    def _load_existing(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.db_path):
            return []
        items = []
        # robust decode to avoid crashing on bad bytes
        with open(self.db_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        continue
        return items

    def load_existing(self) -> List[Dict[str, Any]]:
        return self._load_existing()

    def store(self, events: List[Dict[str, Any]]) -> None:
        # merge with existing so agents can pick from new + old
        existing = self._load_existing()
        merged = {}
        for e in existing + events:
            # prefer id as unique key to preserve user-added events
            if e.get("id"):
                key = ("id", e.get("id"))
            else:
                key = ("tu", e.get("title", ""), e.get("url", ""))
            prev = merged.get(key)
            if prev is None:
                merged[key] = e
            else:
                # keep the newer timestamp if same id
                if e.get("timestamp", 0) >= prev.get("timestamp", 0):
                    merged[key] = e
        # keep newest first
        merged_items = list(merged.values())
        merged_items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        merged_items = merged_items[: self.max_db_size]
        with open(self.db_path, "w", encoding="utf-8") as f:
            for evt in merged_items:
                f.write(json.dumps(evt, ensure_ascii=False) + "\n")

    def summarize_events(self, events: List[Dict[str, Any]], llm=None) -> str:
        if not events:
            return "暂无事件"
        if llm is None:
            return "\n".join([f"- {e['title']}" for e in events])
        prompt = "请用中文总结以下事件，生成简短报告：\n" + "\n".join([e["title"] + ": " + e["content"] for e in events])
        out = llm.generate(prompt)
        try:
            runtime_store.log_agent("info_center", "summarize_events", prompt, out)
        except Exception:
            pass
        return out

    def distribute(self, events: List[Dict[str, Any]], llm=None) -> Dict[str, Any]:
        summary = self.summarize_events(events, llm=llm)
        return {
            "events": events,
            "summary": summary,
        }

