import json
import os
import time
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from utils.runtime_store import runtime_store


class InductionLibrary:
    """
    Induction content library backed by JSON/YAML files.
    File structure: data/inductions/<event_id>.json or .yaml
    Each file contains a list of items with fields:
    - id, category, content, interest_tags, target_groups, stance_direction, intensity
    - enabled: bool (default True)
    - use_in_sim: bool (default True)
    """
    def __init__(self, base_dir: str = "data/inductions", max_db_size: int = 500):
        self.base_dir = base_dir
        self.max_db_size = max_db_size
        os.makedirs(self.base_dir, exist_ok=True)

    def _load_file(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if path.endswith(".json"):
                data = json.loads(text)
            elif path.endswith(".yaml") or path.endswith(".yml"):
                if yaml is None:
                    return []
                data = yaml.safe_load(text)
            else:
                return []
            if isinstance(data, list):
                return data
        except Exception:
            return []
        return []

    def load_event_items(self, event_id: str) -> List[Dict[str, Any]]:
        if not event_id:
            return []
        candidates = [
            os.path.join(self.base_dir, f"{event_id}.json"),
            os.path.join(self.base_dir, f"{event_id}.yaml"),
            os.path.join(self.base_dir, f"{event_id}.yml"),
        ]
        items: List[Dict[str, Any]] = []
        for path in candidates:
            items = self._load_file(path)
            if items:
                break
        # normalize
        normalized = []
        now = time.time()
        for item in items:
            if not isinstance(item, dict):
                continue
            it = dict(item)
            it.setdefault("event_id", event_id)
            it.setdefault("timestamp", now)
            it.setdefault("enabled", True)
            it.setdefault("use_in_sim", True)
            it.setdefault("intensity", 3)
            normalized.append(it)
        return normalized

    def filter_items(
        self,
        event_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
        stance_direction: Optional[str] = None,
        target_groups: Optional[List[str]] = None,
        target_agent: Optional[str] = None,
        round_idx: Optional[int] = None,
        min_intensity: Optional[int] = None,
        max_intensity: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        items = self.load_event_items(event_id or "") if event_id else []
        results = []
        for item in items:
            if not item.get("enabled", True) or not item.get("use_in_sim", True):
                continue
            if categories and item.get("category") not in categories:
                continue
            if stance_direction and item.get("stance_direction") != stance_direction:
                continue
            # round gating
            if round_idx is not None:
                rounds = item.get("rounds")
                if isinstance(rounds, list) and round_idx not in rounds:
                    continue
                rmin = item.get("round_min")
                rmax = item.get("round_max")
                if rmin is not None and round_idx < int(rmin):
                    continue
                if rmax is not None and round_idx > int(rmax):
                    continue
            if target_groups:
                groups = item.get("target_groups", []) or []
                if not any(g in groups for g in target_groups):
                    continue
            # target agent gating
            if target_agent:
                targets = item.get("target_agents", []) or []
                if targets and target_agent not in targets:
                    continue
            intensity = int(item.get("intensity", 3) or 3)
            if min_intensity is not None and intensity < min_intensity:
                continue
            if max_intensity is not None and intensity > max_intensity:
                continue
            results.append(item)
        return results

    def auto_generate(self, llm, event: Dict[str, Any], count: int = 4) -> List[Dict[str, Any]]:
        if llm is None:
            return []
        prompt = (
            "你是诱导内容生成器。根据事件生成诱导内容，输出JSON数组。"
            "每个对象包含字段：category(news/official/public/comparison), content, interest_tags(数组), "
            "target_groups(数组), stance_direction(pro_multiparty/anti_multiparty/neutral), intensity(1-5), enabled, use_in_sim。"
            "请贴合事件内容，不要编造具体事实来源。只输出JSON数组。\n"
            f"事件标题：{event.get('title','')}\n事件内容：{event.get('content','')}\n"
            f"需要数量：{count}"
        )
        raw = llm.generate(prompt)
        runtime_store.log_agent("induction", "auto_generate", prompt, raw)
        text = raw.strip() if isinstance(raw, str) else ""
        if text.startswith("```"):
            text = text.strip("`").replace("json", "", 1).strip()
        try:
            items = json.loads(text)
        except Exception:
            return []
        if not isinstance(items, list):
            return []
        now = time.time()
        for item in items:
            item["event_id"] = event.get("id", "")
            item.setdefault("timestamp", now)
            item.setdefault("enabled", True)
            item.setdefault("use_in_sim", True)
            if not item.get("id"):
                item["id"] = f"ind_auto_{int(now*1000)}_{abs(hash(item.get('content','')))%10000}"
        return items

