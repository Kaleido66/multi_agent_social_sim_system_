import time
from typing import Any, Dict, List

class FreeTalkBoard:
    def __init__(self):
        self.posts: List[Dict[str, Any]] = []
        self.topics: Dict[str, Dict[str, Any]] = {}
        self.follow_map: Dict[str, List[str]] = {}

    def add_topic(self, topic: str) -> None:
        if topic not in self.topics:
            self.topics[topic] = {"topic": topic, "created_at": time.time()}

    def follow_topic(self, agent: str, topic: str) -> None:
        self.follow_map.setdefault(agent, [])
        if topic not in self.follow_map[agent]:
            self.follow_map[agent].append(topic)

    def post(self, agent: str, content: str, topic: str = "general") -> Dict[str, Any]:
        self.add_topic(topic)
        item = {
            "agent": agent,
            "content": content,
            "topic": topic,
            "timestamp": time.time(),
        }
        self.posts.append(item)
        return item

    def recent(self, k: int = 20) -> List[Dict[str, Any]]:
        return self.posts[-k:]

    def topics_list(self) -> List[str]:
        return list(self.topics.keys())

    def followed(self, agent: str) -> List[str]:
        return self.follow_map.get(agent, [])
