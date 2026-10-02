from typing import Dict, Any

# Social tools (OpenClaw-style behaviors)

def publish_news(agent: str, content: str, event_id: str) -> Dict[str, Any]:
    return {"tool": "publish_news", "agent": agent, "event_id": event_id, "content": content}


def comment(agent: str, content: str, target: str) -> Dict[str, Any]:
    return {"tool": "comment", "agent": agent, "target": target, "content": content}


def propose_policy(agent: str, content: str, event_id: str) -> Dict[str, Any]:
    return {"tool": "propose_policy", "agent": agent, "event_id": event_id, "content": content}


def mobilize(agent: str, content: str, topic: str) -> Dict[str, Any]:
    return {"tool": "mobilize", "agent": agent, "topic": topic, "content": content}


def adjust_resource(agent: str, resource: str, delta: int) -> Dict[str, Any]:
    return {"tool": "adjust_resource", "agent": agent, "resource": resource, "delta": delta}


def use_resource(agent: str, resource: str, amount: int) -> Dict[str, Any]:
    return {"tool": "use_resource", "agent": agent, "resource": resource, "amount": amount}


def help_actor(agent: str, target: str, content: str) -> Dict[str, Any]:
    return {"tool": "help", "agent": agent, "target": target, "content": content}


def harm_actor(agent: str, target: str, content: str) -> Dict[str, Any]:
    return {"tool": "harm", "agent": agent, "target": target, "content": content}


def hunt(agent: str, target: str) -> Dict[str, Any]:
    return {"tool": "hunt", "agent": agent, "target": target}


def move(agent: str, x: float, y: float) -> Dict[str, Any]:
    return {"tool": "move", "agent": agent, "x": x, "y": y}


def negotiate(agent: str, target: str, content: str) -> Dict[str, Any]:
    return {"tool": "negotiate", "agent": agent, "target": target, "content": content}


def donate(agent: str, resource: str, amount: int) -> Dict[str, Any]:
    return {"tool": "donate", "agent": agent, "resource": resource, "amount": amount}


def boycott(agent: str, target: str, content: str) -> Dict[str, Any]:
    return {"tool": "boycott", "agent": agent, "target": target, "content": content}


def verify(agent: str, content: str) -> Dict[str, Any]:
    return {"tool": "verify", "agent": agent, "content": content}


def investigate(agent: str, content: str) -> Dict[str, Any]:
    return {"tool": "investigate", "agent": agent, "content": content}


def collaborate(agent: str, target: str, content: str) -> Dict[str, Any]:
    return {"tool": "collaborate", "agent": agent, "target": target, "content": content}
