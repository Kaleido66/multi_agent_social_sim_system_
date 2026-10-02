from typing import Dict, Any, Callable

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable[..., Dict[str, Any]]] = {}

    def register(self, name: str, func: Callable[..., Dict[str, Any]]) -> None:
        self._tools[name] = func

    def call(self, name: str, *args, **kwargs) -> Dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Tool not registered: {name}")
        return self._tools[name](*args, **kwargs)

    def list_tools(self):
        return list(self._tools.keys())
