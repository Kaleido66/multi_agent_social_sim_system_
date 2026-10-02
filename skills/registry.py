from typing import Dict, Any, Callable

class SkillRegistry:
    def __init__(self):
        self._skills: Dict[str, Callable[..., Dict[str, Any]]] = {}

    def register(self, name: str, func: Callable[..., Dict[str, Any]]) -> None:
        self._skills[name] = func

    def call(self, name: str, *args, **kwargs) -> Dict[str, Any]:
        if name not in self._skills:
            raise KeyError(f"Skill not registered: {name}")
        return self._skills[name](*args, **kwargs)
