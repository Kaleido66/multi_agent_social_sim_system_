from typing import Dict, Any
from utils.runtime_store import runtime_store

class InfoActionAgent:
    """
    Minimal info-center agent: atomic actions only, no observe/reflect.
    """
    def __init__(
        self,
        name: str,
        llm=None,
        skills=None,
        language_mode: str = "zh",
        classical_translate: bool = False,
        use_native_language: bool = False,
        translate_to_zh: bool = True,
    ):
        self.name = name
        self.llm = llm
        self.skills = skills
        self.language_mode = language_mode
        self.classical_translate = classical_translate
        self.use_native_language = use_native_language
        self.translate_to_zh = translate_to_zh

    def run(self, query: str) -> Dict[str, Any]:
        # simple atomic pipeline using skills
        result = {"query": query}
        if self.skills:
            s = self.skills.call("search", query)
            result["search"] = s
            if s.get("results"):
                doc_id = s["results"][0].get("id")
                r = self.skills.call("read", doc_id)
                e = self.skills.call("extract", r.get("content", ""))
                sm = self.skills.call("summarize", r.get("content", ""))
                st = self.skills.call("store", self, {"raw": r, "facts": e, "summary": sm}, mem_type="episodic")
                result.update({"read": r, "extract": e, "summary": sm, "store": st})
        runtime_store.log_agent(self.name, "info_pipeline", str(query), str(result))
        return result

    def add_memory(self, content: str, mem_type: str, meta: Dict[str, Any]):
        # no-op for info agent
        return None
