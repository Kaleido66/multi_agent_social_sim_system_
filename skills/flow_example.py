import os
import sys
import importlib

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SKILLS_DIR = os.path.abspath(os.path.dirname(__file__))
_norm = lambda p: os.path.normcase(os.path.abspath(p))

sys.path = [p for p in sys.path if _norm(p) != _norm(SKILLS_DIR)]

if "skills" in sys.modules:
    mod = sys.modules.get("skills")
    mod_path = getattr(mod, "__file__", "") or ""
    if _norm(mod_path).startswith(_norm(SKILLS_DIR)):
        del sys.modules["skills"]

VectorMemory = importlib.import_module("memory.memory_store").VectorMemory
skills_mod = importlib.import_module("skills.skills")
registry_mod = importlib.import_module("skills.registry")

search = skills_mod.search
read = skills_mod.read
extract = skills_mod.extract
summarize = skills_mod.summarize
store = skills_mod.store
SkillRegistry = registry_mod.SkillRegistry


def build_registry() -> SkillRegistry:
    reg = SkillRegistry()
    reg.register("search", search)
    reg.register("read", read)
    reg.register("extract", extract)
    reg.register("summarize", summarize)
    reg.register("store", store)
    return reg


def run_info_pipeline(query: str) -> dict:
    memory = VectorMemory(persist_path=None, user_id="flow_example")
    reg = build_registry()

    s = reg.call("search", query)
    doc_id = s["results"][0]["id"]
    r = reg.call("read", doc_id)
    e = reg.call("extract", r["content"])
    sm = reg.call("summarize", r["content"])
    st = reg.call("store", memory, {"raw": r, "facts": e, "summary": sm}, mem_type="episodic")

    return {"search": s, "read": r, "extract": e, "summarize": sm, "store": st}


if __name__ == "__main__":
    print(run_info_pipeline("public safety"))
