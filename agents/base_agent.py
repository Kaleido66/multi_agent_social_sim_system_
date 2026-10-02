from dataclasses import dataclass, field
from typing import Dict, Any
from models.base import BaseLLM
from skills.registry import SkillRegistry
from utils.runtime_store import runtime_store
from utils.language import build_language_instruction, build_translation_instruction, build_translate_to_zh_instruction
from utils.translation import robust_translate
from utils.text_fix import repair_text
import json

@dataclass
class Persona:
    role: str
    goal: str
    personality: str
    name: str = ""
    gender: str = ""
    age: int = 0
    class_level: str = ""
    assets: str = ""
    values: str = ""
    belief: str = ""
    nationality: str = ""
    political_identity: str = ""
    regime_stance: str = ""
    authority_preference: str = ""
    legitimacy_source: str = ""
    distribution_preference: str = ""
    polity_view: str = ""
    region: str = ""
    language: str = "中文"
    faction: str = ""
    avatar: str = ""
    income_level: str = ""
    occupation_type: str = ""
    work_intensity: str = ""
    has_children: bool = False
    education_anxiety: float = 0.0
    family_burden: float = 0.0
    value_weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class BaseAgent:
    name: str
    persona: Persona
    llm: BaseLLM
    memory: Any
    skills: SkillRegistry = field(default_factory=SkillRegistry)
    language_mode: str = "zh"
    classical_translate: bool = False
    use_native_language: bool = False
    translate_to_zh: bool = True
    memory_summary_threshold: int = 60
    max_reply_chars: int = 200

    def _ensure_llm(self):
        if self.llm is None:
            raise RuntimeError(f"LLM is required for agent {self.name}")

    def _call_llm(self, step: str, prompt: str, allow_translate: bool = True) -> str:
        self._ensure_llm()
        lang_instr = build_language_instruction(
            self.language_mode,
            persona_language=self.persona.language,
            use_native_language=self.use_native_language,
        )
        # If simulation runs in English mode, first translate the assembled prompt
        # into English (so the prompt content and examples are in the target language),
        # log that translation, then call the LLM with the English prompt.
        full_prompt = f"{lang_instr}\n{prompt}"
        full_prompt = repair_text(full_prompt)
        # enforce concise output instruction to reduce overly long replies
        try:
            max_chars = int(getattr(self, "max_reply_chars", 0) or 0)
        except Exception:
            max_chars = 0
        if max_chars and step not in {"decide"} and ("请将回答控制" not in full_prompt):
            length_instr = f"\n请将回答控制在不超过{max_chars}字以内，尽量精炼、分点陈述。"
            full_prompt = full_prompt + length_instr
        if (self.language_mode or "").lower() == "en" and allow_translate:
            try:
                # use robust_translate which will try multiple prompt strategies and log attempts
                translated = robust_translate(self.llm, prompt, src="zh", tgt="en")
                runtime_store.log_agent(self.name, f"prompt_translate_to_en", prompt, translated)
                full_prompt = f"{lang_instr}\n{translated}"
            except Exception as exc:
                runtime_store.log_agent(self.name, "prompt_translate_error", prompt, f"ERROR: {exc}")
                full_prompt = f"{lang_instr}\n{prompt}"
        try:
            output = self.llm.generate(full_prompt)
        except Exception as exc:
            # avoid breaking the run on LLM errors (e.g., context length)
            runtime_store.log_agent(self.name, f"{step}_error", full_prompt, f"ERROR: {exc}")
            return ""
        output = repair_text(output)
        runtime_store.log_agent(self.name, step, full_prompt, output)
        if self.language_mode == "classical" and self.classical_translate:
            translate_prompt = build_translation_instruction() + "\n" + output
            try:
                translated = self.llm.generate(translate_prompt)
                runtime_store.log_agent(self.name, "classical_translate", translate_prompt, translated)
                output = translated
            except Exception as exc:
                runtime_store.log_agent(self.name, "classical_translate_error", translate_prompt, f"ERROR: {exc}")
        if self.translate_to_zh and allow_translate and lang_instr != "请用中文回答。":
            try:
                zh_text = robust_translate(self.llm, output, src="en", tgt="zh")
                runtime_store.log_agent(self.name, "translate_to_zh", output, zh_text)
                # enforce max length post-translation as well
                if max_chars and step not in {"decide"} and isinstance(zh_text, str) and len(zh_text) > max_chars:
                    truncated = zh_text[: max_chars]
                    runtime_store.log_agent(self.name, "truncate_translate", zh_text, truncated)
                    return truncated
                return zh_text
            except Exception as exc:
                runtime_store.log_agent(self.name, "translate_to_zh_error", output, f"ERROR: {exc}")
                return output
        # enforce max chars on direct output path
        if max_chars and step not in {"decide"} and isinstance(output, str) and len(output) > max_chars:
            truncated = output[: max_chars]
            runtime_store.log_agent(self.name, step, full_prompt, truncated, meta={"truncated": True})
            return truncated
        return output

    def update_persona(self, updates: Dict[str, Any]) -> None:
        role = updates.get("role") or self.persona.role
        goal = updates.get("goal") or self.persona.goal
        personality = updates.get("personality") or self.persona.personality
        self.persona.role = role
        self.persona.goal = goal
        self.persona.personality = personality
        self.memory.add_memory(
            f"Persona updated: role={role}, goal={goal}, personality={personality}",
            "reflective",
            {"step": "persona_update"},
        )

    def _mem_meta(self, step: str) -> Dict[str, Any]:
        meta = {"step": step}
        ctx = runtime_store.context if hasattr(runtime_store, "context") else {}
        if isinstance(ctx, dict):
            if ctx.get("event_id"):
                meta["event_id"] = ctx.get("event_id")
            if ctx.get("round") is not None:
                meta["round"] = ctx.get("round")
        return meta

    def _maybe_summarize_memory(self) -> None:
        mem = self.memory
        if not mem or not hasattr(mem, "summarize_and_compress"):
            return
        try:
            mem.summarize_and_compress(self.llm, mem_type="short_term", threshold=self.memory_summary_threshold)
        except Exception:
            return

    def _persona_header(self) -> str:
        return (
            f"你是{self.persona.role}，姓名{self.persona.name}，性别{self.persona.gender}，年龄{self.persona.age}。"
            # f"国籍：{self.persona.nationality}，"
            f"政治身份：{self.persona.political_identity}，阵营：{self.persona.faction}。"
            f"政体立场：{self.persona.regime_stance}；权力偏好：{self.persona.authority_preference}；"
            f"合法性来源：{self.persona.legitimacy_source}；分配倾向：{self.persona.distribution_preference}。"
            f"价值观：{self.persona.values}。信念：{self.persona.belief}。政治观点：{self.persona.polity_view}。"
            f"个人属性：收入{self.persona.income_level}，职业类型{self.persona.occupation_type}，工作强度{self.persona.work_intensity}。"
            f"家庭属性：{'有子女' if self.persona.has_children else '无子女'}，教育焦虑{self.persona.education_anxiety}，家庭负担{self.persona.family_burden}。"
            f"价值权重：{self.persona.value_weights}。"
        )

    def observation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"{self._persona_header()}"
            f"目标：{self.persona.goal}。请观察当前状态并总结关键信号。状态：{state}"
        )
        obs = self._call_llm("observation", prompt)
        self.memory.add_memory(obs, "short_term", self._mem_meta("observation"))
        self.memory.add_memory(obs, "episodic", self._mem_meta("observation"))
        return {"agent": self.name, "observation": obs}

    def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"{self._persona_header()}"
            "基于观察结果提出洞察和假设。"
            f"观察：{observation}"
        )
        thought = self._call_llm("think", prompt)
        self.memory.add_memory(thought, "semantic", self._mem_meta("think"))
        self.memory.add_memory(thought, "short_term", self._mem_meta("think"))
        return {"agent": self.name, "thought": thought}

    def reflect(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"{self._persona_header()}"
            "请反思你的思考，识别风险或偏见，并提出改进建议。"
            f"思考：{thought}"
        )
        reflection = self._call_llm("reflect", prompt)
        self.memory.add_memory(reflection, "reflective", self._mem_meta("reflect"))
        self.memory.add_memory(reflection, "short_term", self._mem_meta("reflect"))
        return {"agent": self.name, "reflection": reflection}

    def act(self, state: Dict[str, Any], reflection: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"{self._persona_header()}"
            "根据当前状态和反思决定下一步行动。"
            f"状态：{state}。反思：{reflection}"
        )
        raw = self._call_llm("act", prompt)

        # try to parse JSON output from agent; if not valid JSON, wrap into structured schema
        parsed_action = None
        if isinstance(raw, str):
            try:
                parsed_action = json.loads(raw)
            except Exception:
                parsed_action = None

        if not parsed_action or not isinstance(parsed_action, dict):
            parsed_action = {
                "tool": parsed_action.get("tool") if isinstance(parsed_action, dict) and parsed_action.get("tool") else "comment",
                "args": {},
                "message": (raw or "").strip(),
                "stance": "",
                "argument": {"claim": "", "evidence": []},
                "impact": {"bias": False, "polarization": False, "pollution": False, "influence": {"changed": False, "from_agents": [], "from_inductions": [], "reason": ""}},
            }

        # enforce max chars on message and log truncation
        try:
            max_chars = int(getattr(self, "max_reply_chars", 0) or 0)
        except Exception:
            max_chars = 0
        if max_chars and isinstance(parsed_action.get("message"), str) and len(parsed_action.get("message")) > max_chars:
            original = parsed_action.get("message")
            truncated = original[:max_chars]
            parsed_action["message"] = truncated
            parsed_action["_truncated"] = True
            runtime_store.log_agent(self.name, "act_truncate", original, truncated)

        # construct parse wrapper expected by simulator
        interaction_wrapper = {"parsed": parsed_action, "raw": raw, "valid": True, "error": None}

        # store structured action in memories
        try:
            mem_entry = json.dumps(interaction_wrapper, ensure_ascii=False)
            self.memory.add_memory(mem_entry, "episodic", self._mem_meta("act"))
            self.memory.add_memory(mem_entry, "long_term", self._mem_meta("act"))
        except Exception:
            try:
                self.memory.add_memory(str(parsed_action), "episodic", self._mem_meta("act"))
            except Exception:
                pass
        self._maybe_summarize_memory()
        return {"interaction": interaction_wrapper}

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        obs = self.observation(state)
        th = self.think(obs)
        rf = self.reflect(th)
        act = self.act(state, rf)
        return {"observation": obs, "thought": th, "reflection": rf, "action": act}
