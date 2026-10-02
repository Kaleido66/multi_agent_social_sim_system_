from typing import Dict, Any
from .base_agent import BaseAgent
from utils.runtime_store import runtime_store
from utils.classical_converter import ClassicalConverter
from utils.json_utils import parse_with_schema

ROLE_BEHAVIOR = {
    "医生": ["help", "use_resource", "comment"],
    "护士": ["help", "use_resource", "comment"],
    "社工": ["help", "mobilize", "comment"],
    "记者": ["publish_news", "comment"],
    "编辑": ["publish_news", "comment"],
    "政策制定者": ["propose_policy", "adjust_resource"],
    "多党倡议者": ["mobilize", "comment", "publish_news"],
    "社区领袖": ["mobilize", "comment"],
    "企业高管": ["adjust_resource", "comment"],
    "工厂工人": ["use_resource", "comment"],
}

INTERACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "message": {"type": "string"},
        "stance": {"type": "string"},
        "argument": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "evidence": {"type": "array"},
            },
            "required": ["claim", "evidence"],
            "additionalProperties": True,
        },
        "impact": {
            "type": "object",
            "properties": {
                "bias": {"type": "boolean"},
                "polarization": {"type": "boolean"},
                "pollution": {"type": "boolean"},
                "resource_delta": {"type": "object"},
                "relation_delta": {"type": "object"},
                "move": {"type": "object"},
                "emotion": {"type": "string"},
                "influence": {
                    "type": "object",
                    "properties": {
                        "changed": {"type": "boolean"},
                        "from_agents": {"type": "array"},
                        "from_inductions": {"type": "array"},
                        "from_messages": {"type": "array"},
                        "reason": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            },
            "required": ["bias", "polarization", "pollution"],
            "additionalProperties": True,
        },
    },
    "required": ["tool", "args", "message", "impact", "argument"],
    "additionalProperties": True,
}

class SocialAgent(BaseAgent):
    def _compact_memory(self, memory_ctx: Dict[str, Any]) -> Dict[str, Any]:
        compact = {}
        for k in ["short_term", "long_term", "social"]:
            items = memory_ctx.get(k, []) or []
            trimmed = []
            for it in items[:2]:
                if isinstance(it, dict):
                    trimmed.append(it.get("content", "")[:120])
                else:
                    trimmed.append(str(it)[:120])
            compact[k] = trimmed
        return compact

    def _compact_state(self, social_state: Dict[str, Any], max_messages: int = 4) -> Dict[str, Any]:
        if not isinstance(social_state, dict):
            return {}
        ev = social_state.get("event", {}) if isinstance(social_state.get("event"), dict) else {}
        msgs = social_state.get("messages", []) or []
        tail = []
        for msg in msgs[-max_messages:]:
            if not isinstance(msg, dict):
                continue
            agent = msg.get("agent")
            interaction = msg.get("interaction", {}) if isinstance(msg.get("interaction"), dict) else {}
            parsed = interaction.get("parsed", {}) if isinstance(interaction.get("parsed"), dict) else {}
            content = parsed.get("message") or ""
            tail.append({"agent": agent, "message": content[:120]})
        return {
            "event_id": ev.get("id"),
            "event_title": ev.get("title"),
            "round": social_state.get("round"),
            "recent_messages": tail,
        }
    def _recent_other_statements(self, social_state: Dict[str, Any], limit: int = 6) -> list:
        messages = social_state.get("messages", []) or []
        others = []
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            agent = msg.get("agent")
            if agent == self.name:
                continue
            interaction = msg.get("interaction", {})
            parsed = interaction.get("parsed") if isinstance(interaction, dict) else {}
            content = ""
            if isinstance(parsed, dict):
                content = parsed.get("message") or ""
            if content:
                others.append({"agent": agent, "message": content})
            if len(others) >= limit:
                break
        return list(reversed(others))

    def _memory_context(self, social_state: Dict[str, Any]) -> Dict[str, Any]:
        query = social_state.get("event_summary", "") or social_state.get("scenario", "")
        event_id = ""
        if isinstance(social_state, dict):
            ev = social_state.get("event", {})
            if isinstance(ev, dict):
                event_id = ev.get("id", "")
        context = {"short_term": [], "long_term": [], "social": []}
        if hasattr(self.memory, "search_memory") and query:
            try:
                context["short_term"] = self.memory.search_memory(query, mem_type="short_term", top_k=3, event_id=event_id or None)
                context["long_term"] = self.memory.search_memory(query, mem_type="long_term", top_k=3, event_id=event_id or None)
                context["social"] = self.memory.search_memory(query, mem_type="social", top_k=3, event_id=event_id or None)
                # fallback to global if no event-scoped memory is found
                if event_id and not any(context.values()):
                    context["short_term"] = self.memory.search_memory(query, mem_type="short_term", top_k=3)
                    context["long_term"] = self.memory.search_memory(query, mem_type="long_term", top_k=3)
                    context["social"] = self.memory.search_memory(query, mem_type="social", top_k=3)
            except Exception:
                pass
        return context

    def _store_social_memory(self, statements: list) -> None:
        for s in statements:
            agent = s.get("agent", "")
            message = s.get("message", "")
            if not message:
                continue
            event_id = ""
            try:
                ctx = runtime_store.context if hasattr(runtime_store, "context") else {}
                if isinstance(ctx, dict):
                    event_id = ctx.get("event_id", "") or ""
            except Exception:
                event_id = ""
            self.memory.add_memory(
                f"他人观点({agent}): {message}",
                "social",
                {"source_agent": agent, "step": "influence", "event_id": event_id},
            )

    def _recent_self_statement(self, social_state: Dict[str, Any]) -> str:
        messages = social_state.get("messages", []) or []
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("agent") != self.name:
                continue
            interaction = msg.get("interaction", {})
            parsed = interaction.get("parsed") if isinstance(interaction, dict) else {}
            if isinstance(parsed, dict) and parsed.get("message"):
                return parsed.get("message")
        return ""

    def _format_inductions_for_prompt(self, items: list) -> str:
        if not items:
            return "无"
        lines = []
        for idx, it in enumerate(items[:5], start=1):
            if isinstance(it, dict):
                iid = str(it.get("id", "")).strip()
                content = str(it.get("content", "")).strip()
                direction = str(it.get("direction", "")).strip()
                score = it.get("score", "")
                seg = f"{idx}. {content}" if content else f"{idx}. (无内容)"
                if iid:
                    seg += f" [id={iid}]"
                if direction:
                    seg += f" [方向={direction}]"
                if score != "":
                    seg += f" [score={score}]"
                lines.append(seg)
            else:
                lines.append(f"{idx}. {str(it).strip()}")
        return "；".join(lines) if lines else "无"

    def _format_other_views_for_prompt(self, views: list) -> str:
        if not views:
            return "无"
        lines = []
        for v in views[:6]:
            if not isinstance(v, dict):
                continue
            agent = str(v.get("agent", "")).strip() or "他人"
            msg = str(v.get("message", "")).strip()
            if msg:
                lines.append(f"{agent}: {msg}")
        return "；".join(lines) if lines else "无"

    def _format_memory_for_prompt(self, memory_compact: Dict[str, Any]) -> str:
        if not isinstance(memory_compact, dict):
            return "无"
        parts = []
        for key in ["short_term", "long_term", "social"]:
            arr = memory_compact.get(key, []) or []
            texts = [str(x).strip() for x in arr if str(x).strip()]
            if texts:
                parts.append(f"{key}: " + " | ".join(texts[:3]))
        return "；".join(parts) if parts else "无"

    def _format_state_for_prompt(self, social_state: Dict[str, Any]) -> str:
        compact = self._compact_state(social_state)
        if not isinstance(compact, dict):
            return "无"
        ridx = compact.get("round")
        title = compact.get("event_title", "")
        msgs = compact.get("recent_messages", []) or []
        msg_texts = []
        for m in msgs[:4]:
            if not isinstance(m, dict):
                continue
            a = str(m.get("agent", "")).strip()
            t = str(m.get("message", "")).strip()
            if a and t:
                msg_texts.append(f"{a}: {t}")
        head = f"round={ridx}, event={title}" if title else f"round={ridx}"
        if msg_texts:
            return head + "；recent=" + " | ".join(msg_texts)
        return head

    def decide(self, social_state: Dict[str, Any], tools: list, agent_goal: str = "", use_event_goals: bool = True, one_shot_action: bool = True, classical_config: Dict[str, Any] = None, classical_to_en: bool = False) -> Dict[str, Any]:
        role_pref = ROLE_BEHAVIOR.get(self.persona.role, [])
        other_views = self._recent_other_statements(social_state)
        if other_views:
            self._store_social_memory(other_views)
        memory_ctx = self._memory_context(social_state)
        memory_compact = self._compact_memory(memory_ctx)
        last_self = self._recent_self_statement(social_state)
        inductions = {}
        if isinstance(social_state, dict):
            inductions = social_state.get("inductions", {}) or {}
        my_inductions = inductions.get(self.name, []) if isinstance(inductions, dict) else []
        goal_line = ""
        if use_event_goals and agent_goal:
            goal_line = f"你在本事件中的目标：{agent_goal}。"
        one_shot_line = ""
        if one_shot_action:
            one_shot_line = "请在内部完成观察-思考-反思，但只输出最终JSON，不要输出中间过程。"
        focus_line = "若你的目标与当前事件主题无关，以事件讨论为准；避免引入其他事件。"
        event_obj = social_state.get("event", {}) if isinstance(social_state, dict) else {}
        event_title = event_obj.get("title", "") if isinstance(event_obj, dict) else ""
        event_content = ""
        if isinstance(social_state, dict):
            event_content = social_state.get("event_content_current", "") or ""
        if not event_content and isinstance(event_obj, dict):
            event_content = event_obj.get("content", "") or ""
        topic_line = f"讨论必须围绕当前事件：{event_title}。请基于事件内容“{event_content}”回扣核心问题，不可偏题。"
        concise_line = "输出message需聚焦主题，最多3点，尽量不重复上一轮自己的表述，观点和论据总字数不超过120字。"
        tone_line = "表达可以带情绪与立场，不必过度理性化。"
        structure_line = "message必须包含“观点:”与“论据:”两部分，且argument.claim与argument.evidence需与message一致。论据至少2条，且彼此支撑。"
        stance_line = ""
        if isinstance(social_state, dict) and social_state.get("initial_stances"):
            stance_line = f"你的初始立场：{social_state['initial_stances'].get(self.name, '')}。"
        prompt_lines = [
            f"你是{self.persona.role}，姓名{self.persona.name}，国籍{self.persona.nationality}，政治身份{self.persona.political_identity}，阵营{self.persona.faction}。",
            # f"你的政体立场：{self.persona.regime_stance}；",
            f"权力偏好：{self.persona.authority_preference}；合法性来源认知：{self.persona.legitimacy_source}；分配倾向：{self.persona.distribution_preference}。",
            f"价值观：{self.persona.values}；信念：{self.persona.belief}。",
            # f"政治观点：{self.persona.polity_view}。",
            f"个人属性：收入{self.persona.income_level}，职业类型{self.persona.occupation_type}，工作强度{self.persona.work_intensity}。",
            f"家庭属性：{'有子女' if self.persona.has_children else '无子女'}，教育焦虑{self.persona.education_anxiety}，家庭负担{self.persona.family_burden}。",
            f"价值权重：{self.persona.value_weights}。",
            "你是自主社会智能体，需要基于自己的记忆与当前社会状态做出行动选择，注意在message必须有明确的观点内容表达，不可只说明使用何种工具。",
            "你必须结合事件情境与自身目标，做出具体行动。你必须围绕主题，不可避而不答，请遵从内心真实的想法。",
            "请识别他人观点是否影响你的立场，并在impact.influence中标注changed与reason。",
            "若参考信息与自身利益强相关（score>=0.6），你更可能调整立场；若发生变化，必须在impact.influence中说明来源与原因，并标记changed=true。",
            "请在impact.influence.from_agents填写影响你的其他角色；在impact.influence.from_inductions填写影响你的参考信息id或摘要。",
            structure_line,
            goal_line,
            one_shot_line,
            focus_line,
            topic_line,
            concise_line,
            tone_line,
        ]
        if other_views:
            prompt_lines.append("你可以回应他人观点（如“回应张三：…”），也可以选择你最关注的角度直接表态。")
        else:
            prompt_lines.append("当前暂无他人观点，可先表达你的初步看法并提出问题。")
        if self.persona.polity_view:
            prompt_lines.append(f"你的政体观点补充：{self.persona.polity_view}")
        if last_self:
            prompt_lines.append(f"上一轮你说过：{last_self}")
        induction_text = self._format_inductions_for_prompt(my_inductions)
        other_views_text = self._format_other_views_for_prompt(other_views)
        memory_text = self._format_memory_for_prompt(memory_compact)
        state_text = self._format_state_for_prompt(social_state)
        prompt_lines.extend([
            f"推荐工具：{role_pref}。可用工具：{tools}。",
            "你必须输出JSON，包含tool/args/message/impact/argument。",
            f"Schema: {INTERACTION_SCHEMA}。",
            f"参考信息（自然语言摘要）：{induction_text}。",
            f"近期他人观点（自然语言摘要）：{other_views_text}。",
            f"记忆检索（自然语言摘要）：{memory_text}。",
            f"状态摘要（自然语言）：{state_text}。",
        ])
        prompt = "\n".join([line for line in prompt_lines if line])
        prompt = prompt.replace("\"event_constraints\":", "事件条件：").replace("'event_constraints':", "事件条件：")
        response = None
        if self.language_mode == "classical" and classical_config and classical_config.get("enabled"):
            converter = ClassicalConverter(self.llm, classical_config)
            classical_prompt = converter.to_classical_prompt(prompt)
            response = self._call_llm("decide_classical", classical_prompt, allow_translate=False)
        else:
            response = self._call_llm("decide", prompt, allow_translate=False)
        parsed = parse_with_schema(response, INTERACTION_SCHEMA)

        if parsed.get("valid") and isinstance(parsed.get("parsed"), dict):
            pobj = parsed.get("parsed", {})
            if "argument" not in pobj or not isinstance(pobj.get("argument"), dict):
                msg = pobj.get("message", "")
                pobj["argument"] = {"claim": msg, "evidence": [msg] if msg else []}
                parsed["parsed"] = pobj
            # ensure evidence has at least 2 points
            try:
                arg = pobj.get("argument", {})
                ev = arg.get("evidence", [])
                if not isinstance(ev, list):
                    ev = [str(ev)]
                if len(ev) < 2:
                    msg = pobj.get("message", "")
                    parts = [p.strip() for p in msg.replace("；", "。").split("。") if p.strip()]
                    if len(parts) >= 2:
                        ev = parts[:2]
                    elif len(parts) == 1:
                        ev = [parts[0], parts[0]]
                    else:
                        ev = [msg, msg] if msg else ["", ""]
                    arg["evidence"] = ev
                    pobj["argument"] = arg
                    parsed["parsed"] = pobj
            except Exception:
                pass
            # ensure message contains explicit claim/evidence structure
            try:
                msg = pobj.get("message", "")
                arg = pobj.get("argument", {}) or {}
                claim = arg.get("claim", "") or msg
                ev = arg.get("evidence", []) or []
                if not isinstance(ev, list):
                    ev = [str(ev)]
                ev_text = "；".join([str(e) for e in ev if str(e).strip()])
                if "观点" not in msg or "论据" not in msg:
                    pobj["message"] = f"观点: {claim} 论据: {ev_text}"
                    parsed["parsed"] = pobj
            except Exception:
                pass

        if not parsed.get("valid") or not parsed.get("parsed") or not parsed["parsed"].get("message"):
            fallback_prompt = (
                f"你是{self.persona.role}，姓名{self.persona.name}。国籍{self.persona.nationality}，政治身份{self.persona.political_identity}，阵营{self.persona.faction}。"
                # f"你的政体立场：{self.persona.regime_stance}；"
                f"权力偏好：{self.persona.authority_preference}；合法性来源认知：{self.persona.legitimacy_source}；分配倾向：{self.persona.distribution_preference}。"
                f"价值观：{self.persona.values}；信念：{self.persona.belief}。"
                # f"政治观点：{self.persona.polity_view}。"
                f"事件摘要：{(social_state.get('event_content_current') or social_state.get('event_summary', ''))}请基于事件摘要输出一条中文消息，包含“观点:”与“论据:”。"
            )
            msg = self._call_llm("decide_fallback", fallback_prompt)
            claim = msg
            evidence = []
            if "观点" in msg and "论据" in msg:
                parts = msg.split("论据", 1)
                claim = parts[0].replace("观点", "").replace("：", "").strip()
                ev = parts[1].replace("：", "").strip()
                if ev:
                    evidence = [e.strip() for e in ev.split("；") if e.strip()]
            parsed = {
                "parsed": {
                    "tool": role_pref[0] if role_pref else "comment",
                    "args": {"content": msg},
                    "message": msg,
                    "stance": "未明确",
                    "argument": {"claim": claim, "evidence": evidence or [msg]},
                    "impact": {"bias": False, "polarization": False, "pollution": False},
                },
                "valid": True,
                "error": "",
                "raw": response,
            }

        event_id = ""
        try:
            ctx = runtime_store.context if hasattr(runtime_store, "context") else {}
            if isinstance(ctx, dict):
                event_id = ctx.get("event_id", "") or ""
        except Exception:
            event_id = ""
        self.memory.add_memory(response, "episodic", {"step": "decide", "parsed": parsed, "event_id": event_id})
        parsed_obj = parsed.get("parsed") if isinstance(parsed, dict) else {}
        impact = parsed_obj.get("impact", {}) if isinstance(parsed_obj, dict) else {}
        if isinstance(impact, dict) and impact.get("emotion"):
            self.memory.add_memory(
                f"情绪: {impact.get('emotion')}",
                "emotional",
                {"step": "decide", "message": parsed_obj.get("message", "")},
            )
        # translate classical response to zh/en for analysis
        if self.language_mode == "classical" and classical_config and classical_config.get("enabled") and isinstance(parsed_obj, dict):
            converter = ClassicalConverter(self.llm, classical_config)
            msg = parsed_obj.get("message", "")
            stance = parsed_obj.get("stance", "")
            if msg:
                parsed_obj["message_classical"] = msg
                parsed_obj["message_zh"] = converter.classical_to_zh(msg)
                if classical_to_en:
                    parsed_obj["message_en"] = converter.classical_to_en(msg)
            if stance:
                parsed_obj["stance_classical"] = stance
                parsed_obj["stance_zh"] = converter.classical_to_zh(stance)
                if classical_to_en:
                    parsed_obj["stance_en"] = converter.classical_to_en(stance)
            parsed["parsed"] = parsed_obj
        return parsed

    def act(self, state: Dict[str, Any], reflection: Dict[str, Any]) -> Dict[str, Any]:
        social_state = state.get("social_state", {})
        tools = state.get("tools", [])
        agent_goal = state.get("agent_goal", "")
        use_event_goals = bool(state.get("use_event_goals", True))
        one_shot_action = bool(state.get("one_shot_action", True))
        classical_config = state.get("classical_config", {}) or {}
        classical_to_en = bool(state.get("classical_to_en", False))
        # add internal observation/thought/reflection for autonomous behavior
        if not one_shot_action:
            obs = self.observation({"social_state": social_state, "round": state.get("round")})
            th = self.think(obs)
            rf = self.reflect(th)
        parsed = self.decide(
            social_state,
            tools,
            agent_goal=agent_goal,
            use_event_goals=use_event_goals,
            one_shot_action=one_shot_action,
            classical_config=classical_config,
            classical_to_en=classical_to_en,
        )
        # update persona belief when stance changes
        impact = {}
        if isinstance(parsed, dict):
            impact = parsed.get("parsed", {}).get("impact", {}) or {}
        influence = impact.get("influence", {}) if isinstance(impact, dict) else {}
        if isinstance(influence, dict) and influence.get("changed"):
            stance = parsed.get("parsed", {}).get("stance") if isinstance(parsed, dict) else ""
            if stance:
                self.persona.belief = f"{self.persona.belief} | {stance}".strip(" |")
            self.memory.add_memory(
                f"立场变化: {stance or influence.get('reason','')}",
                "long_term",
                {"step": "stance_change", "reason": influence.get("reason", "")},
            )
        return {"agent": self.name, "interaction": parsed}

class MediaAgent(SocialAgent):
    pass

class CitizenAgent(SocialAgent):
    pass

class PolicyAgent(SocialAgent):
    pass
