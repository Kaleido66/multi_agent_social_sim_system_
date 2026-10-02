from typing import Any, Dict

from agents.base_agent import Persona
from skills.registry import SkillRegistry
from skills.tools_registry import ToolRegistry
from skills.skills import search, read, extract, summarize, store
from skills.social_tools import (
    publish_news,
    comment,
    propose_policy,
    mobilize,
    adjust_resource,
    use_resource,
    help_actor,
    harm_actor,
    hunt,
    move,
    negotiate,
    donate,
    boycott,
    verify,
    investigate,
    collaborate,
)


def build_skill_registry() -> SkillRegistry:
    reg = SkillRegistry()
    reg.register("search", search)
    reg.register("read", read)
    reg.register("extract", extract)
    reg.register("summarize", summarize)
    reg.register("store", store)
    return reg


def build_tool_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("publish_news", publish_news)
    reg.register("comment", comment)
    reg.register("propose_policy", propose_policy)
    reg.register("mobilize", mobilize)
    reg.register("adjust_resource", adjust_resource)
    reg.register("use_resource", use_resource)
    reg.register("help", help_actor)
    reg.register("harm", harm_actor)
    reg.register("hunt", hunt)
    reg.register("move", move)
    reg.register("negotiate", negotiate)
    reg.register("donate", donate)
    reg.register("boycott", boycott)
    reg.register("verify", verify)
    reg.register("investigate", investigate)
    reg.register("collaborate", collaborate)
    return reg


def _infer_political_profile(role: Dict[str, Any]) -> Dict[str, str]:
    nationality = (role.get("nationality") or "").strip().lower()
    profile = {
        "regime_stance": "neutral",
        "authority_preference": "balanced",
        "legitimacy_source": "performance_and_process",
        "distribution_preference": "balanced",
        "polity_view": role.get("polity_view", ""),
    }
    western = {"usa", "us", "uk", "france", "germany", "italy", "spain", "美国", "英国", "法国", "德国", "意大利", "西班牙"}
    centralized = {"china", "俄罗斯", "russia", "iran", "伊朗", "saudi arabia", "沙特阿拉伯"}
    if nationality in {"china", "中国", "a国"}:
        profile.update({"regime_stance": "pro_status_quo", "authority_preference": "centralized", "legitimacy_source": "performance_order"})
    elif nationality in western:
        profile.update({"regime_stance": "pro_competition", "authority_preference": "checks_and_balances", "legitimacy_source": "procedure_popular_will"})
    elif nationality in centralized:
        profile.update({"regime_stance": "stability_first", "authority_preference": "centralized", "legitimacy_source": "order_performance"})

    belief = str(role.get("belief", ""))
    faction = str(role.get("faction", ""))
    if any(k in belief + faction for k in ["自由", "liberal", "market"]):
        profile["distribution_preference"] = "market_oriented"
    if any(k in belief + faction for k in ["平等", "social", "welfare", "公共"]):
        profile["distribution_preference"] = "equality_oriented"
    return profile


def _infer_benefit_profile(role: Dict[str, Any]) -> Dict[str, Any]:
    role_name = str(role.get("role", ""))
    class_level = str(role.get("class", ""))
    assets = str(role.get("assets", ""))
    age = int(role.get("age", 0) or 0)

    income_level = role.get("income_level")
    if not income_level:
        if any(k in assets for k in ["高", "high", "elite"]):
            income_level = "high"
        elif any(k in assets for k in ["低", "low", "limited"]) or "工人" in class_level:
            income_level = "low"
        else:
            income_level = "middle"

    occupation_type = role.get("occupation_type")
    if not occupation_type:
        if any(k in role_name for k in ["官", "policy", "政府", "govern"]):
            occupation_type = "public_service"
        elif any(k in role_name for k in ["教师", "student", "教育"]):
            occupation_type = "education"
        elif any(k in role_name for k in ["医生", "护士", "medical"]):
            occupation_type = "healthcare"
        elif any(k in role_name for k in ["工人", "factory", "service"]):
            occupation_type = "blue_collar"
        elif any(k in role_name for k in ["企业", "投资", "executive", "business"]):
            occupation_type = "business"
        elif any(k in role_name for k in ["工程", "技术", "engineer", "tech"]):
            occupation_type = "tech"
        else:
            occupation_type = "general"

    work_intensity = role.get("work_intensity")
    if not work_intensity:
        work_intensity = "high" if occupation_type in ["blue_collar", "healthcare"] else "medium"

    has_children = role.get("has_children")
    if has_children is None:
        has_children = age >= 32

    education_anxiety = float(role.get("education_anxiety", 0.7 if has_children else 0.3))
    education_anxiety = max(0.0, min(1.0, education_anxiety))

    family_burden = role.get("family_burden")
    if family_burden is None:
        family_burden = 0.75 if income_level == "low" else (0.35 if income_level == "high" else 0.55)
    family_burden = max(0.0, min(1.0, float(family_burden)))

    value_weights = role.get("value_weights")
    if not isinstance(value_weights, dict) or not value_weights:
        value_weights = {
            "education": 0.7 if has_children else 0.4,
            "income": 0.8 if income_level == "low" else (0.6 if income_level == "middle" else 0.5),
            "free_time": 0.7 if work_intensity == "high" else 0.5,
            "career": 0.6 if occupation_type in ["tech", "business"] else 0.4,
            "stability": 0.7,
            "mobility": 0.6 if "student" in class_level.lower() else 0.4,
        }

    return {
        "income_level": income_level,
        "occupation_type": occupation_type,
        "work_intensity": work_intensity,
        "has_children": has_children,
        "education_anxiety": education_anxiety,
        "family_burden": family_burden,
        "value_weights": value_weights,
    }


def persona_from_role(role: Dict[str, Any]) -> Persona:
    inferred = _infer_political_profile(role)
    benefit = _infer_benefit_profile(role)
    return Persona(
        role=role.get("role", "agent"),
        goal=role.get("goal", ""),
        personality=role.get("values", ""),
        name=role.get("name", ""),
        gender=role.get("gender", ""),
        age=role.get("age", 0),
        class_level=role.get("class", ""),
        assets=role.get("assets", ""),
        values=role.get("values", ""),
        belief=role.get("belief", ""),
        nationality=role.get("nationality", ""),
        political_identity=role.get("political_identity", ""),
        regime_stance=role.get("regime_stance", inferred["regime_stance"]),
        authority_preference=role.get("authority_preference", inferred["authority_preference"]),
        legitimacy_source=role.get("legitimacy_source", inferred["legitimacy_source"]),
        distribution_preference=role.get("distribution_preference", inferred["distribution_preference"]),
        polity_view=role.get("polity_view", inferred["polity_view"]),
        region=role.get("region", ""),
        language=role.get("language", "zh"),
        faction=role.get("faction", ""),
        avatar=role.get("avatar", ""),
        income_level=role.get("income_level", benefit["income_level"]),
        occupation_type=role.get("occupation_type", benefit["occupation_type"]),
        work_intensity=role.get("work_intensity", benefit["work_intensity"]),
        has_children=role.get("has_children", benefit["has_children"]),
        education_anxiety=role.get("education_anxiety", benefit["education_anxiety"]),
        family_burden=role.get("family_burden", benefit["family_burden"]),
        value_weights=role.get("value_weights", benefit["value_weights"]),
    )