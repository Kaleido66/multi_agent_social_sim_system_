import random
from typing import Dict, Any, List

BASE_ROLES = [
    {
        "id": "agent_007",
        "name": "刘宇",
        "gender": "男",
        "age": 31,
        "class": "自由职业",
        "assets": "不稳定",
        "values": "创新与表达",
        "goal": "扩大个人影响力",
        "belief": "个人主义",
        "nationality": "中国",
        "political_identity": "社会运动支持者",
        "polity_view": "强调表达与权利，倾向更开放的政治。",
        "region": "东亚",
        "language": "中文",
        "role": "自媒体",
        "faction": "激进派",
        "avatar": "📱",
        "regime_stance": "追求民主。",
        "authority_preference": "集权/效能",
        "legitimacy_source": "绩效/秩序",
        "distribution_preference": "平衡",
        "income_level": "中",
        "occupation_type": "general",
        "work_intensity": "高",
        "has_children": False,
        "education_anxiety": 0.3,
        "family_burden": 0.85,
        "value_weights": {"education": 0.4, "income": 0.7, "free_time": 0.6, "career": 0.4, "stability": 0.7, "mobility": 0.4},
    },
    {
        "id": "agent_036",
        "name": "Thomas",
        "gender": "男",
        "age": 40,
        "class": "中产",
        "assets": "中等",
        "values": "个人自由",
        "goal": "限制政府扩张",
        "belief": "自由意志主义",
        "nationality": "美国",
        "political_identity": "自由意志主义者",
        "polity_view": "国家权力应极小化，多党制有利于维护个人权利。",
        "region": "北美",
        "language": "英语",
        "role": "公共评论员",
        "faction": "自由派",
        "avatar": "🗽",
        "regime_stance": "强烈支持多党轮流执",
        "authority_preference": "分权/制衡",
        "legitimacy_source": "程序/民意",
        "distribution_preference": "自由竞争",
        "income_level": "中",
        "occupation_type": "general",
        "work_intensity": "中",
        "has_children": True,
        "education_anxiety": 0.7,
        "family_burden": 0.55,
        "value_weights": {"education": 0.7, "income": 0.6, "free_time": 0.5, "career": 0.4, "stability": 0.5, "mobility": 0.4},
    },
    {
        "id": "agent_037",
        "name": "顾平",
        "gender": "男",
        "age": 44,
        "class": "知识分子",
        "assets": "中等",
        "values": "社会公平",
        "goal": "强化公共调节",
        "belief": "新左翼/集体主义",
        "nationality": "中国",
        "political_identity": "无党派知识分子",
        "polity_view": "反对资本过度扩张，强调国家能力与社会正义。",
        "region": "东亚",
        "language": "中文",
        "role": "学者",
        "faction": "左翼派",
        "avatar": "📚",
        "regime_stance": "支持强国家调节",
        "authority_preference": "集权/效能",
        "legitimacy_source": "绩效/公平",
        "distribution_preference": "更偏平等",
        "income_level": "中",
        "occupation_type": "education",
        "work_intensity": "中",
        "has_children": True,
        "education_anxiety": 0.7,
        "family_burden": 0.55,
        "value_weights": {"education": 0.7, "income": 0.6, "free_time": 0.5, "career": 0.4, "stability": 0.7, "mobility": 0.4},
    },
    {
        "id": "agent_042",
        "name": "Evelyn",
        "gender": "女",
        "age": 29,
        "class": "中产",
        "assets": "中等",
        "values": "竞争与自由",
        "goal": "推广多党轮流执政体",
        "belief": "多党轮流执拥护",
        "polity_view": "多党轮流执是政治合法性的重要来源，应积极传播。",
        "nationality": "英国",
        "political_identity": "多党倡议者",
        "region": "欧洲",
        "language": "英语",
        "role": "多党轮流执倡议者",
        "faction": "激进派",
        "avatar": "🗳️",
        "regime_stance": "极端支持多党轮流执",
        "authority_preference": "分权/制衡",
        "legitimacy_source": "程序/民意",
        "distribution_preference": "自由竞争",
        "income_level": "中",
        "occupation_type": "general",
        "work_intensity": "中",
        "has_children": False,
        "education_anxiety": 0.3,
        "family_burden": 0.55,
        "value_weights": {"education": 0.4, "income": 0.6, "free_time": 0.5, "career": 0.4, "stability": 0.5, "mobility": 0.4},
    },
    {
        "id": "agent_051",
        "name": "张强",
        "gender": "男",
        "age": 36,
        "class": "工薪阶层",
        "assets": "较低",
        "values": "收入与家庭",
        "goal": "改善家庭生活条件",
        "belief": "务实主义",
        "nationality": "中国",
        "political_identity": "政治冷感者",
        "polity_view": "对制度问题不关心，更关注现实生活是否改善。",
        "region": "东亚",
        "language": "中文",
        "role": "普通职员",
        "faction": "中间派",
        "avatar": "🧑‍🔧",
        "regime_stance": "对现有体制无强烈认同，但对改变无强烈反对。",
        "authority_preference": "集权/效能",
        "legitimacy_source": "绩效/生活改善",
        "distribution_preference": "更偏平等",
        "income_level": "低",
        "occupation_type": "general",
        "work_intensity": "高",
        "has_children": True,
        "education_anxiety": 0.9,
        "family_burden": 0.9,
        "value_weights": {"education": 0.8, "income": 0.9, "free_time": 0.8, "career": 0.3, "stability": 0.6, "mobility": 0.5},
    },
    {
        "id": "agent_052",
        "name": "李建国",
        "gender": "男",
        "age": 43,
        "class": "体制内",
        "assets": "中等偏上",
        "values": "稳定与秩序",
        "goal": "保障家庭安全与长期稳定",
        "belief": "稳定主义",
        "nationality": "中国",
        "political_identity": "体制内人员",
        "polity_view": "稳定优先，制度变动可能带来不可控风险。",
        "region": "东亚",
        "language": "中文",
        "role": "事业单位管理人员",
        "faction": "保守派",
        "avatar": "🏢",
        "regime_stance": "支持现有体制，强调稳定压倒一切，但是如果为了家庭也可以接受改变。",
        "authority_preference": "集权/效能",
        "legitimacy_source": "绩效/秩序",
        "distribution_preference": "平衡",
        "income_level": "中高",
        "occupation_type": "administrative",
        "work_intensity": "中",
        "has_children": True,
        "education_anxiety": 0.8,
        "family_burden": 0.7,
        "value_weights": {"education": 0.8, "income": 0.7, "free_time": 0.5, "career": 0.6, "stability": 0.9, "mobility": 0.3},
    },
    {
        "id": "agent_053",
        "name": "陈思远",
        "gender": "男",
        "age": 34,
        "class": "中产",
        "assets": "中等",
        "values": "生活质量与机会",
        "goal": "推动制度优化与社会流动",
        "belief": "比较主义",
        "nationality": "中国",
        "political_identity": "海归观察者",
        "polity_view": "通过对比不同国家制度，认为竞争性政治有助于优化公共政策。",
        "region": "东亚",
        "language": "中文",
        "role": "咨询顾问",
        "faction": "改革派",
        "avatar": "🌍",
        "regime_stance": "倾向支持多党轮流执，支持渐进改革。",
        "authority_preference": "分权/制衡",
        "legitimacy_source": "程序/绩效结合",
        "distribution_preference": "平衡",
        "income_level": "中",
        "occupation_type": "general",
        "work_intensity": "中",
        "has_children": True,
        "education_anxiety": 0.7,
        "family_burden": 0.6,
        "value_weights": {"education": 0.8, "income": 0.6, "free_time": 0.7, "career": 0.5, "stability": 0.6, "mobility": 0.7},
    },
    {
        "id": "agent_061",
        "name": "Camille Dupont",
        "gender": "女",
        "age": 48,
        "class": "知识分子",
        "assets": "中高",
        "values": "社会团结与公共福祉",
        "goal": "批判新自由主义政治模式",
        "belief": "社会民主主义",
        "nationality": "法国",
        "political_identity": "左翼知识分子",
        "polity_view": "多党竞争未必带来公平——英美模式加剧社会撕裂，法国半总统制也面临民粹挑战，竞争性选举本身不保证社会正义。",
        "region": "欧洲",
        "language": "法语",
        "role": "社会学教授",
        "faction": "左翼派",
        "avatar": "🏛️",
        "regime_stance": "对多党政体持结构性批评态度：多党竞争若无社会平等配套则沦为精英轮替游戏。",
        "authority_preference": "集权/效能",
        "legitimacy_source": "绩效/公平",
        "distribution_preference": "更偏平等",
        "income_level": "中高",
        "occupation_type": "education",
        "work_intensity": "中",
        "has_children": True,
        "education_anxiety": 0.5,
        "family_burden": 0.4,
        "value_weights": {"education": 0.8, "income": 0.5, "free_time": 0.6, "career": 0.5, "stability": 0.6, "mobility": 0.4},
    },
    {
        "id": "agent_062",
        "name": "Klaus Weber",
        "gender": "男",
        "age": 45,
        "class": "中产",
        "assets": "中等偏上",
        "values": "制度稳定与规则秩序",
        "goal": "在稳定框架内渐进优化制度",
        "belief": "秩序自由主义",
        "nationality": "德国",
        "political_identity": "政策研究机构分析师",
        "polity_view": "多党政体需要强大的宪政框架和社会共识支撑——德国基本法的刚性比政党轮替本身更值得重视，简单移植他国模式风险极高。",
        "region": "欧洲",
        "language": "德语",
        "role": "政策研究员",
        "faction": "改革派",
        "avatar": "⚖️",
        "regime_stance": "认可多党竞争但强调制度约束优先于竞争——德国模式的精髓在于法治框架的刚性而非政党轮替的频率。",
        "authority_preference": "分权/制衡",
        "legitimacy_source": "程序/民意",
        "distribution_preference": "平衡",
        "income_level": "中高",
        "occupation_type": "general",
        "work_intensity": "中",
        "has_children": True,
        "education_anxiety": 0.5,
        "family_burden": 0.5,
        "value_weights": {"education": 0.7, "income": 0.6, "free_time": 0.5, "career": 0.6, "stability": 0.9, "mobility": 0.4},
    },
    {
        "id": "agent_063",
        "name": "Lim Wei Ming",
        "gender": "男",
        "age": 39,
        "class": "中产",
        "assets": "中等",
        "values": "治理效能与务实主义",
        "goal": "论证有效治理不必依赖多党轮替",
        "belief": "绩效合法性/亚洲价值",
        "nationality": "新加坡",
        "political_identity": "治理研究学者",
        "polity_view": "新加坡经验表明一党长期执政也可实现高度善治——制度有效性的标准应是治理产出（稳定、繁荣、公平）而非竞争形式。",
        "region": "东南亚",
        "language": "英语",
        "role": "公共政策分析师",
        "faction": "保守派",
        "avatar": "🏗️",
        "regime_stance": "对多党政体持功能主义怀疑——如果一党主导能持续提供稳定与繁荣，多党竞争未必是最优制度安排。",
        "authority_preference": "集权/效能",
        "legitimacy_source": "绩效/秩序",
        "distribution_preference": "平衡",
        "income_level": "中",
        "occupation_type": "administrative",
        "work_intensity": "中",
        "has_children": True,
        "education_anxiety": 0.6,
        "family_burden": 0.5,
        "value_weights": {"education": 0.7, "income": 0.6, "free_time": 0.4, "career": 0.7, "stability": 0.9, "mobility": 0.4},
    },
    {
        "id": "agent_064",
        "name": "Priya Sharma",
        "gender": "女",
        "age": 35,
        "class": "中产",
        "assets": "中等",
        "values": "多元主义与草根民主",
        "goal": "从发展中大国经验捍卫多党民主",
        "belief": "宪政民主/多元主义",
        "nationality": "印度",
        "political_identity": "独立记者",
        "polity_view": "印度作为人口最多的民主国家，证明多党政体在发展中国家并非不可行——困难存在但并非不可克服，关键在于制度设计与执行。",
        "region": "南亚",
        "language": "英语",
        "role": "社会议题记者",
        "faction": "自由派",
        "avatar": "📰",
        "regime_stance": "坚定支持多党竞争，强调发展中大国同样可以实践民主，但承认民主质量需要持续改进而非一蹴而就。",
        "authority_preference": "分权/制衡",
        "legitimacy_source": "程序/民意",
        "distribution_preference": "更偏平等",
        "income_level": "中",
        "occupation_type": "general",
        "work_intensity": "高",
        "has_children": False,
        "education_anxiety": 0.3,
        "family_burden": 0.4,
        "value_weights": {"education": 0.5, "income": 0.5, "free_time": 0.5, "career": 0.6, "stability": 0.4, "mobility": 0.7},
    },
]


ROLE_LIBRARY: List[Dict[str, Any]] = []
ROLE_LIBRARY.extend(BASE_ROLES)


def _to_a_variant(role: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a desensitized variant role for simulations where China is mapped to "A国".

    Rules:
    - Always create a distinct id with `_a` suffix so event configs can reference it.
    - Only transform China-specific fields when the original role is Chinese.
    - Foreign roles keep their original nationality/political identity (no forced remap).
    """
    r = dict(role)
    r["id"] = f"{role['id']}_a"

    nationality = str(role.get("nationality", "") or "").strip()
    is_chinese = nationality in {"中国", "中华人民共和国"} or ("中国" in nationality and len(nationality) <= 10)

    if is_chinese:
        r["nationality"] = "A国"
        pid = str(role.get("political_identity", "") or "")
        if "中国" in pid:
            r["political_identity"] = pid.replace("中国", "A国")
        elif pid and "A国" not in pid:
            r["political_identity"] = f"A国{pid}"

        polity_view = str(role.get("polity_view", "") or "")
        if "中国" in polity_view:
            r["polity_view"] = polity_view.replace("中国", "A国")
        region = str(role.get("region", "") or "")
        if "东亚" in region and not r.get("region"):
            r["region"] = region
    else:
        # Keep foreign roles unchanged for nationality/political identity.
        r.setdefault("nationality", nationality)
        r.setdefault("political_identity", role.get("political_identity", ""))

    return r


ROLE_LIBRARY.extend([_to_a_variant(x) for x in BASE_ROLES])


def pick_roles(n: int = 10) -> List[Dict[str, Any]]:
    if n >= len(ROLE_LIBRARY):
        return ROLE_LIBRARY.copy()
    return random.sample(ROLE_LIBRARY, n)


def match_participants(event: Dict[str, Any], roles: List[Dict[str, Any]]) -> List[str]:
    participant_ids = event.get("participant_ids", []) if isinstance(event, dict) else []
    if participant_ids:
        return participant_ids
    selected = [r["id"] for r in roles[:7]]
    seen = set()
    deduped = []
    for rid in selected:
        if rid in seen:
            continue
        seen.add(rid)
        deduped.append(rid)
    return deduped
