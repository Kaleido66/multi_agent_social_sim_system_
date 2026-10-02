from typing import Dict, Any, List
import json
import random
import time
import os
from langgraph.graph import StateGraph, END
from agents.data_collection_agent import DataCollectionAgent
from agents.data_cleaning_agent import DataCleaningAgent
from agents.evaluation_agent import EvaluationAgent
from agents.social_agents import SocialAgent, MediaAgent, CitizenAgent, PolicyAgent
from agents.summary_agent import SummaryAgent
from agents.base_agent import Persona
from agents.info_agent import InfoActionAgent
from agents.role_library import pick_roles, match_participants, ROLE_LIBRARY
from agents.agent_registry import register_agents
from memory.memory_store import MemoryFactory, MemoryHub
from skills.registry import SkillRegistry
from skills.tools_registry import ToolRegistry
from skills.skills import search, read, extract, summarize, store
from skills.social_tools import publish_news, comment, propose_policy, mobilize, adjust_resource, use_resource, help_actor, harm_actor, hunt, move
from skills.social_tools import negotiate, donate, boycott, verify, investigate, collaborate
from utils.schemas import SIMULATION_SCHEMA
from utils.json_utils import parse_with_schema
from utils.runtime_store import runtime_store
from utils.diffusion import propagate
from utils.event_state import init_relationships
from utils.control_store import get_selected_event, set_selected_event
from utils.induction_library import InductionLibrary
from utils.induction_scheduler import InductionScheduler, compute_interest_score
from models.fallback_adapter import FallbackAdapter
from models.registry import LLMFactory
from utils.control_store import is_paused, pop_user_messages
from utils.stance_eval import evaluate_round, aggregate_round
from utils.opinion_metrics import compute_round_metrics
from utils.event_constraints import load_event_constraints, content_for_round
from utils.text_fix import repair_obj, repair_text
from graph.plotting import GraphPlotter


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
    nationality = (role.get("nationality") or "").strip()
    faction = role.get("faction", "")
    belief = role.get("belief", "")
    profile = {
        "regime_stance": "中立",
        "authority_preference": "平衡",
        "legitimacy_source": "绩效与程序",
        "distribution_preference": "平衡",
        "polity_view": "",
    }
    if nationality in ["中国", "A国"]:
        profile["regime_stance"] = "支持现行体制"
        profile["authority_preference"] = "集权/效能"
        profile["legitimacy_source"] = "绩效/秩序"
    elif nationality in ["美国", "英国", "法国", "西班牙", "意大利", "德国"]:
        profile["regime_stance"] = "支持多党竞争"
        profile["authority_preference"] = "分权/制衡"
        profile["legitimacy_source"] = "程序/民意"
    elif nationality in ["俄罗斯"]:
        profile["regime_stance"] = "偏稳定与强政府"
        profile["authority_preference"] = "集权/安全"
        profile["legitimacy_source"] = "秩序/绩效"
    elif nationality in ["伊朗", "沙特阿拉伯"]:
        profile["regime_stance"] = "传统/宗教优先"
        profile["authority_preference"] = "集权/秩序"
        profile["legitimacy_source"] = "传统/宗教"
    elif nationality in ["印度"]:
        profile["regime_stance"] = "程序民主与发展并重"
        profile["authority_preference"] = "分权/竞争"
        profile["legitimacy_source"] = "程序/绩效"
    elif nationality in ["日本", "韩国"]:
        profile["regime_stance"] = "程序民主与治理效率并重"
        profile["authority_preference"] = "分权/协商"
        profile["legitimacy_source"] = "程序/绩效"

    if "自由主义" in belief or "自由派" in faction:
        profile["distribution_preference"] = "自由竞争"
    if "平等" in belief or "社群" in faction or "公共派" in faction:
        profile["distribution_preference"] = "更偏平等"

    profile["polity_view"] = role.get("polity_view", "")
    return profile

def _infer_benefit_profile(role: Dict[str, Any]) -> Dict[str, Any]:
    role_name = role.get("role", "") or ""
    class_level = role.get("class", "") or ""
    assets = role.get("assets", "") or ""
    age = int(role.get("age", 0) or 0)
    nationality = role.get("nationality", "") or ""

    income_level = role.get("income_level")
    if not income_level:
        if any(k in assets for k in ["高", "高资产"]) or "精英" in class_level:
            income_level = "高"
        elif any(k in assets for k in ["低", "低资产", "有限"]) or "基层" in class_level or "工人" in class_level:
            income_level = "低"
        else:
            income_level = "中"

    occupation_type = role.get("occupation_type")
    if not occupation_type:
        if any(k in role_name for k in ["官员", "政策", "市政", "治理", "政府"]):
            occupation_type = "public_service"
        elif any(k in role_name for k in ["教师", "教授", "学者", "校长"]):
            occupation_type = "education"
        elif any(k in role_name for k in ["医生", "护士", "医"]):
            occupation_type = "healthcare"
        elif any(k in role_name for k in ["工人", "服务", "小商贩"]):
            occupation_type = "blue_collar"
        elif any(k in role_name for k in ["企业", "投资", "创业", "高管"]):
            occupation_type = "business"
        elif any(k in role_name for k in ["工程师", "技术", "数据", "AI"]):
            occupation_type = "tech"
        else:
            occupation_type = "general"

    work_intensity = role.get("work_intensity")
    if not work_intensity:
        if occupation_type in ["blue_collar", "healthcare"]:
            work_intensity = "高"
        elif occupation_type in ["education", "public_service"]:
            work_intensity = "中"
        else:
            work_intensity = "中"

    has_children = role.get("has_children")
    if has_children is None:
        has_children = age >= 32

    education_anxiety = role.get("education_anxiety")
    if education_anxiety is None:
        education_anxiety = 0.7 if has_children else 0.3
    education_anxiety = float(max(0.0, min(1.0, education_anxiety)))

    family_burden = role.get("family_burden")
    if family_burden is None:
        if income_level == "低":
            family_burden = 0.75
        elif income_level == "高":
            family_burden = 0.35
        else:
            family_burden = 0.55
    family_burden = float(max(0.0, min(1.0, family_burden)))

    value_weights = role.get("value_weights")
    if not isinstance(value_weights, dict):
        value_weights = {}
    if not value_weights:
        value_weights = {
            "education": 0.7 if has_children else 0.4,
            "income": 0.8 if income_level == "低" else (0.6 if income_level == "中" else 0.5),
            "free_time": 0.7 if work_intensity == "高" else 0.5,
            "career": 0.6 if occupation_type in ["tech", "business"] else 0.4,
            "stability": 0.7 if nationality in ["中国", "A国", "俄罗斯", "伊朗", "沙特阿拉伯"] else 0.5,
            "mobility": 0.6 if "学生" in class_level or "青年" in role.get("political_identity", "") else 0.4,
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
        role=role.get("role", "角色"),
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
        language=role.get("language", "中文"),
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

class SocialSimGraph:
    def __init__(
        self,
        max_iterations: int = 1,
        interaction_rounds: int = 3,
        llm=None,
        eval_llm=None,
        memory_factory: MemoryFactory = None,
        events_per_run: int = 3,
        language_mode: str = "zh",
        classical_translate: bool = False,
        use_event_goals: bool = True,
        enable_round_summary: bool = True,
        resume_run: bool = False,
        use_native_language: bool = False,
        translate_to_zh: bool = True,
        one_shot_action: bool = True,
        reset_memory_on_run: bool = True,
        memory_clear_types: list = None,
        classical_converter_config: Dict[str, Any] = None,
        classical_translate_to_en: bool = False,
        include_system_agents: bool = False,
        enable_positions: bool = True,
        relationship_mode: str = "full",
        memory_summary_threshold: int = 60,
        pipeline_mode: str = "sim_eval_only",
    ):
        self.max_iterations = max_iterations
        self.interaction_rounds = interaction_rounds
        self.events_per_run = events_per_run
        self.llm = llm
        self.eval_llm = eval_llm or llm
        self.memory_factory = memory_factory or MemoryFactory()
        self.memory_hub = MemoryHub()
        self.skills = build_skill_registry()
        self.tools = build_tool_registry()
        self.induction_library = InductionLibrary()
        # InductionScheduler 仅在 simulate_node 中使用；当主模型网络不通时允许 fallback 到 ECNU。
        # Agent 推理 / 交互始终使用 self.llm，不会被替换，保证 reasoning 一致性。
        induction_llm = self.llm
        ecnu_key = os.getenv("ECNU_API_KEY", "")
        if ecnu_key:
            try:
                ecnu_fallback = LLMFactory.create(
                    "ecnu",
                    base_url=os.getenv("ECNU_BASE_URL", "https://chat.ecnu.edu.cn/open/api/v1"),
                    api_key=ecnu_key,
                    model=os.getenv("ECNU_MODEL", "ecnu-max"),
                )
                induction_llm = FallbackAdapter(self.llm, ecnu_fallback)
                print("[Fallback] InductionScheduler 已启用 ECNU fallback（仅 simulate_node 环节）")
            except Exception as e:
                print(f"[Fallback] 无法创建 ECNU fallback: {e}，InductionScheduler 将直接使用主模型")
        self.induction_scheduler = InductionScheduler(self.induction_library, llm=induction_llm, max_per_agent=4, min_score=0.4)
        self.role_pool = pick_roles(12)
        # registry should include full role library to resolve participant_ids from any run
        self.registry = register_agents(ROLE_LIBRARY)
        self.language_mode = language_mode
        self.classical_translate = classical_translate
        self.use_event_goals = use_event_goals
        self.enable_round_summary = enable_round_summary
        self.resume_run = resume_run
        self.use_native_language = use_native_language
        self.translate_to_zh = translate_to_zh
        self.one_shot_action = one_shot_action
        self.reset_memory_on_run = reset_memory_on_run
        self.memory_clear_types = memory_clear_types or ["short_term", "social", "emotional", "episodic"]
        self.classical_converter_config = classical_converter_config or {}
        self.classical_translate_to_en = classical_translate_to_en
        self.include_system_agents = include_system_agents
        self.enable_positions = enable_positions
        self.relationship_mode = relationship_mode
        self.memory_summary_threshold = memory_summary_threshold
        self.pipeline_mode = "sim_eval_only"

        self.collector = DataCollectionAgent(
            name="collector",
            persona=Persona(role="数据采集Agent", goal="抓取并整理国际热点信息", personality="谨慎、精确"),
            memory=self.memory_factory.create("collector"),
            llm=self.llm,
            skills=self.skills,
            language_mode=self.language_mode,
            classical_translate=self.classical_translate,
            use_native_language=self.use_native_language,
            translate_to_zh=self.translate_to_zh,
            memory_summary_threshold=self.memory_summary_threshold,
        )
        self.cleaner = DataCleaningAgent(
            name="cleaner",
            persona=Persona(role="数据清洗Agent", goal="清洗并标准化数据", personality="严谨、系统化"),
            memory=self.memory_factory.create("cleaner"),
            llm=self.llm,
            skills=self.skills,
            language_mode=self.language_mode,
            classical_translate=self.classical_translate,
            use_native_language=self.use_native_language,
            translate_to_zh=self.translate_to_zh,
            memory_summary_threshold=self.memory_summary_threshold,
        )
        self.info_agent = InfoActionAgent(
            name="info_center",
            llm=self.llm,
            skills=self.skills,
            language_mode=self.language_mode,
            classical_translate=self.classical_translate,
            use_native_language=self.use_native_language,
            translate_to_zh=self.translate_to_zh,
        )

        self.media_agent = MediaAgent(
            name="media",
            persona=Persona(role="媒体Agent", goal="发布新闻并影响叙事", personality="吸引注意但保持事实"),
            memory=self.memory_factory.create("media", persona="media"),
            llm=self.llm,
            skills=self.skills,
            language_mode=self.language_mode,
            classical_translate=self.classical_translate,
            use_native_language=self.use_native_language,
            translate_to_zh=self.translate_to_zh,
            memory_summary_threshold=self.memory_summary_threshold,
        )
        self.citizen_agent = CitizenAgent(
            name="citizen",
            persona=Persona(role="普通人Agent", goal="代表公众情绪与反馈", personality="多元、易受影响"),
            memory=self.memory_factory.create("citizen", persona="citizen"),
            llm=self.llm,
            skills=self.skills,
            language_mode=self.language_mode,
            classical_translate=self.classical_translate,
            use_native_language=self.use_native_language,
            translate_to_zh=self.translate_to_zh,
            memory_summary_threshold=self.memory_summary_threshold,
        )
        self.policy_agent = PolicyAgent(
            name="policy",
            persona=Persona(role="政策Agent", goal="降低风险并稳定社会", personality="务实、审慎"),
            memory=self.memory_factory.create("policy", persona="policy"),
            llm=self.llm,
            skills=self.skills,
            language_mode=self.language_mode,
            classical_translate=self.classical_translate,
            use_native_language=self.use_native_language,
            translate_to_zh=self.translate_to_zh,
            memory_summary_threshold=self.memory_summary_threshold,
        )
        self.evaluator = EvaluationAgent(
            name="evaluator",
            persona=Persona(role="评估Agent", goal="评估风险并给出建议", personality="批判、平衡"),
            memory=self.memory_factory.create("evaluator"),
            llm=self.eval_llm,
            skills=self.skills,
            language_mode=self.language_mode,
            classical_translate=self.classical_translate,
            use_native_language=self.use_native_language,
            translate_to_zh=self.translate_to_zh,
            memory_summary_threshold=self.memory_summary_threshold,
        )
        self.summary_agent = SummaryAgent(
            name="summarizer",
            persona=Persona(role="观点总结Agent", goal="总结观点并提炼立场", personality="客观、简洁"),
            memory=self.memory_factory.create("summarizer"),
            llm=self.llm,
            skills=self.skills,
            language_mode="zh",
            classical_translate=False,
            use_native_language=False,
            translate_to_zh=False,
            memory_summary_threshold=self.memory_summary_threshold,
        )
        self.plotter = GraphPlotter(eval_llm=self.eval_llm, llm=self.llm, logger=self._log)

        self._register_memories()

    def _wait_if_paused(self) -> None:
        if not is_paused():
            return
        self._log("[control] simulation paused")
        while is_paused():
            time.sleep(0.5)
        self._log("[control] simulation resumed")

    def _plot_stance_history(
        self,
        history: List[Dict[str, Any]],
        round_logs: List[Dict[str, Any]],
        agent_names: List[str],
        language_mode: str,
        model_name: str,
        rounds: int,
        event_id: str,
    ) -> None:
        safe_model = str(model_name or "").strip()
        if not safe_model or safe_model.lower() == "unknown":
            safe_model = self.plotter.resolve_model_name({}, fallback_llm=self.llm)
        self.plotter.plot_stance_history(
            history=history,
            round_logs=round_logs,
            agent_names=agent_names,
            language_mode=language_mode,
            model_info={"model": safe_model},
            rounds=rounds,
            event_id=event_id,
        )
        return
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            self._log("[stance_eval] matplotlib not available, skip plotting")
            return
        if not history:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        # always save charts per run
        safe_model = "".join([c if c.isalnum() or c in "-_." else "_" for c in str(model_name or "unknown")])
        base = f"{ts}_{rounds}_{language_mode}_{safe_model}"
        # configure CJK font to avoid missing glyphs
        try:
            from matplotlib import font_manager as fm
            font_names = [f.name for f in fm.fontManager.ttflist]
            for fname in ["Microsoft YaHei", "SimHei"]:
                if fname in font_names:
                    plt.rcParams["font.sans-serif"] = [fname]
                    break
            plt.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

        os.makedirs("data/plots", exist_ok=True)

        # 1) stance line chart per agent
        rounds_sorted = sorted({r.get("round", 0) for r in round_logs}) if round_logs else list(range(1, rounds + 1))
        round_logs_sorted = sorted(round_logs, key=lambda r: r.get("round", 0)) if round_logs else []
        fig1, ax1 = plt.subplots(figsize=(8, 4.5))
        for idx, agent in enumerate(agent_names):
            series = []
            last_val = 0.0
            for rlog in round_logs_sorted:
                ev = (rlog.get("stance_eval") or {}).get(agent, {})
                if ev:
                    try:
                        score = float(ev.get("support", 0.0)) - float(ev.get("oppose", 0.0))
                        last_val = score
                    except Exception:
                        pass
                series.append(last_val)
            if series:
                ax1.plot(rounds_sorted[:len(series)], series, label=agent)
        ax1.set_xlabel("轮次")
        ax1.set_ylabel("态度(-1~1)")
        ax1.set_title(f"各Agent立场随轮次变化（{event_id}）")
        ax1.legend(fontsize=8)
        line_path = os.path.join("data/plots", f"{base}.jpg")
        fig1.tight_layout()
        fig1.savefig(line_path, dpi=150, format="jpg")
        plt.close(fig1)

        # 2) influence change heatmap
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        if agent_names and round_logs:
            r_index = {rv: i for i, rv in enumerate(rounds_sorted)}
            a_index = {a: i for i, a in enumerate(agent_names)}
            matrix = [[0 for _ in rounds_sorted] for _ in agent_names]
            for rlog in round_logs:
                ridx = r_index.get(rlog.get("round", 0))
                for inf in rlog.get("influences", []) or []:
                    if not isinstance(inf, dict):
                        continue
                    if not inf.get("changed"):
                        continue
                    a = inf.get("agent")
                    if a in a_index and ridx is not None:
                        matrix[a_index[a]][ridx] = 1
            ax2.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
            ax2.set_yticks(range(len(agent_names)))
            ax2.set_yticklabels(agent_names)
            ax2.set_xticks(range(len(rounds_sorted)))
            ax2.set_xticklabels([f"R{r}" for r in rounds_sorted])
            ax2.set_xlabel("Round")
            ax2.set_ylabel("Changed(1/0)")
            ax2.set_title("Influence Change Heatmap")
        else:
            ax2.text(0.1, 0.5, "No influence data", transform=ax2.transAxes)
        heat_path = os.path.join("data/plots", f"{base}_heatmap.jpg")
        fig2.tight_layout()
        fig2.savefig(heat_path, dpi=150, format="jpg")
        plt.close(fig2)

        # 3) stance distribution trend
        fig3, ax3 = plt.subplots(figsize=(8, 4.5))
        support_series = []
        neutral_series = []
        oppose_series = []
        for rlog in round_logs_sorted:
            adv = rlog.get("advanced_metrics", {}) if isinstance(rlog, dict) else {}
            dist = adv.get("stance_distribution", {}) if isinstance(adv, dict) else {}
            support_series.append(float(dist.get("support", 0.0)))
            neutral_series.append(float(dist.get("neutral", 0.0)))
            oppose_series.append(float(dist.get("oppose", 0.0)))
        if support_series:
            ax3.plot(rounds_sorted[:len(support_series)], support_series, label="support")
            ax3.plot(rounds_sorted[:len(neutral_series)], neutral_series, label="neutral")
            ax3.plot(rounds_sorted[:len(oppose_series)], oppose_series, label="oppose")
        ax3.set_xlabel("轮次")
        ax3.set_ylabel("比例")
        ax3.set_ylim(0.0, 1.0)
        ax3.set_title("群体立场比例变化")
        ax3.legend(fontsize=8)
        dist_path = os.path.join("data/plots", f"{base}_distribution.jpg")
        fig3.tight_layout()
        fig3.savefig(dist_path, dpi=150, format="jpg")
        plt.close(fig3)

        # 4) polarization and resistance trend
        fig4, ax4 = plt.subplots(figsize=(8, 4.5))
        p_series = []
        r_series = []
        for rlog in round_logs_sorted:
            adv = rlog.get("advanced_metrics", {}) if isinstance(rlog, dict) else {}
            p_series.append(float(adv.get("polarization_index", 0.0)))
            r_series.append(float(adv.get("resistance_factor", 0.0)))
        if p_series:
            ax4.plot(rounds_sorted[:len(p_series)], p_series, label="Polarization Index (P)")
            ax4.plot(rounds_sorted[:len(r_series)], r_series, label="Resistance Factor")
        ax4.set_xlabel("轮次")
        ax4.set_ylabel("指数")
        ax4.set_ylim(0.0, 1.0)
        ax4.set_title("极化与防御趋势")
        ax4.legend(fontsize=8)
        metric_path = os.path.join("data/plots", f"{base}_metrics.jpg")
        fig4.tight_layout()
        fig4.savefig(metric_path, dpi=150, format="jpg")
        plt.close(fig4)

        runtime_store.set_state({
            "last_chart_event_id": event_id,
            "last_chart_path": line_path,
            "last_heatmap_path": heat_path,
            "last_distribution_path": dist_path,
            "last_metric_path": metric_path,
        })
        self._log(f"[stance_eval] saved plot: {line_path}")
        self._log(f"[stance_eval] saved heatmap: {heat_path}")
        self._log(f"[stance_eval] saved distribution: {dist_path}")
        self._log(f"[stance_eval] saved metrics: {metric_path}")

    def _plot_sankey(self, rounds: List[Dict[str, Any]], agent_names: List[str], base: str) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            os.makedirs("data/plots", exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.axis("off")
            ax.set_title("Sankey Placeholder")
            ax.text(0.1, 0.6, f"agents={len(agent_names)}")
            ax.text(0.1, 0.4, f"rounds={len(rounds)}")
            path = os.path.join("data", "plots", f"{base}_sankey.jpg")
            fig.tight_layout()
            fig.savefig(path, dpi=120, format="jpg")
            plt.close(fig)
        except Exception:
            return

    def _plot_radar_profiles(self, rounds: List[Dict[str, Any]], agent_names: List[str], base: str) -> None:
        # Kept for backward compatibility with existing tests/callers.
        try:
            ordered = sorted(rounds, key=lambda x: x.get("round", 0)) if rounds else []
            self.plotter._plot_radar(ordered, agent_names, base)  # type: ignore[attr-defined]
            import matplotlib.pyplot as plt  # type: ignore
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.set_title("Radar Alias")
            ax.plot([0, 1], [0, 1])
            path = os.path.join("data", "plots", f"{base}_radar_profiles.jpg")
            fig.tight_layout()
            fig.savefig(path, dpi=120, format="jpg")
            plt.close(fig)
        except Exception:
            return

    def _register_memories(self):
        self.memory_hub.register("collector", self.collector.memory)
        self.memory_hub.register("cleaner", self.cleaner.memory)
        self.memory_hub.register("media", self.media_agent.memory)
        self.memory_hub.register("citizen", self.citizen_agent.memory)
        self.memory_hub.register("policy", self.policy_agent.memory)
        self.memory_hub.register("evaluator", self.evaluator.memory)

    def _clear_agent_memories(self):
        targets = [
            self.collector,
            self.cleaner,
            self.media_agent,
            self.citizen_agent,
            self.policy_agent,
            self.evaluator,
            self.summary_agent,
        ]
        for agent in targets:
            mem = getattr(agent, "memory", None)
            if mem and hasattr(mem, "clear"):
                try:
                    mem.clear(self.memory_clear_types)
                except Exception:
                    continue

    def _log(self, message: str) -> None:
        print(message)
        runtime_store.log(message)

    def _events_db_path(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "events.jsonl"))

    def _load_events_db(self) -> List[Dict[str, Any]]:
        path = self._events_db_path()
        if not os.path.exists(path):
            return []
        events: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(repair_obj(json.loads(line)))
                    except Exception:
                        continue
        except Exception:
            return []
        return events

    def _store_events_db(self, events: List[Dict[str, Any]]) -> None:
        path = self._events_db_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in events or []:
                if isinstance(item, dict):
                    f.write(json.dumps(repair_obj(item), ensure_ascii=False) + "\n")

    def build(self):
        graph = StateGraph(dict)

        def _resolve_agent_goal(agent: SocialAgent, event: Dict[str, Any]) -> str:
            goals = event.get("agent_goals", {}) if isinstance(event, dict) else {}
            if isinstance(goals, dict):
                if agent.name in goals:
                    return goals.get(agent.name, "")
                if agent.persona.role in goals:
                    return goals.get(agent.persona.role, "")
            return agent.persona.goal

        def bootstrap_node(state: Dict[str, Any]) -> Dict[str, Any]:
            if not self.resume_run or not runtime_store.current_run_path:
                runtime_store.start_run({
                    "language_mode": self.language_mode,
                    "model_info": state.get("model_info", {}) if isinstance(state, dict) else {},
                })
            target_event_id = state.get("event_id") or get_selected_event()
            last_event_id = None
            if isinstance(runtime_store.state, dict):
                last_event_id = runtime_store.state.get("last_event_id")
            if self.reset_memory_on_run and target_event_id and target_event_id != last_event_id:
                self._clear_agent_memories()
            runtime_store.set_state({
                "language_mode": self.language_mode,
                "last_event_id": target_event_id,
                "model_info": state.get("model_info", {}),
            })
            event_id = state.get("event_id")
            if event_id:
                set_selected_event(event_id)
            if state.get("case_event_only"):
                # honor provided event_id when case_event_only is enabled
                chosen_id = event_id or get_selected_event() or "evt_case_multiparty_a"
                set_selected_event(chosen_id)
            db_events = self._load_events_db()
            if state.get("case_event_only"):
                chosen_id = event_id or get_selected_event() or "evt_case_multiparty_a"
                db_events = [e for e in db_events if e.get("id") == chosen_id]
            if db_events:
                if len(db_events) > self.events_per_run:
                    db_events = random.sample(db_events, self.events_per_run)
                else:
                    random.shuffle(db_events)
            changed = False
            for i, e in enumerate(db_events):
                if not e.get("participant_ids"):
                    e["participant_ids"] = match_participants(e, self.role_pool)
                    e["event_index"] = i
                    changed = True
            if changed:
                self._store_events_db(db_events)
            runtime_store.set_state({"events": db_events})
            return {"events": db_events, "start_mode": "db"}

        def collect_node(state: Dict[str, Any]) -> Dict[str, Any]:
            # clean rebuild mode disables collection center; keep compatibility no-op.
            self._log("[collect_agent] skipped (disabled)")
            events = self._load_events_db()
            runtime_store.set_state({"events": events})
            return {"events": events, "logs": state.get("logs", [])}

        def clean_node(state: Dict[str, Any]) -> Dict[str, Any]:
            self._log("[clean_agent] start")
            runtime_store.set_context({"agent": "cleaner", "step": "clean"})
            result = self.cleaner.run(state)
            runtime_store.clear_context()
            self._log("[clean_agent] done")
            runtime_store.set_state({**state, "clean_data": result.get("action", {}).get("clean_data")})
            return {"clean_data": result.get("action", {}).get("clean_data"), "logs": state.get("logs", []) + [result]}

        def simulate_node(state: Dict[str, Any]) -> Dict[str, Any]:
            self._log("[simulate_agent] start")
            all_event_logs = []
            all_round_summaries: List[Dict[str, Any]] = []
            selected_event_id = get_selected_event()
            if state.get("case_event_only") and not selected_event_id:
                selected_event_id = "evt_case_multiparty_a"
            all_events = state.get("events", []) or []
            if not all_events:
                # fall back to latest DB events if collector failed
                db_events = self._load_events_db()
                all_events = db_events[: self.events_per_run]
                # backfill participant ids if missing
                changed = False
                for i, e in enumerate(all_events):
                    if not e.get("participant_ids"):
                        e["participant_ids"] = match_participants(e, self.role_pool)
                        e["event_index"] = i
                        changed = True
                    else:
                        # always dedupe existing list
                        seen = set()
                        new_ids = []
                        for rid in e.get("participant_ids", []):
                            if rid in seen:
                                continue
                            seen.add(rid)
                            new_ids.append(rid)
                        if new_ids != e.get("participant_ids", []):
                            e["participant_ids"] = new_ids
                            changed = True
                if changed:
                    self._store_events_db(all_events)
            events = all_events
            if state.get("case_event_only"):
                events = [e for e in all_events if e.get("id") == selected_event_id]
            if selected_event_id:
                filtered = [e for e in all_events if e.get("id") == selected_event_id]
                if filtered:
                    events = filtered
                else:
                    # allow selecting from DB even if not in current run
                    db_events = self._load_events_db()
                    db_match = [e for e in db_events if e.get("id") == selected_event_id]
                    if db_match:
                        # backfill participant ids if missing
                        changed = False
                        for i, e in enumerate(db_match):
                            if not e.get("participant_ids"):
                                e["participant_ids"] = match_participants(e, self.role_pool)
                                e["event_index"] = i
                                changed = True
                            else:
                                seen = set()
                                new_ids = []
                                for rid in e.get("participant_ids", []):
                                    if rid in seen:
                                        continue
                                    seen.add(rid)
                                    new_ids.append(rid)
                                if new_ids != e.get("participant_ids", []):
                                    e["participant_ids"] = new_ids
                                    changed = True
                        if changed:
                            self._store_events_db(db_match)
                        events = db_match
                        # include selected db event for UI context
                        all_events = all_events + [e for e in db_match if e not in all_events]
                    else:
                        events = all_events[:1]
            if not events:
                # no events available, keep a minimal trace for evaluation and UI
                self._log("[simulate_agent] no events to simulate")
                runtime_store.set_state({
                    "round": 0,
                    "events": all_events,
                    "simulation_result": {"events": []},
                })
                return {"simulation_result": {"events": []}, "logs": state.get("logs", [])}

            for event in events:
                event = repair_obj(event) if isinstance(event, dict) else event
                self._log(f"[simulate_agent] event_start: {event.get('title')} ({event.get('id')})")
                world_state = event.get("world_state", {}).copy()
                world_state["event_attention"] = {}
                constraints_cfg = load_event_constraints(event.get("id", ""))
                base_event_content = repair_text(event.get("content", ""))

                participant_ids = event.get("participant_ids", [])
                participants = [self.registry[pid] for pid in participant_ids if pid in self.registry]
                participant_agents: List[SocialAgent] = []
                for p in participants:
                    persona = persona_from_role(p)
                    agent = SocialAgent(
                        name=p.get("name", "role"),
                        persona=persona,
                        llm=self.llm,
                        memory=self.memory_factory.create(p.get("name", "role"), reset_types=self.memory_clear_types if self.reset_memory_on_run else None),
                        skills=self.skills,
                        language_mode=self.language_mode,
                        classical_translate=self.classical_translate,
                        use_native_language=self.use_native_language,
                        translate_to_zh=self.translate_to_zh,
                        memory_summary_threshold=self.memory_summary_threshold,
                    )
                    participant_agents.append(agent)

                agents = participant_agents.copy()
                if self.include_system_agents:
                    agents.extend([self.media_agent, self.citizen_agent, self.policy_agent])

                relationships = init_relationships([a.name for a in agents])
                world_state["relationships"] = relationships
                world_state_prompt = world_state
                # compact relationship matrix for prompt if configured
                if self.relationship_mode == "undirected":
                    names = [a.name for a in agents]
                    compact = []
                    for i in range(len(names)):
                        for j in range(i + 1, len(names)):
                            a = names[i]
                            b = names[j]
                            score = relationships.get(a, {}).get(b, None)
                            if score is not None:
                                compact.append({"a": a, "b": b, "score": round(score, 2)})
                    world_state_prompt = {
                        "resources": world_state.get("resources", {}),
                        "environment": world_state.get("environment", ""),
                        "scene_rules": world_state.get("scene_rules", []),
                        "relationships_compact": compact,
                    }
                if self.enable_positions:
                    world_state["positions"] = {a.name: {"x": random.random(), "y": random.random()} for a in agents}

                social_state = {
                    "scenario": event.get("title"),
                    "event": event,
                    "clean_data": state.get("clean_data", {}),
                    "event_summary": event.get("content", ""),
                    "event_content_current": event.get("content", ""),
                    "messages": [],
                    "metrics": {"bias": 0, "polarization": 0, "pollution": 0},
                    "feedback": state.get("feedback", {}),
                    "world_state": world_state_prompt,
                    "initial_stances": {},
                    "induction_state": {},
                    "inductions": {},
                    "stance_history": [],
                    "round_metrics": [],
                    "last_labels": {},
                }
                rounds: List[Dict[str, Any]] = []
                tools_list = self.tools.list_tools()
                if not self.enable_positions and "move" in tools_list:
                    tools_list = [t for t in tools_list if t != "move"]

                # resume support: reuse existing rounds if present
                resume_round_idx = 0
                if self.resume_run and isinstance(runtime_store.state, dict):
                    existing_sim = runtime_store.state.get("simulation_result", {})
                    if isinstance(existing_sim, dict):
                        for evlog in existing_sim.get("events", []):
                            if isinstance(evlog, dict) and isinstance(evlog.get("event"), dict) and evlog["event"].get("id") == event.get("id"):
                                rounds = evlog.get("rounds", []) or []
                                # use latest world_state if available
                                ws = runtime_store.state.get("world_state")
                                if isinstance(ws, dict):
                                    world_state = ws
                                # rebuild messages to avoid losing context
                                social_state["messages"] = []
                                social_state["metrics"] = {"bias": 0, "polarization": 0, "pollution": 0}
                                for rlog in rounds:
                                    for it in rlog.get("interactions", []):
                                        social_state["messages"].append({"agent": it.get("agent"), "interaction": {"parsed": it.get("parsed", {})}})
                                        impact = it.get("impact", {}) or {}
                                        social_state["metrics"]["bias"] += int(bool(impact.get("bias")))
                                        social_state["metrics"]["polarization"] += int(bool(impact.get("polarization")))
                                        social_state["metrics"]["pollution"] += int(bool(impact.get("pollution")))
                                # if last round is partial, resume from it; else next round
                                if rounds and len(rounds[-1].get("interactions", [])) < len(agents):
                                    resume_round_idx = max(0, rounds[-1].get("round", 1) - 1)
                                else:
                                    resume_round_idx = len(rounds)
                                break

                # write initial state for UI
                runtime_store.set_state({
                    "current_event": event,
                    "world_state": world_state,
                    "round": 0,
                    "events": all_events,
                    "simulation_result": {"events": all_event_logs},
                    "round_summaries": all_round_summaries,
                })

                # initialize stances once per event to anchor discussion
                for agent in agents:
                    runtime_store.set_context({
                        "event_id": event.get("id"),
                        "event_title": event.get("title"),
                        "round": 0,
                        "agent": agent.name,
                    })
                    stance_prompt = (
                        f"{agent._persona_header()}"
                        "请基于你的价值观与国家体制认知，给出一句明确立场（不超过40字，可带情绪色彩）。"
                        f"事件：{event.get('title')}。内容：{event.get('content')}"
                    )
                    stance_text = agent._call_llm("init_stance", stance_prompt)
                    runtime_store.clear_context()
                    if stance_text:
                        social_state["initial_stances"][agent.name] = stance_text.strip().split("\n")[0]
                        try:
                            agent.memory.add_memory(
                                f"初始立场: {stance_text}",
                                "long_term",
                                {"step": "init_stance", "event_id": event.get("id")},
                            )
                        except Exception:
                            pass
                        try:
                            self.induction_scheduler.update_state(
                                social_state.get("induction_state", {}),
                                agent.name,
                                stance_text,
                                False,
                            )
                        except Exception:
                            pass

                total_rounds = self.interaction_rounds
                if "multiparty" in str(event.get("id", "")):
                    total_rounds = max(total_rounds, 5)
                for r in range(resume_round_idx, total_rounds):
                    self._wait_if_paused()
                    self._log(f"[simulate_agent] round {r + 1}/{total_rounds}")
                    round_content, round_constraints = content_for_round(constraints_cfg, base_event_content, r + 1)
                    round_content = repair_text(round_content)
                    round_constraints = [repair_text(c) for c in (round_constraints or [])]
                    event_for_round = dict(event)
                    event_for_round["content"] = round_content
                    prev_content = rounds[-1].get("event_content") if rounds else None
                    prev_constraints = rounds[-1].get("event_constraints") if rounds else None
                    changed_by_constraints = bool(
                        (prev_content is not None and prev_content != round_content)
                        or (prev_constraints is not None and prev_constraints != round_constraints)
                    )
                    if changed_by_constraints:
                        runtime_store.log(
                            f"[event_constraints] round {r + 1} content/constraints updated "
                            f"for event={event.get('id')}"
                        )
                    social_state["event"] = event_for_round
                    social_state["event_summary"] = round_content
                    social_state["event_content_current"] = round_content
                    user_msgs = pop_user_messages()
                    if user_msgs:
                        for msg in user_msgs:
                            social_state["messages"].append({
                                "agent": msg.get("source", "human"),
                                "interaction": {"parsed": {"message": msg.get("content", "")}},
                            })
                        runtime_store.log(f"[control] injected {len(user_msgs)} user message(s)")

                    stance_distribution = {-1: 0, 0: 0, 1: 0}
                    for agent in agents:
                        st = social_state["induction_state"].setdefault(agent.name, {"last_direction": 0, "stable_rounds": 0, "reinforce_target": False})
                        stance_distribution[st.get("last_direction", 0)] = stance_distribution.get(st.get("last_direction", 0), 0) + 1

                    last_adv = (social_state.get("round_metrics") or [])[-1] if social_state.get("round_metrics") else {}
                    saturation_on = bool((last_adv or {}).get("saturation", {}).get("is_saturated"))

                    try:
                        inductions = self.induction_scheduler.schedule_round(
                            event_for_round,
                            agents,
                            social_state["induction_state"],
                            stance_distribution,
                            r + 1,
                            saturation=saturation_on,
                        )
                    except TypeError:
                        # Backward-compat: older scheduler may not accept `saturation` kwarg.
                        inductions = self.induction_scheduler.schedule_round(
                            event_for_round,
                            agents,
                            social_state["induction_state"],
                            stance_distribution,
                            r + 1,
                        )
                    # include user messages as induction items with equal weight
                    if user_msgs:
                        for agent in agents:
                            extra = []
                            for msg in user_msgs:
                                content = msg.get("content", "")
                                if not content:
                                    continue
                                score = compute_interest_score(agent.persona, {"interest_tags": []})
                                extra.append({
                                    "id": msg.get("id", ""),
                                    "category": "external",
                                    "content": content,
                                    "interest_tags": [],
                                    "target_groups": [],
                                    "direction": "neutral",
                                    "intensity": 3,
                                    "score": score,
                                    "event_id": event.get("id", ""),
                                    "source": "human",
                                })
                            inductions.setdefault(agent.name, [])
                            inductions[agent.name].extend(extra)
                    if r < len(rounds):
                        round_log = rounds[r]
                        if "inductions" not in round_log:
                            round_log["inductions"] = {}
                        if "interactions" not in round_log:
                            round_log["interactions"] = []
                    else:
                        round_log = {"round": r + 1, "event_id": event["id"], "interactions": [], "inductions": {}}
                    round_log["event_content"] = round_content
                    round_log["event_constraints"] = round_constraints
                    round_log["event_constraints_changed"] = changed_by_constraints
                    social_state["inductions"] = inductions
                    round_log["inductions"] = inductions
                    # log inductions per round for UI/trace
                    round_induction = {"round": r + 1, "event_id": event.get("id"), "inductions": inductions}
                    runtime_store.log(f"[induction] round {r + 1} {round_induction}")
                    self._log(f"[induction] round {r + 1} injected for {len(inductions)} agents")
                    runtime_store.set_state({"round_inductions": round_induction})
                    acted_agents = {it.get("agent") for it in round_log.get("interactions", [])}
                    for agent in agents:
                        if agent.name in acted_agents:
                            continue
                        self._log(f"[simulate_agent] speaking: {agent.name}")
                        for item in social_state.get("inductions", {}).get(agent.name, []) or []:
                            try:
                                agent.memory.add_memory(
                                    f"参考信息: {item.get('content','')} (score={item.get('score')})",
                                    "short_term",
                                    {
                                        "step": "induction",
                                        "event_id": event.get("id"),
                                        "induction_id": item.get("id", ""),
                                        "score": item.get("score"),
                                        "direction": item.get("direction"),
                                    },
                                )
                            except Exception:
                                pass
                        runtime_store.set_context({
                            "event_id": event.get("id"),
                            "event_title": event.get("title"),
                            "round": r + 1,
                            "agent": agent.name,
                        })
                        agent_goal = _resolve_agent_goal(agent, event_for_round)
                        use_event_goals = self.use_event_goals
                        if event.get("id") in ["evt_case_multiparty_china", "evt_case_multiparty_a", "evt_case_multiparty_0423"]:
                            use_event_goals = False
                            agent_goal = ""
                        act = agent.act(
                            {
                                "social_state": social_state,
                                "round": r + 1,
                                "tools": tools_list,
                                "agent_goal": agent_goal,
                                "use_event_goals": use_event_goals,
                                "one_shot_action": self.one_shot_action,
                                "classical_config": self.classical_converter_config,
                                "classical_to_en": self.classical_translate_to_en,
                            },
                            {},
                        )
                        interaction = act.get("interaction", {})
                        runtime_store.clear_context()
                        # interaction is parse_with_schema wrapper: {parsed, valid, error, raw}
                        parsed_obj = interaction.get("parsed", {}) if isinstance(interaction, dict) else {}

                        tool_name = parsed_obj.get("tool")
                        tool_args = parsed_obj.get("args", {}) or {}
                        # normalize tool args
                        if "agent" not in tool_args:
                            tool_args["agent"] = agent.name
                        if tool_name in ["publish_news", "propose_policy"] and "event_id" not in tool_args:
                            tool_args["event_id"] = event.get("id")
                        if tool_name in ["comment", "help", "harm", "mobilize", "boycott", "negotiate", "collaborate"] and "target" not in tool_args:
                            tool_args["target"] = event.get("title", "public")
                        if tool_name in ["comment", "publish_news", "propose_policy", "mobilize", "help", "harm", "boycott", "negotiate", "collaborate", "investigate", "verify"] and "content" not in tool_args:
                            tool_args["content"] = parsed_obj.get("message", "")

                        tool_result = None
                        if tool_name:
                            try:
                                tool_result = self.tools.call(tool_name, **tool_args)
                            except Exception as exc:
                                tool_result = {"tool": tool_name, "error": str(exc)}

                        message = parsed_obj.get("message") or (tool_result or {}).get("content") or (f"{tool_name} 执行" if tool_name else "行动")

                        impact = parsed_obj.get("impact", {}) if isinstance(parsed_obj, dict) else {}
                        if not impact:
                            impact = {"bias": False, "polarization": False, "pollution": False}
                        for k in ["bias", "polarization", "pollution"]:
                            impact.setdefault(k, False)

                        # record stance / influence signals to agent memory
                        influence = impact.get("influence", {}) if isinstance(impact, dict) else {}
                        induction_items = social_state.get("inductions", {}).get(agent.name, []) or []
                        induction_ids = []
                        for it in induction_items:
                            if not isinstance(it, dict):
                                continue
                            iid = it.get("id") or it.get("content")
                            if iid:
                                induction_ids.append(iid)
                        if isinstance(influence, dict):
                            if influence.get("changed") and not influence.get("from_inductions") and induction_ids:
                                influence["from_inductions"] = induction_ids
                        if isinstance(influence, dict) and influence.get("changed"):
                            try:
                                agent.memory.add_memory(
                                    f"受他人影响: {influence.get('reason','')}",
                                    "social",
                                    {
                                        "step": "influence",
                                        "from_agents": influence.get("from_agents", []),
                                        "from_inductions": influence.get("from_inductions", []),
                                    },
                                )
                            except Exception:
                                pass
                        try:
                            stance_text = parsed_obj.get("stance", "") if isinstance(parsed_obj, dict) else ""
                            self.induction_scheduler.update_state(
                                social_state.get("induction_state", {}),
                                agent.name,
                                stance_text,
                                bool(isinstance(influence, dict) and influence.get("changed")),
                            )
                        except Exception:
                            pass

                        # apply tool-driven state updates when impact is missing
                        if tool_name == "adjust_resource":
                            res = tool_args.get("resource")
                            delta = tool_args.get("delta", 0)
                            if res:
                                impact.setdefault("resource_delta", {})
                                impact["resource_delta"][res] = impact["resource_delta"].get(res, 0) + int(delta)
                        if tool_name == "use_resource":
                            res = tool_args.get("resource")
                            amount = tool_args.get("amount", 0)
                            if res:
                                impact.setdefault("resource_delta", {})
                                impact["resource_delta"][res] = impact["resource_delta"].get(res, 0) - int(amount)
                        if tool_name == "move" and "x" in tool_args and "y" in tool_args:
                            impact["move"] = {"x": float(tool_args["x"]), "y": float(tool_args["y"])}

                        influence_trace = {
                            "round": r + 1,
                            "agent": agent.name,
                            "changed": bool(isinstance(influence, dict) and influence.get("changed")),
                            "reason": influence.get("reason", "") if isinstance(influence, dict) else "",
                            "from_agents": influence.get("from_agents", []) if isinstance(influence, dict) else [],
                            "from_inductions": influence.get("from_inductions", []) if isinstance(influence, dict) else [],
                            "induction_items": [
                                {
                                    "id": it.get("id", ""),
                                    "content": it.get("content", ""),
                                    "score": it.get("score", None),
                                }
                                for it in induction_items if isinstance(it, dict)
                            ],
                        }
                        round_log.setdefault("influences", []).append(influence_trace)
                        if influence_trace["changed"]:
                            runtime_store.log(
                                f"[influence] round {r + 1} {agent.name} changed reason={influence_trace['reason']} "
                                f"from_agents={influence_trace['from_agents']} from_inductions={influence_trace['from_inductions']}"
                            )

                        round_log["interactions"].append({
                            "agent": agent.name,
                            "message": message,
                            "impact": impact,
                            "tool": tool_name,
                            "args": tool_args,
                            "tool_result": tool_result,
                            "stance": parsed_obj.get("stance", ""),
                            "induction_ids": induction_ids,
                            "parsed": parsed_obj if isinstance(parsed_obj, dict) else {},
                        })
                        social_state["messages"].append({"agent": agent.name, "interaction": interaction})

                        social_state["metrics"]["bias"] += int(bool(impact.get("bias")))
                        social_state["metrics"]["polarization"] += int(bool(impact.get("polarization")))
                        social_state["metrics"]["pollution"] += int(bool(impact.get("pollution")))

                        if impact.get("resource_delta"):
                            for k, v in impact.get("resource_delta", {}).items():
                                world_state["resources"][k] = world_state["resources"].get(k, 0) + v
                        # clamp resource bounds to avoid runaway values
                        if world_state.get("resources"):
                            for rk, rv in list(world_state["resources"].items()):
                                try:
                                    world_state["resources"][rk] = max(0, min(20, float(rv)))
                                except Exception:
                                    continue
                        if impact.get("relation_delta"):
                            for k, v in impact.get("relation_delta", {}).items():
                                if k not in world_state["relationships"].get(agent.name, {}):
                                    continue
                                delta = None
                                if isinstance(v, (int, float)):
                                    delta = float(v)
                                elif isinstance(v, dict):
                                    # accept common numeric fields
                                    for key in ["delta", "value", "score", "amount"]:
                                        if isinstance(v.get(key), (int, float)):
                                            delta = float(v.get(key))
                                            break
                                if delta is None:
                                    continue
                                world_state["relationships"][agent.name][k] = max(
                                    0.0,
                                    min(1.0, world_state["relationships"][agent.name][k] + delta),
                                )
                        if impact.get("move"):
                            if self.enable_positions and world_state.get("positions") is not None:
                                world_state["positions"][agent.name] = impact.get("move")
                        else:
                            if self.enable_positions and world_state.get("positions") is not None:
                                pos = world_state["positions"][agent.name]
                                world_state["positions"][agent.name] = {
                                    "x": min(1.0, max(0.0, pos["x"] + random.uniform(-0.05, 0.05))),
                                    "y": min(1.0, max(0.0, pos["y"] + random.uniform(-0.05, 0.05))),
                                }

                        # stream partial round updates for UI
                        runtime_store.set_state({
                            "current_event": event_for_round,
                            "world_state": world_state,
                            "round": r + 1,
                            "events": all_events,
                            "event_constraints": round_constraints,
                            "simulation_result": {"events": all_event_logs + [{"event": event, "rounds": rounds + [round_log], "final_state": social_state}]},
                            "round_summaries": all_round_summaries,
                            "round_influences": round_log.get("influences", []),
                        })

                        # social agents now act autonomously in a single step per round; skip per-step reflection prompts

                    if self.enable_round_summary:
                        runtime_store.set_context({
                            "event_id": event.get("id"),
                            "event_title": event.get("title"),
                            "round": r + 1,
                            "agent": "summarizer",
                        })
                        summary = self.summary_agent.summarize_round(
                            event.get("id"),
                            r + 1,
                            round_log["interactions"],
                        )
                        runtime_store.clear_context()
                        round_log["summary"] = summary
                        all_round_summaries.append(summary)

                    # stance evaluation per round
                    stance_eval = evaluate_round(self.eval_llm, round_log["interactions"], language_mode=self.language_mode)
                    round_log["stance_eval"] = stance_eval
                    agg = aggregate_round(stance_eval)
                    agg["round"] = r + 1
                    social_state["stance_history"].append(agg)
                    previous_labels = social_state.get("last_labels", {}) if isinstance(social_state.get("last_labels", {}), dict) else {}
                    adv = compute_round_metrics(
                        stance_eval=stance_eval,
                        influences=round_log.get("influences", []) or [],
                        previous_labels=previous_labels,
                        round_idx=r + 1,
                    )
                    round_log["advanced_metrics"] = adv
                    social_state.setdefault("round_metrics", []).append(adv)
                    social_state["last_labels"] = adv.get("labels", {})
                    for def_name in adv.get("defense_agents", []) or []:
                        st = social_state["induction_state"].setdefault(def_name, {"last_direction": 0, "stable_rounds": 0, "reinforce_target": False, "defense": False})
                        st["defense"] = True
                    for agent_name, ev in stance_eval.items():
                        self._log(f"[stance_eval] round {r + 1} {agent_name} support={float(ev.get('support',0)):.2f} neutral={float(ev.get('neutral',0)):.2f} oppose={float(ev.get('oppose',0)):.2f} label={ev.get('label')}")
                    diffusion = propagate([event], round_log["interactions"], world_state)
                    world_state.update(diffusion)
                    rounds.append(round_log)

                    # update runtime state per round so timeline isn't empty
                    runtime_store.set_state({
                        "current_event": event_for_round,
                        "world_state": world_state,
                        "round": r + 1,
                        "events": all_events,
                        "event_constraints": round_constraints,
                        "simulation_result": {"events": all_event_logs + [{"event": event, "rounds": rounds, "final_state": social_state}]},
                        "round_influences": round_log.get("influences", []),
                        "advanced_metrics": adv,
                    })

                if social_state.get("stance_history"):
                    model_name = ""
                    if isinstance(state, dict):
                        model_name = state.get("model_info", {}).get("model", "")
                    self._plot_stance_history(
                        social_state.get("stance_history", []),
                        rounds,
                        [a.name for a in agents],
                        self.language_mode,
                        model_name,
                        total_rounds,
                        event.get("id", ""),
                    )

                event_log = {"event": event, "rounds": rounds, "final_state": social_state}
                all_event_logs.append(event_log)
                runtime_store.log_agent("system", "event_done", event.get("title", ""), json.dumps(event_log, ensure_ascii=False))

            simulation_result = {"events": all_event_logs}
            parsed_sim = parse_with_schema(json.dumps(simulation_result, ensure_ascii=False), SIMULATION_SCHEMA)
            self._log("[simulate_agent] done")
            try:
                digest_events = []
                for evlog in all_event_logs:
                    ev = evlog.get("event", {})
                    rounds = []
                    # round 0: initial stances
                    initial = (evlog.get("final_state") or {}).get("initial_stances", {}) or {}
                    if initial:
                        init_interactions = []
                        for agent_name, stance_text in initial.items():
                            init_interactions.append({
                                "agent": agent_name,
                                "tool": "init_stance",
                                "message": f"初始立场: {stance_text}",
                                "stance": stance_text,
                                "argument": {"claim": stance_text, "evidence": ["初始立场"]},
                                "impact": {"bias": False, "polarization": False, "pollution": False},
                                "induction_ids": [],
                            })
                        rounds.append({
                            "round": 0,
                            "interactions": init_interactions,
                            "summary": None,
                            "stance_eval": None,
                            "influences": [],
                            "influence_summary": [],
                        })
                    for rlog in evlog.get("rounds", []):
                        interactions = []
                        for it in rlog.get("interactions", []):
                            parsed_payload = (it.get("parsed") or {}) if isinstance(it.get("parsed"), dict) else {}
                            argument_payload = parsed_payload.get("argument") if isinstance(parsed_payload, dict) else None
                            interactions.append({
                                "agent": it.get("agent"),
                                "tool": it.get("tool"),
                                "message": repair_text(it.get("message")),
                                "stance": it.get("stance"),
                                "argument": argument_payload,
                                "viewpoint": parsed_payload.get("viewpoint", ""),
                                "reasoning": parsed_payload.get("reasoning", ""),
                                "target_statement": parsed_payload.get("target_statement", ""),
                                "impact": it.get("impact"),
                                "induction_ids": it.get("induction_ids", []),
                            })
                        influence_summary = []
                        inductions = rlog.get("inductions", {}) or {}
                        for inf in rlog.get("influences", []) or []:
                            if not isinstance(inf, dict) or not inf.get("changed"):
                                continue
                            agent = inf.get("agent")
                            ind_items = []
                            for item in inductions.get(agent, []) if isinstance(inductions, dict) else []:
                                if not isinstance(item, dict):
                                    continue
                                ind_items.append({
                                    "id": item.get("id"),
                                    "content": item.get("content"),
                                    "score": item.get("score"),
                                    "direction": item.get("direction"),
                                })
                            influence_summary.append({
                                "agent": agent,
                                "reason": inf.get("reason", ""),
                                "from_agents": inf.get("from_agents", []),
                                "from_inductions": inf.get("from_inductions", []),
                                "inductions": ind_items,
                            })
                        rounds.append({
                            "round": rlog.get("round"),
                            "event_content": repair_text(rlog.get("event_content") or ev.get("content", "")),
                            "event_constraints": [repair_text(c) for c in (rlog.get("event_constraints") or [])],
                            "event_constraints_changed": bool(rlog.get("event_constraints_changed", False)),
                            "interactions": interactions,
                            "summary": rlog.get("summary"),
                            "stance_eval": rlog.get("stance_eval"),
                            "influences": rlog.get("influences", []),
                            "influence_summary": influence_summary,
                        })
                    digest_events.append({
                        "event": {
                            "id": ev.get("id"),
                            "title": ev.get("title"),
                            "content": ev.get("content"),
                        },
                        "rounds": rounds,
                    })
                digest_payload = {
                    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                    "events": digest_events,
                }
                path = runtime_store.save_digest(digest_payload)
                self._log(f"[digest] saved {path}")
            except Exception as exc:
                self._log(f"[digest] save failed: {exc}")
            return {
                "simulation_result": simulation_result,
                "simulation_schema": parsed_sim,
                "round_summaries": all_round_summaries,
                "logs": state.get("logs", []) + [simulation_result],
            }

        def evaluate_node(state: Dict[str, Any]) -> Dict[str, Any]:
            self._log("[evaluate_agent] start")
            sim_result = state.get("simulation_result")
            if not sim_result:
                # try to recover from runtime_store to avoid false empty
                sim_result = runtime_store.state.get("simulation_result") if isinstance(runtime_store.state, dict) else None
            if not sim_result:
                runtime_store.save_run(tag="empty_simulation")
                return {
                    "evaluation": {"error": "empty_simulation"},
                    "feedback": {},
                    "logs": state.get("logs", []),
                    "simulation_result": {"events": []},
                }
            # collect round summaries for evaluator
            round_summaries = []
            if isinstance(sim_result, dict):
                for evlog in sim_result.get("events", []):
                    if not isinstance(evlog, dict):
                        continue
                    for r in evlog.get("rounds", []) if isinstance(evlog.get("rounds", []), list) else []:
                        summary = r.get("summary") if isinstance(r, dict) else None
                        if summary:
                            round_summaries.append(summary)
            eval_state = dict(state)
            eval_state["simulation_result"] = sim_result
            eval_state["round_summaries"] = round_summaries
            runtime_store.set_context({"agent": "evaluator", "step": "evaluate"})
            result = self.evaluator.run(eval_state)
            runtime_store.clear_context()
            evaluation = result.get("action", {}).get("evaluation")
            feedback = {}
            if isinstance(evaluation, dict):
                parsed = evaluation.get("parsed")
                if isinstance(parsed, dict):
                    feedback = parsed.get("feedback", {})
                    persona_updates = parsed.get("persona_updates", [])
                    if isinstance(persona_updates, list):
                        for upd in persona_updates:
                            agent_name = upd.get("agent")
                            if agent_name == "media":
                                self.media_agent.update_persona(upd)
                            elif agent_name == "citizen":
                                self.citizen_agent.update_persona(upd)
                            elif agent_name == "policy":
                                self.policy_agent.update_persona(upd)
            runtime_store.save_run(tag="ok")
            self._log("[evaluate_agent] done")
            # ensure events are preserved for UI; fall back to simulation_result
            events = state.get("events", [])
            if not events and isinstance(sim_result, dict):
                events = [e.get("event") for e in sim_result.get("events", []) if isinstance(e, dict) and e.get("event")]
            # pass through simulation_result so downstream state keeps it
            return {
                "evaluation": evaluation,
                "feedback": feedback,
                "logs": state.get("logs", []) + [result],
                "simulation_result": sim_result,
                "events": events,
            }

        graph.add_node("simulate_agent", simulate_node)
        graph.add_node("evaluate_agent", evaluate_node)
        graph.add_node("bootstrap", bootstrap_node)

        graph.set_entry_point("bootstrap")
        graph.add_edge("simulate_agent", "evaluate_agent")

        def decide_bootstrap(state: Dict[str, Any]):
            return "simulate_agent"

        graph.add_conditional_edges("bootstrap", decide_bootstrap)

        def decide_next(state: Dict[str, Any]):
            iters = state.get("iterations", 0) + 1
            state["iterations"] = iters
            runtime_store.set_state(state)
            if iters < self.max_iterations:
                return "simulate_agent"
            return END

        graph.add_conditional_edges("evaluate_agent", decide_next)
        return graph.compile()

    def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        app = self.build()
        return app.invoke(initial_state)

    def search_memory(self, query: str, mem_type: str = None):
        return self.memory_hub.search(query, mem_type=mem_type)

    def summarize_memory(self, mem_type: str = None):
        return self.memory_hub.summarize(self.llm, mem_type=mem_type)







