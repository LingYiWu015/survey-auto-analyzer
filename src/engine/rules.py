"""Rule definitions for matching variable types to analysis methods.

Each rule maps a condition (based on variable types present in the survey)
to one or more analysis methods with priority scores and dependency info.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum


class MethodCategory(Enum):
    """Analysis method categories."""
    QUANTITATIVE = "定量分析"
    TEXT = "文本分析"
    QUALITATIVE = "定性分析"
    COMPREHENSIVE = "综合评价"


@dataclass
class MethodInfo:
    """Metadata for an analysis method."""

    method_id: str                              # Unique ID, e.g. "reliability"
    name: str                                   # Chinese name: "信效度检验"
    category: MethodCategory                    # Category enum
    description: str                            # One-line description
    triggers: List[str] = field(default_factory=list)  # Variable types that trigger this
    prerequisites: List[str] = field(default_factory=list)  # Method IDs that must run first
    priority: int = 50                          # Base priority (0-100)
    is_foundational: bool = False               # True for always-applicable methods


# ─── Method Registry ────────────────────────────────────────────────

METHODS: Dict[str, MethodInfo] = {
    # ── 定量分析 ──
    "reliability": MethodInfo(
        method_id="reliability",
        name="信效度检验",
        category=MethodCategory.QUANTITATIVE,
        description="Cronbach's α 信度 + KMO/Bartlett 效度检验",
        triggers=["likert_scale"],
        priority=95,
        is_foundational=False,
    ),
    "descriptive": MethodInfo(
        method_id="descriptive",
        name="描述统计与列联表",
        category=MethodCategory.QUANTITATIVE,
        description="频数/百分比/均值±标准差，样本画像",
        triggers=[],  # Always applicable
        priority=100,
        is_foundational=True,
    ),
    "ttest_anova": MethodInfo(
        method_id="ttest_anova",
        name="T检验与方差分析",
        category=MethodCategory.QUANTITATIVE,
        description="独立样本T检验、单因素ANOVA、事后多重比较",
        triggers=["continuous", "likert_scale"],
        priority=70,
    ),
    "correlation": MethodInfo(
        method_id="correlation",
        name="相关分析",
        category=MethodCategory.QUANTITATIVE,
        description="Pearson/Spearman 相关系数矩阵",
        triggers=["continuous", "likert_scale"],
        priority=65,
    ),
    "regression": MethodInfo(
        method_id="regression",
        name="回归分析",
        category=MethodCategory.QUANTITATIVE,
        description="线性/Logistic/Lasso回归，中介/调节效应",
        triggers=["continuous", "likert_scale"],
        prerequisites=["correlation"],
        priority=60,
    ),
    "factor_analysis": MethodInfo(
        method_id="factor_analysis",
        name="因子分析",
        category=MethodCategory.QUANTITATIVE,
        description="探索性因子分析，主成分法，Varimax旋转",
        triggers=["likert_scale"],
        prerequisites=["reliability"],
        priority=55,
    ),
    "clustering": MethodInfo(
        method_id="clustering",
        name="聚类分析",
        category=MethodCategory.QUANTITATIVE,
        description="K-Means 聚类，用户分群，雷达图",
        triggers=["continuous", "likert_scale"],
        priority=40,
    ),
    "correspondence": MethodInfo(
        method_id="correspondence",
        name="对应分析",
        category=MethodCategory.QUANTITATIVE,
        description="CA/MCA 对应分析，感知图（双标图）",
        triggers=["single_choice", "multi_choice"],
        prerequisites=["chi_square"],
        priority=35,
    ),
    "chi_square": MethodInfo(
        method_id="chi_square",
        name="卡方检验",
        category=MethodCategory.QUANTITATIVE,
        description="独立性检验，Cramer's V 效应量",
        triggers=["single_choice", "multi_choice"],
        priority=60,
    ),
    "runs_test": MethodInfo(
        method_id="runs_test",
        name="游程检验",
        category=MethodCategory.QUANTITATIVE,
        description="随机性检验，序列是否为随机排列",
        triggers=["continuous"],  # Rarely triggered — needs ordered data
        priority=10,
    ),
    "nps": MethodInfo(
        method_id="nps",
        name="NPS净推荐值",
        category=MethodCategory.QUANTITATIVE,
        description="推荐者/被动者/贬损者分组，NPS得分",
        triggers=["nps"],
        priority=75,
    ),

    # ── 文本分析 ──
    "sentiment": MethodInfo(
        method_id="sentiment",
        name="情感分析",
        category=MethodCategory.TEXT,
        description="SnowNLP 中文情感分析，正负面分布",
        triggers=["text"],
        priority=80,
    ),
    "lda_topic": MethodInfo(
        method_id="lda_topic",
        name="LDA主题模型",
        category=MethodCategory.TEXT,
        description="gensim LDA，主题-词分布，pyLDAvis可视化",
        triggers=["text"],
        prerequisites=["sentiment"],
        priority=70,
    ),
    "tfidf": MethodInfo(
        method_id="tfidf",
        name="TF-IDF关键词提取",
        category=MethodCategory.TEXT,
        description="TfidfVectorizer，关键词提取，词云",
        triggers=["text"],
        priority=65,
    ),
    "so_pmi": MethodInfo(
        method_id="so_pmi",
        name="SO-PMI情感词典",
        category=MethodCategory.TEXT,
        description="点互信息构建领域情感词典",
        triggers=["text"],
        priority=30,
    ),
    "sdgs": MethodInfo(
        method_id="sdgs",
        name="SDGs映射",
        category=MethodCategory.TEXT,
        description="SDGs 17项目标关键词匹配",
        triggers=["text"],
        priority=15,
    ),

    # ── 综合评价 ──
    "fuzzy_eval": MethodInfo(
        method_id="fuzzy_eval",
        name="模糊综合评判",
        category=MethodCategory.COMPREHENSIVE,
        description="模糊矩阵合成，隶属度向量，去模糊化",
        triggers=["likert_scale"],
        priority=25,
    ),
    "ccsi_acsi": MethodInfo(
        method_id="ccsi_acsi",
        name="CCSI与ACSI",
        category=MethodCategory.COMPREHENSIVE,
        description="结构方程模型满意度指数，IPA四象限图",
        triggers=["likert_scale"],
        prerequisites=["reliability", "factor_analysis"],
        priority=20,
    ),
    "fsqca": MethodInfo(
        method_id="fsqca",
        name="fsQCA",
        category=MethodCategory.COMPREHENSIVE,
        description="模糊集定性比较分析，必要性/充分性分析",
        triggers=["single_choice", "likert_scale"],
        priority=10,
    ),
}


# ─── Rule Definitions ────────────────────────────────────────────────

@dataclass
class Rule:
    """A rule that maps survey characteristics to one or more methods."""

    rule_id: int
    name: str
    condition: Callable[["SurveyContext"], bool]  # Condition function
    methods: List[str]                            # Method IDs to trigger
    priority_boost: int = 0                       # Extra priority points
    note: str = ""                                # Human-readable reason


class SurveyContext:
    """Encapsulates survey characteristics for rule evaluation.

    This is a lightweight snapshot of the survey definition used by rules.
    """

    def __init__(self, survey_def):
        from src.parser.survey_parser import SurveyDefinition
        self.survey: SurveyDefinition = survey_def
        self.var_types = survey_def.var_types_summary
        self.n_likert = self.var_types.get("likert_scale", 0)
        self.n_continuous = self.var_types.get("continuous", 0)
        self.n_single_choice = self.var_types.get("single_choice", 0)
        self.n_multi_choice = self.var_types.get("multi_choice", 0)
        self.n_text = self.var_types.get("text", 0)
        self.n_nps = self.var_types.get("nps", 0)
        self.n_demographic = self.var_types.get("demographic", 0)
        self.n_categorical = self.n_single_choice + self.n_multi_choice
        self.n_numeric = self.n_continuous + self.n_likert
        self.n_total = survey_def.n_questions
        self.sample_size = survey_def.sample_size


def _make_rules() -> List[Rule]:
    """Build the complete rule set."""
    return [
        # Rule 1: Foundational — always run descriptive stats
        Rule(
            rule_id=1, name="基础描述统计",
            condition=lambda ctx: True,
            methods=["descriptive"],
            priority_boost=0,
            note="所有问卷都应进行描述统计",
        ),
        # Rule 2: Likert scale → reliability + factor analysis
        Rule(
            rule_id=2, name="量表信效度",
            condition=lambda ctx: ctx.n_likert >= 3,
            methods=["reliability", "factor_analysis"],
            priority_boost=10,
            note="检测到多个Likert量表题，建议信效度检验",
        ),
        # Rule 3: Categorical + continuous → T-test / ANOVA
        Rule(
            rule_id=3, name="群体差异比较",
            condition=lambda ctx: ctx.n_categorical >= 1 and ctx.n_numeric >= 1,
            methods=["ttest_anova"],
            priority_boost=5,
            note="存在分组变量和连续变量，可进行差异比较",
        ),
        # Rule 4: Multiple continuous → correlation
        Rule(
            rule_id=4, name="变量相关关系",
            condition=lambda ctx: ctx.n_numeric >= 2,
            methods=["correlation"],
            priority_boost=0,
            note="多个连续变量可进行相关分析",
        ),
        # Rule 5: Multiple continuous with target → regression
        Rule(
            rule_id=5, name="影响因素回归",
            condition=lambda ctx: ctx.n_numeric >= 3,
            methods=["regression"],
            priority_boost=0,
            note="多个连续变量可建立回归模型",
        ),
        # Rule 6: Clustering
        Rule(
            rule_id=6, name="用户聚类分群",
            condition=lambda ctx: ctx.n_numeric >= 4,
            methods=["clustering"],
            priority_boost=-5,
            note="较多连续变量可尝试聚类分析",
        ),
        # Rule 7: Categorical × categorical → chi-square + correspondence
        Rule(
            rule_id=7, name="分类变量关联",
            condition=lambda ctx: ctx.n_categorical >= 2,
            methods=["chi_square", "correspondence"],
            priority_boost=0,
            note="分类变量之间可检验关联性",
        ),
        # Rule 8: Ordered sequence → runs test
        Rule(
            rule_id=8, name="序列随机性",
            condition=lambda ctx: ctx.n_numeric >= 1 and ctx.sample_size >= 30,
            methods=["runs_test"],
            priority_boost=-20,
            note="可检验数据序列的随机性",
        ),
        # Rule 9: NPS
        Rule(
            rule_id=9, name="净推荐值NPS",
            condition=lambda ctx: ctx.n_nps >= 1,
            methods=["nps"],
            priority_boost=15,
            note="检测到NPS评分题",
        ),
        # Rule 10: Text → sentiment + LDA + TF-IDF
        Rule(
            rule_id=10, name="文本挖掘",
            condition=lambda ctx: ctx.n_text >= 1,
            methods=["sentiment", "lda_topic", "tfidf"],
            priority_boost=10,
            note="存在开放文本题，可进行文本挖掘",
        ),
        # Rule 11: Text → SO-PMI (expert level)
        Rule(
            rule_id=11, name="领域情感词典",
            condition=lambda ctx: ctx.n_text >= 3,
            methods=["so_pmi"],
            priority_boost=-5,
            note="较多文本可构建领域情感词典",
        ),
        # Rule 12: Text → SDGs mapping
        Rule(
            rule_id=12, name="SDGs议题映射",
            condition=lambda ctx: ctx.n_text >= 1,
            methods=["sdgs"],
            priority_boost=-10,
            note="文本可映射到SDGs目标",
        ),
        # Rule 13: Likert multi-dim → fuzzy evaluation
        Rule(
            rule_id=13, name="模糊综合评判",
            condition=lambda ctx: ctx.n_likert >= 4,
            methods=["fuzzy_eval"],
            priority_boost=-5,
            note="多维Likert量表可进行模糊综合评判",
        ),
        # Rule 14: Likert + regression → CCSI/ACSI
        Rule(
            rule_id=14, name="满意度指数模型",
            condition=lambda ctx: ctx.n_likert >= 5,
            methods=["ccsi_acsi"],
            priority_boost=-10,
            note="较多Likert量表可构建满意度指数模型",
        ),
        # Rule 15: Categorical + Likert → fsQCA
        Rule(
            rule_id=15, name="模糊集定性比较",
            condition=lambda ctx: ctx.n_categorical >= 1 and ctx.n_likert >= 3,
            methods=["fsqca"],
            priority_boost=-15,
            note="分类变量+量表数据可进行fsQCA分析",
        ),
    ]


RULES: List[Rule] = _make_rules()


def get_method_info(method_id: str) -> Optional[MethodInfo]:
    """Get the MethodInfo for a given method ID."""
    return METHODS.get(method_id)


def get_all_methods() -> Dict[str, MethodInfo]:
    """Get all registered methods."""
    return METHODS


def get_rules() -> List[Rule]:
    """Get all rules."""
    return RULES
