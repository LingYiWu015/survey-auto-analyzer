"""Prompt templates for DeepSeek LLM.

These prompts are designed to:
1. Understand question semantics (what is this question really asking?)
2. Generate research hypotheses
3. Suggest additional analysis methods beyond the rule engine
"""

# ─── Main Semantic Analysis Prompt ───────────────────────────────────

SEMANTIC_ANALYSIS_PROMPT = """你是一位资深的市场调查与数据分析专家。请分析以下问卷题目，完成三项任务。

## 问卷题目列表
{questions}

## 任务要求

请严格按照以下JSON格式输出（不要输出其他内容）：

```json
{{
  "questions": [
    {{
      "index": 1,
      "semantic_category": "满意度|忠诚度|质量感知|价值感知|期望|行为意向|人口统计|开放反馈|其他",
      "suggested_role": "因变量|自变量|中介变量|调节变量|分组变量|控制变量|仅描述",
      "dimension": "所属维度名称（如：产品满意度、服务满意度、品牌形象等）",
      "notes": "简要说明（10字以内）"
    }}
  ],
  "hypotheses": [
    {{
      "text": "研究假设文字表述（如：感知质量正向影响顾客满意度）",
      "variables_involved": ["变量1", "变量2"],
      "recommended_method": "推荐的分析方法ID（如：regression, correlation, ttest_anova 等）",
      "confidence": 0.85
    }}
  ],
  "method_suggestions": [
    {{
      "method_id": "方法ID（如：regression, clustering, sentiment 等）",
      "reason": "推荐理由（30字以内）",
      "extra_score": 15
    }}
  ]
}}
```

## 方法ID对照表
- reliability: 信效度检验
- descriptive: 描述统计
- ttest_anova: T检验与方差分析
- correlation: 相关分析
- regression: 回归分析
- factor_analysis: 因子分析
- clustering: 聚类分析
- correspondence: 对应分析
- chi_square: 卡方检验
- runs_test: 游程检验
- nps: NPS净推荐值
- sentiment: 情感分析
- lda_topic: LDA主题模型
- tfidf: TF-IDF关键词提取
- so_pmi: SO-PMI情感词典
- sdgs: SDGs映射
- fuzzy_eval: 模糊综合评判
- ccsi_acsi: CCSI/ACSI满意度模型
- fsqca: fsQCA定性比较分析

## 注意事项
1. semantic_category 要根据题目文字语义判断，不能仅看题型
2. 如果题目明显是多维度的（如满意度包含产品、服务、价格等），请正确归入维度
3. 研究假设要有实际意义，避免"变量A与变量B相关"这种空洞假设
4. confidence 表示你对这个假设的确信程度（0-1）
5. 方法推荐要结合题型和语义双重判断
6. extra_score 范围为 0-25 分，表示AI对该方法的额外推荐程度
"""


# ─── Draft Mode Summary Prompt ───────────────────────────────────────

DRAFT_SUMMARY_PROMPT = """你是一位资深的数据分析报告撰写专家。请基于以下信息，生成一份简洁的问卷分析草稿。

## 问卷概况
- 样本量: {sample_size}
- 题目数: {n_questions}
- 题目类型分布: {var_types}

## 推荐的分析方法（按优先级排序）
{recommended_methods}

## 研究假设（AI生成）
{hypotheses}

## 任务
请用中文生成一份简洁的分析草稿（500-800字），包含：
1. 问卷概况总结（1-2句）
2. 推荐的分析方案（按优先级列举3-5个核心方法）
3. 可能的研究发现预测（基于假设，2-3点）
4. 分析注意事项（1-2条）

请直接输出Markdown格式的报告草稿，不要输出JSON。
"""


# ─── Result Interpretation Prompt ─────────────────────────────────────

INTERPRETATION_PROMPT = """你是一位资深的数据分析解读专家。请解读以下分析结果。

## 分析方法
{method_name}：{method_description}

## 分析结果
{result_summary}

## 任务
请用通俗易懂的中文（3-5句话）解读上述结果，包含：
1. 这个分析做了什么？
2. 关键发现是什么？
3. 对决策者有什么启示？

请直接输出解读文字，不要输出JSON。
"""
