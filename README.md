# Survey Auto Analyzer — 智能问卷分析系统

上传问卷数据 → 自动识别题型 → AI 匹配分析方法 → 自动生成报告

## 安装

```bash
# 基础安装（核心统计方法）
pip install survey-auto-analyzer

# 含文本分析支持
pip install survey-auto-analyzer[nlp]

# 含结构方程模型支持
pip install survey-auto-analyzer[sem]

# 含可视化增强
pip install survey-auto-analyzer[viz]

# 全量安装
pip install survey-auto-analyzer[all]
```

## 快速开始

### Web 界面

```bash
streamlit run app.py
```

或安装后：
```bash
survey-auto --help
```

### 命令行

```bash
# 草稿模式：仅方法推荐 + AI 假设
survey-auto analyze survey_data.xlsx --draft

# 完整分析
survey-auto analyze survey_data.xlsx -o report.html

# 指定运行哪些方法
survey-auto analyze survey_data.csv --methods reliability,descriptive,nps

# 带 AI 语义分析
survey-auto analyze survey_data.xlsx --api-key sk-your-deepseek-key

# 列出所有可用方法
survey-auto --list-methods
```

## 支持的问卷平台

- 问卷星 (wjx.cn)
- 腾讯问卷 (wj.qq.com)
- Google Forms
- 通用 CSV / Excel

## 分析方法

### 定量分析
- 信效度检验（Cronbach's α + KMO + Bartlett）
- 描述统计与列联表
- T检验与方差分析（ANOVA + 事后比较）
- 相关分析（Pearson / Spearman）
- 回归分析（线性 / Logistic / Lasso）
- 因子分析（主成分法 + Varimax 旋转）
- 聚类分析（K-Means + 肘部法）
- 卡方检验（独立性检验 + Cramer's V）
- 对应分析（CA / MCA 感知图）
- 游程检验（随机性检验）
- NPS 净推荐值

### 文本分析
- 情感分析（SnowNLP）
- LDA 主题模型
- TF-IDF 关键词提取
- SO-PMI 情感词典构建
- SDGs 目标映射

### 综合评价
- 模糊综合评判
- CCSI / ACSI 满意度指数（SEM）
- fsQCA 定性比较分析

## 配置

通过环境变量或 `~/.streamlit/secrets.toml` 配置 DeepSeek API Key：

```toml
[deepseek]
api_key = "sk-your-api-key"
```

## 开发

```bash
git clone <repo>
pip install -e ".[all]"
streamlit run app.py
```

## License

MIT
