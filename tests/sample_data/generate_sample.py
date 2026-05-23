"""Generate sample survey data for testing the Survey Auto Analyzer.

Creates a realistic Chinese customer satisfaction survey with:
- Demographics (gender, age, education)
- Likert scales (product satisfaction, service satisfaction, price satisfaction)
- Single/multi choice (purchase channel, usage frequency)
- NPS question
- Open-ended text feedback
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_data(n_samples: int = 500, output_dir: str = None) -> str:
    """Generate sample survey data and save to CSV & Excel.

    Returns the file path of the saved CSV.
    """
    np.random.seed(42)

    data = {}

    # ── Demographics ──
    data["1.您的性别"] = np.random.choice(["男", "女"], n_samples, p=[0.48, 0.52])
    data["2.您的年龄段"] = np.random.choice(
        ["18-25岁", "26-35岁", "36-45岁", "46-55岁", "55岁以上"],
        n_samples, p=[0.15, 0.35, 0.30, 0.15, 0.05],
    )
    data["3.您的学历"] = np.random.choice(
        ["高中及以下", "大专", "本科", "硕士及以上"],
        n_samples, p=[0.15, 0.25, 0.45, 0.15],
    )

    # ── Usage behavior ──
    data["4.您使用本产品的时间"] = np.random.choice(
        ["少于1个月", "1-6个月", "6-12个月", "1-2年", "2年以上"],
        n_samples, p=[0.10, 0.20, 0.30, 0.25, 0.15],
    )
    data["5.您主要通过什么渠道了解本产品？（多选_朋友推荐）"] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    data["5.您主要通过什么渠道了解本产品？（多选_社交媒体）"] = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    data["5.您主要通过什么渠道了解本产品？（多选_搜索引擎）"] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    data["5.您主要通过什么渠道了解本产品？（多选_线下活动）"] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])

    # ── Likert: Product Quality (5 items) ──
    base_quality = np.random.normal(3.8, 1.0, n_samples).clip(1, 5)
    data["6.产品质量-整体满意度"] = np.round(base_quality + np.random.normal(0, 0.3, n_samples)).clip(1, 5).astype(int)
    data["7.产品质量-功能完善度"] = np.round(base_quality + np.random.normal(0, 0.4, n_samples)).clip(1, 5).astype(int)
    data["8.产品质量-使用稳定性"] = np.round(base_quality + np.random.normal(0, 0.5, n_samples)).clip(1, 5).astype(int)
    data["9.产品质量-界面设计"] = np.round(base_quality + np.random.normal(0.2, 0.4, n_samples)).clip(1, 5).astype(int)

    # ── Likert: Service Satisfaction (4 items) ──
    base_service = np.random.normal(3.5, 1.0, n_samples).clip(1, 5)
    data["10.服务质量-客服响应速度"] = np.round(base_service + np.random.normal(0, 0.3, n_samples)).clip(1, 5).astype(int)
    data["11.服务质量-问题解决能力"] = np.round(base_service + np.random.normal(-0.1, 0.4, n_samples)).clip(1, 5).astype(int)
    data["12.服务质量-服务态度"] = np.round(base_service + np.random.normal(0.3, 0.3, n_samples)).clip(1, 5).astype(int)
    data["13.服务质量-售后跟进"] = np.round(base_service + np.random.normal(-0.2, 0.5, n_samples)).clip(1, 5).astype(int)

    # ── Likert: Price Perception (3 items) ──
    base_price = np.random.normal(3.2, 1.0, n_samples).clip(1, 5)
    data["14.价格感知-价格合理性"] = np.round(base_price + np.random.normal(0, 0.3, n_samples)).clip(1, 5).astype(int)
    data["15.价格感知-性价比评价"] = np.round(base_price + np.random.normal(0.1, 0.3, n_samples)).clip(1, 5).astype(int)
    data["16.价格感知-与竞品相比"] = np.round(base_price + np.random.normal(-0.1, 0.4, n_samples)).clip(1, 5).astype(int)

    # ── NPS ──
    # Weighted toward promoters for realistic data
    nps_raw = np.random.choice(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        n_samples,
        p=[0.02, 0.03, 0.03, 0.04, 0.05, 0.08, 0.15, 0.20, 0.25, 0.15],
    )
    data["17.您有多大可能向朋友推荐本产品？（0-10分）"] = nps_raw

    # ── Overall satisfaction (DV candidate) ──
    overall = 0.4 * (data["6.产品质量-整体满意度"] - 1) / 4 + \
              0.3 * (data["10.服务质量-客服响应速度"] - 1) / 4 + \
              0.2 * (data["14.价格感知-价格合理性"] - 1) / 4 + \
              0.1 * nps_raw / 10
    overall = np.round(overall * 4 + 1).clip(1, 5).astype(int)
    data["18.总体满意度评价"] = overall

    # ── Open-ended feedback ──
    positive_templates = [
        "产品很好用，{feature}方面特别满意，会继续使用。",
        "整体体验不错，{feature}表现优秀，推荐给朋友。",
        "已经用了{time}了，{feature}越来越好了。",
        "功能很全，{feature}是我最常用的，希望保持。",
    ]
    negative_templates = [
        "价格有点贵，{feature}还需要改进。",
        "客服响应太慢，{feature}的问题一直没解决。",
        "界面设计不够友好，{feature}找起来很麻烦。",
    ]
    neutral_templates = [
        "还行吧，{feature}还有提升空间。",
        "一般般，没什么特别的感受。",
    ]

    features = [
        "搜索功能", "界面设计", "响应速度", "客户服务",
        "价格设置", "操作便捷性", "数据安全", "更新频率",
    ]
    times = ["一个月", "三个月", "半年", "一年"]

    feedbacks = []
    for i in range(n_samples):
        score = nps_raw[i]
        if score >= 9:
            template = np.random.choice(positive_templates)
        elif score >= 7:
            template = np.random.choice(positive_templates + neutral_templates)
        elif score >= 5:
            template = np.random.choice(neutral_templates + negative_templates)
        else:
            template = np.random.choice(negative_templates)

        text = template.format(
            feature=np.random.choice(features),
            time=np.random.choice(times),
        )
        feedbacks.append(text)

    data["19.请留下您的宝贵建议（开放题）"] = feedbacks

    # ── Build DataFrame ──
    df = pd.DataFrame(data)

    # Add some missing values (realistic)
    for col in df.columns:
        mask = np.random.random(n_samples) < 0.03  # 3% missing
        df.loc[mask, col] = np.nan

    # ── Save ──
    if output_dir is None:
        output_dir = Path(__file__).parent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "sample_survey_data.csv"
    xlsx_path = output_dir / "sample_survey_data.xlsx"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)

    print(f"Sample data generated: {len(df)} rows × {len(df.columns)} columns")
    print(f"  CSV: {csv_path}")
    print(f"  Excel: {xlsx_path}")

    return str(csv_path)


if __name__ == "__main__":
    generate_sample_data()
