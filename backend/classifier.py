"""
任务分类器 v2
优先使用 Gemini LLM 进行智能分类，不可用时降级到关键词匹配
"""

import logging

import llm_client
from config import TASK_TYPES

logger = logging.getLogger(__name__)

# ============================================================
# 关键词匹配 Fallback
# ============================================================

KEYWORDS = {
    "writing": [
        "写", "文案", "文章", "故事", "小说", "诗", "剧本", "广告",
        "营销", "推广", "宣传", "标题", "slogan", "文风", "润色",
        "改写", "翻译", "小红书", "公众号", "微博", "朋友圈",
        "邮件", "信", "报告", "总结", "摘要", "大纲", "创作",
        "write", "article", "blog", "copy", "content", "essay",
        "文笔", "段落", "开头", "结尾", "描述", "介绍",
    ],
    "coding": [
        "代码", "编程", "程序", "函数", "bug", "调试", "开发",
        "python", "java", "javascript", "react", "vue", "api",
        "数据库", "sql", "html", "css", "算法", "数据结构",
        "部署", "docker", "git", "前端", "后端", "全栈",
        "code", "debug", "deploy", "script", "自动化", "爬虫",
        "接口", "框架", "library", "sdk", "app", "网站", "web",
    ],
    "academic": [
        "论文", "研究", "文献", "引用", "学术", "期刊",
        "综述", "实验", "假设", "方法论", "摘要", "abstract",
        "研究方向", "课题", "答辩", "毕业", "学位",
        "paper", "research", "thesis", "dissertation",
        "理论", "模型", "框架", "分析方法", "统计",
    ],
    "business": [
        "商业", "市场", "竞品", "战略", "策略",
        "商业模式", "盈利", "收入", "成本", "用户", "客户",
        "增长", "融资", "投资", "估值", "财务", "报表",
        "swot", "pest", "商业计划", "bp", "pitch",
        "品牌", "定位", "营收", "利润", "roi",
        "business", "market", "strategy", "competitive",
        "行业", "赛道", "趋势", "风口",
    ],
    "search": [
        "搜索", "查找", "查询", "最新", "新闻", "热点",
        "是什么", "是谁", "在哪", "怎么样", "多少",
        "排名", "排行", "对比", "比较", "区别",
        "百科", "定义", "概念", "解释",
        "search", "find", "latest", "news", "what is",
        "推荐", "哪个好", "评测", "评价",
    ],
    "reasoning": [
        "推理", "逻辑", "数学", "计算", "证明", "公式",
        "方程", "概率", "统计", "优化",
        "为什么", "原因", "因果", "假设", "推断",
        "math", "logic", "reason", "calculate", "solve",
        "谜题", "puzzle", "智力", "脑筋急转弯",
        "评估", "判断", "决策",
    ],
}

# LLM 分类 System Prompt
CLASSIFIER_SYSTEM_PROMPT = """你是一个任务分类专家。用户会给你一段需求描述，你需要分析这个需求并返回 JSON 格式的分类结果。

可选的任务类型有：
- writing: 写作（文案、文章、故事、营销内容等创作类任务）
- coding: 编程（代码编写、调试、架构设计等技术任务）
- academic: 学术（论文、研究、文献综述等学术类任务）
- business: 商业（商业分析、市场调研、商业计划等商业类任务）
- search: 搜索（信息检索、事实查询、新闻资讯等搜索类任务）
- reasoning: 推理（逻辑推理、数学计算、问题分析等推理类任务）

你必须严格按照以下 JSON 格式返回，不要输出任何额外文本：
{
  "task_types": [
    {"type": "类型标识", "confidence": 0.0到1.0的置信度}
  ],
  "complexity": "low/medium/high",
  "intent": "一句话描述用户的核心意图",
  "key_entities": ["需求中的关键实体"],
  "language": "zh或en"
}

规则：
1. task_types 可以有多个，按置信度从高到低排列
2. complexity 根据任务难度判断：简单查询=low，一般任务=medium，需要深度分析=high
3. intent 用简短一句话概括用户真正想要什么
4. key_entities 提取需求中的关键名词/实体
5. 只返回 JSON，不要有任何其他文字"""


def classify_task(user_input: str) -> dict:
    """
    对用户输入进行智能分类。

    Returns:
        分类结果 dict，包含：
        - task_types: 按置信度排序的类型列表
        - complexity: 任务复杂度 (low/medium/high)
        - intent: 用户核心意图
        - key_entities: 关键实体
        - language: 语言
        - source: "llm" 或 "fallback"
    """
    # 优先使用 LLM 分类
    if llm_client.is_available():
        try:
            result = _classify_with_llm(user_input)
            result["source"] = "llm"
            return result
        except Exception as e:
            logger.warning("LLM 分类失败，降级到关键词匹配: %s", e)

    # Fallback: 关键词匹配
    return _classify_with_keywords(user_input)


def _classify_with_llm(user_input: str) -> dict:
    """使用 Gemini LLM 进行分类"""
    result = llm_client.chat_json(
        prompt=f"请分析以下用户需求：\n\n{user_input}",
        system_prompt=CLASSIFIER_SYSTEM_PROMPT,
    )

    # 验证和修正结果
    valid_types = set(TASK_TYPES.keys())

    if "task_types" not in result or not result["task_types"]:
        raise ValueError("LLM 未返回 task_types")

    # 过滤无效类型
    result["task_types"] = [
        t for t in result["task_types"]
        if t.get("type") in valid_types
    ]

    if not result["task_types"]:
        raise ValueError("LLM 返回的类型全部无效")

    # 确保 confidence 在 0-1 之间
    for t in result["task_types"]:
        t["confidence"] = max(0.0, min(1.0, float(t.get("confidence", 0.5))))

    # 确保 complexity 有效
    if result.get("complexity") not in ("low", "medium", "high"):
        result["complexity"] = "medium"

    # 确保其他字段存在
    result.setdefault("intent", user_input)
    result.setdefault("key_entities", [])
    result.setdefault("language", "zh")

    return result


def _classify_with_keywords(user_input: str) -> dict:
    """关键词匹配 fallback"""
    text = user_input.lower()
    results = []

    for task_type, keywords in KEYWORDS.items():
        score = 0
        matched = []
        for kw in keywords:
            if kw.lower() in text:
                weight = len(kw)
                score += weight
                matched.append(kw)

        if score > 0:
            results.append({
                "type": task_type,
                "confidence": min(score / 20.0, 1.0),  # 归一化到 0-1
                "matched_keywords": matched,
            })

    results.sort(key=lambda x: x["confidence"], reverse=True)

    if not results:
        results = [{"type": "writing", "confidence": 0.3, "matched_keywords": []}]

    # 估算复杂度
    text_len = len(user_input)
    if text_len > 100 or len(results) > 2:
        complexity = "high"
    elif text_len > 30:
        complexity = "medium"
    else:
        complexity = "low"

    return {
        "task_types": results,
        "complexity": complexity,
        "intent": user_input,
        "key_entities": [],
        "language": "zh" if any('\u4e00' <= c <= '\u9fff' for c in user_input) else "en",
        "source": "fallback",
    }
