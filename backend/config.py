"""
AI提示词管家 — 模型能力配置库 v2
包含5个主流AI模型的多维度能力评估
"""

import os

# Gemini 配置（Google AI Studio）
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# 兼容旧命名，避免其他模块改动过大
LLM_MODEL = GEMINI_MODEL

# ============================================================
# 模型能力数据库
# scores: 1-10 分制
# cost: 1=最便宜 10=最贵 (相对成本)
# speed: 1=最慢 10=最快 (响应速度)
# context_window: 最大上下文窗口 (tokens)
# ============================================================

MODELS = {
    "GPT-4o": {
        "name": "GPT-4o",
        "provider": "OpenAI",
        "icon": "🤖",
        "color": "#10a37f",
        "description": "OpenAI旗舰多模态模型，综合能力最强，尤其擅长创意写作、复杂指令遵循和多模态理解",
        "scores": {
            "writing": 9,
            "coding": 9,
            "academic": 8,
            "business": 9,
            "search": 7,
            "reasoning": 9,
        },
        "cost": 7,
        "speed": 8,
        "context_window": 128000,
        "strengths": ["创意写作", "多轮对话", "指令遵循", "多模态理解", "插件生态"],
        "weaknesses": ["成本较高", "偶有幻觉", "中文能力略逊于国产模型"],
        "best_for": ["创意内容生成", "产品文案", "多轮复杂对话", "图文理解"],
        "prompt_tips": "GPT-4o偏好清晰的角色设定和结构化输出要求。用Markdown格式组织提示效果最佳，明确告知输出格式和风格。",
        "prompt_style": "structured",
    },
    "Claude": {
        "name": "Claude",
        "provider": "Anthropic",
        "icon": "🧠",
        "color": "#d97706",
        "description": "Anthropic旗舰模型，在长文本分析、严谨推理、代码生成和学术写作方面表现卓越",
        "scores": {
            "writing": 9,
            "coding": 10,
            "academic": 10,
            "business": 9,
            "search": 6,
            "reasoning": 10,
        },
        "cost": 8,
        "speed": 7,
        "context_window": 200000,
        "strengths": ["超长文本处理", "代码生成", "严谨推理", "学术分析", "安全性"],
        "weaknesses": ["无实时搜索", "图像能力有限", "有时过于谨慎"],
        "best_for": ["长文档分析", "复杂编程任务", "学术论文", "逻辑推理", "法律文本"],
        "prompt_tips": "Claude偏好详细的上下文和背景说明。给予充分背景信息效果更好，支持XML标签来组织复杂提示，擅长处理超长输入。",
        "prompt_style": "detailed",
    },
    "Gemini": {
        "name": "Gemini",
        "provider": "Google",
        "icon": "💎",
        "color": "#4285f4",
        "description": "Google旗舰模型，原生支持实时搜索和多模态，擅长信息整合和跨语言任务",
        "scores": {
            "writing": 8,
            "coding": 8,
            "academic": 8,
            "business": 8,
            "search": 10,
            "reasoning": 8,
        },
        "cost": 5,
        "speed": 9,
        "context_window": 1000000,
        "strengths": ["实时搜索", "多模态", "超大上下文", "多语言", "Google生态集成"],
        "weaknesses": ["创意写作略弱", "中文风格偏翻译腔", "输出稳定性有波动"],
        "best_for": ["实时信息查询", "多语言翻译", "视频/图片理解", "大量文档处理"],
        "prompt_tips": "Gemini偏好简洁直接的提示。善用其搜索能力获取最新信息，适合需要事实核查和实时数据的任务。",
        "prompt_style": "concise",
    },
    "DeepSeek": {
        "name": "DeepSeek",
        "provider": "DeepSeek",
        "icon": "🔍",
        "color": "#6366f1",
        "description": "国产高性能开源模型，在深度推理、数学和编程方面表现出色，性价比极高",
        "scores": {
            "writing": 7,
            "coding": 10,
            "academic": 9,
            "business": 7,
            "search": 6,
            "reasoning": 10,
        },
        "cost": 2,
        "speed": 8,
        "context_window": 128000,
        "strengths": ["深度推理", "数学计算", "代码生成", "性价比极高", "开源可部署"],
        "weaknesses": ["创意写作风格较朴素", "英文优于中文", "多模态能力有限"],
        "best_for": ["数学证明", "算法设计", "代码重构", "逻辑分析", "科学计算"],
        "prompt_tips": "DeepSeek偏好分步骤推理。明确要求'请一步一步思考'效果最佳，适合复杂逻辑和数学推导任务。",
        "prompt_style": "step_by_step",
    },
    "Perplexity": {
        "name": "Perplexity",
        "provider": "Perplexity AI",
        "icon": "🌐",
        "color": "#20b2aa",
        "description": "专注搜索增强的AI引擎，每个回答都附带实时来源引用，适合需要最新信息和事实核查的任务",
        "scores": {
            "writing": 6,
            "coding": 6,
            "academic": 7,
            "business": 8,
            "search": 10,
            "reasoning": 7,
        },
        "cost": 4,
        "speed": 7,
        "context_window": 32000,
        "strengths": ["实时搜索引用", "来源透明", "事实核查", "新闻追踪", "市场数据"],
        "weaknesses": ["创作能力弱", "编程能力有限", "长文本生成不足"],
        "best_for": ["实时资讯查询", "竞品调研", "行业报告数据", "学术文献搜索", "事实验证"],
        "prompt_tips": "Perplexity擅长检索式任务。明确告知需要最新数据、来源引用，效果最好。适合需要引用和出处的研究类任务。",
        "prompt_style": "research",
    },
}

# ============================================================
# 任务类型定义
# ============================================================

TASK_TYPES = {
    "writing": {
        "name": "写作",
        "icon": "✍️",
        "description": "文案、文章、故事、营销内容等创作类任务",
        "examples": ["小红书文案", "公众号文章", "品牌故事", "广告标题"],
    },
    "coding": {
        "name": "编程",
        "icon": "💻",
        "description": "代码编写、调试、架构设计等技术任务",
        "examples": ["Python脚本", "API开发", "bug修复", "代码重构"],
    },
    "academic": {
        "name": "学术",
        "icon": "📚",
        "description": "论文、研究、文献综述等学术类任务",
        "examples": ["文献综述", "论文大纲", "研究方法", "数据分析"],
    },
    "business": {
        "name": "商业",
        "icon": "📊",
        "description": "商业分析、市场调研、商业计划等商业类任务",
        "examples": ["商业模式分析", "市场调研", "竞品分析", "BP撰写"],
    },
    "search": {
        "name": "搜索",
        "icon": "🔎",
        "description": "信息检索、事实查询、新闻资讯等搜索类任务",
        "examples": ["最新新闻", "产品对比", "价格查询", "事实核查"],
    },
    "reasoning": {
        "name": "推理",
        "icon": "🧩",
        "description": "逻辑推理、数学计算、问题分析等推理类任务",
        "examples": ["数学证明", "逻辑分析", "因果推理", "决策分析"],
    },
}

# ============================================================
# 复杂度定义（用于推荐算法加权）
# ============================================================

COMPLEXITY_WEIGHTS = {
    "low": {"cost_weight": 0.3, "ability_weight": 0.5, "speed_weight": 0.2},
    "medium": {"cost_weight": 0.2, "ability_weight": 0.6, "speed_weight": 0.2},
    "high": {"cost_weight": 0.1, "ability_weight": 0.8, "speed_weight": 0.1},
}
