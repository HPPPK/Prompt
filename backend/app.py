"""
AI提示词管家 — Flask 后端 v2
"""

import logging
import re
import time
from collections import deque

from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

# 加载 backend/.env 中的环境变量（如 GEMINI_API_KEY）
load_dotenv()

import llm_client
from classifier import classify_task
from prompt_generator import generate_prompt
from recommender import recommend_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 内存历史记录（简单实现，重启后丢失）
history: deque[dict] = deque(maxlen=50)
MODEL_ONLY_ALIASES = {
    "claude", "anthropic",
    "gpt", "chatgpt", "openai", "gpt4", "gpt4o",
    "gemini", "google",
    "deepseek",
    "perplexity",
    "qwen", "llama", "mistral",
}


def _validate_user_input(user_input: str) -> str | None:
    """校验是否是可执行任务输入。返回错误文案或 None。"""
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", user_input.lower()).strip()
    if not normalized:
        return "请输入您的需求"

    compact = normalized.replace(" ", "")
    tokens = normalized.split()

    # 仅输入模型名（或品牌名）时，不进行推荐和提示词生成
    if compact in MODEL_ONLY_ALIASES:
        return (
            "你输入的是模型名，不是任务需求。"
            "请改为：你想让AI完成什么任务，例如“用Claude写一封求职邮件”。"
        )

    # 过短且缺少任务语义，提示用户补全需求
    if len(tokens) <= 1 and len(compact) <= 3:
        return "请描述具体任务，例如“帮我写一篇小红书文案”或“分析特斯拉商业模式”。"

    return None


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """分析用户需求，返回模型推荐和提示词"""
    data = request.get_json()
    user_input = data.get("input", "").strip()

    if not user_input:
        return jsonify({"error": "请输入您的需求"}), 400
    validation_error = _validate_user_input(user_input)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    start_time = time.time()

    # 1. 智能任务分类
    classification = classify_task(user_input)

    # 2. 多维度模型推荐
    recommendations = recommend_models(classification)

    # 3. 为每个推荐模型生成专属提示词
    results = []
    for rec in recommendations:
        prompt = generate_prompt(user_input, rec, classification)
        results.append({
            "model": {
                "name": rec["name"],
                "provider": rec["provider"],
                "icon": rec["icon"],
                "color": rec["color"],
                "description": rec["description"],
                "strengths": rec["strengths"],
                "weaknesses": rec["weaknesses"],
                "best_for": rec["best_for"],
                "prompt_tips": rec["prompt_tips"],
                "match_pct": rec["match_pct"],
                "scores": rec["scores"],
                "cost": rec["cost"],
                "speed": rec["speed"],
                "context_window": rec["context_window"],
            },
            "reason": rec["reason"],
            "prompt": prompt,
        })

    elapsed = round(time.time() - start_time, 2)

    response_data = {
        "input": user_input,
        "classification": {
            "task_types": classification["task_types"],
            "complexity": classification.get("complexity", "medium"),
            "intent": classification.get("intent", user_input),
            "key_entities": classification.get("key_entities", []),
            "source": classification.get("source", "unknown"),
        },
        "recommendations": results,
        "meta": {
            "elapsed_seconds": elapsed,
            "llm_available": llm_client.is_available(),
        },
    }

    # 保存到历史
    history.appendleft({
        "id": int(time.time() * 1000),
        "input": user_input,
        "classification": response_data["classification"],
        "model_names": [r["model"]["name"] for r in results],
        "timestamp": time.time(),
    })

    return jsonify(response_data)


@app.route("/api/history", methods=["GET"])
def get_history():
    """获取历史记录"""
    return jsonify({"history": list(history)})


@app.route("/api/health", methods=["GET"])
def health():
    """健康检查（含 LLM 状态）"""
    ollama_ok = llm_client.check_ollama()
    return jsonify({
        "status": "ok",
        "ollama_available": ollama_ok,
        "ollama_model": llm_client.OLLAMA_MODEL if ollama_ok else None,
    })


if __name__ == "__main__":
    # 启动时检测 Gemini
    if llm_client.check_ollama():
        logger.info("Gemini 可用，将使用 LLM 进行智能分类和提示词生成")
    else:
        logger.warning("Gemini 不可用，将使用关键词匹配和模板生成（降级模式）")

    app.run(debug=True, port=5001)
