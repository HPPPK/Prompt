"""
Gemini LLM 客户端封装
使用 Google AI Studio API（REST）
自动检测可用性，不可用时降级到关键词匹配
"""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from config import GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL

logger = logging.getLogger(__name__)

_llm_available: bool | None = None  # 缓存可用性检测结果

# 导出给其他模块用的别名
OLLAMA_MODEL = GEMINI_MODEL


def check_ollama() -> bool:
    """兼容旧函数名：检测 Gemini 服务是否可用"""
    return check_gemini()


def check_gemini() -> bool:
    """检测 Gemini API 是否可用"""
    global _llm_available
    if not GEMINI_API_KEY:
        _llm_available = False
        return _llm_available

    try:
        url = _build_url("/v1beta/models", {"key": GEMINI_API_KEY})
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            _llm_available = resp.status == 200
    except Exception:
        _llm_available = False
    return _llm_available


def is_available() -> bool:
    """返回 Gemini 是否可用（带缓存）"""
    if _llm_available is None:
        return check_gemini()
    return _llm_available


def reset_cache():
    """重置可用性缓存，强制下次重新检测"""
    global _llm_available
    _llm_available = None


def chat(prompt: str, system_prompt: str = "", model: str = "") -> str:
    """
    调用 Gemini 生成回复（REST API）。

    Args:
        prompt: 用户提示
        system_prompt: 系统提示（可选）
        model: 模型名称（默认使用 config 中配置的模型）

    Returns:
        LLM 生成的文本

    Raises:
        RuntimeError: Gemini 不可用或调用失败
    """
    if not is_available():
        raise RuntimeError("Gemini 服务不可用（请检查 GEMINI_API_KEY）")

    model = (model or GEMINI_MODEL).replace("models/", "")

    payload_dict = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2048,
        },
    }

    if system_prompt:
        payload_dict["system_instruction"] = {
            "parts": [{"text": system_prompt}]
        }

    payload = json.dumps(payload_dict).encode("utf-8")

    url = _build_url(
        f"/v1beta/models/{model}:generateContent",
        {"key": GEMINI_API_KEY},
    )
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return _extract_text(result)
    except urllib.error.HTTPError as e:
        _mark_unavailable()
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            detail = str(e)
        raise RuntimeError(f"Gemini 调用失败: {e.code} {detail}") from e
    except urllib.error.URLError as e:
        _mark_unavailable()
        raise RuntimeError(f"Gemini 调用失败: {e}") from e
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Gemini 响应解析失败: {e}") from e


def chat_json(prompt: str, system_prompt: str = "", model: str = "") -> dict:
    """
    调用 Gemini 并解析 JSON 响应。

    自动处理 LLM 输出中的 markdown 代码块标记。

    Returns:
        解析后的 dict

    Raises:
        RuntimeError: 调用失败或 JSON 解析失败
    """
    raw = chat(prompt, system_prompt, model)

    # 清理常见的 markdown 代码块标记
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试从文本中提取 JSON 对象
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"无法从 LLM 输出中解析 JSON:\n{raw[:500]}")


def _mark_unavailable():
    """标记 Gemini 为不可用"""
    global _llm_available
    _llm_available = False


def _build_url(path: str, params: dict[str, str]) -> str:
    query = urllib.parse.urlencode(params)
    return f"{GEMINI_BASE_URL}{path}?{query}"


def _extract_text(result: dict) -> str:
    candidates = result.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini 未返回候选内容: {result}")

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
    if not texts:
        raise RuntimeError(f"Gemini 返回内容为空: {result}")
    return "\n".join(texts).strip()
