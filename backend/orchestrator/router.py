from __future__ import annotations

import llm_client
from classifier import classify_task

from .code_handler import CodeTaskHandler
from .email_handler import EmailTaskHandler
from .generic_handler import GenericTaskHandler
from .ml_extractor import predict_task_type
from .writing_handler import WritingTaskHandler


EMAIL_HANDLER = EmailTaskHandler()
CODE_HANDLER = CodeTaskHandler()
WRITING_HANDLER = WritingTaskHandler()
GENERIC_HANDLER = GenericTaskHandler()
HANDLERS = [EMAIL_HANDLER, CODE_HANDLER, WRITING_HANDLER, GENERIC_HANDLER]

ROUTER_SYSTEM_PROMPT = """你是任务路由器。请将用户请求路由到以下类型之一：
- email: 写邮件、回复邮件、催办邮件、商务邮件
- code: 代码实现、修复 bug、重构、工程改动
- writing: 一般写作/文案/内容创作（非邮件）
- generic: 无法明确归类但需要继续澄清的任务
- other: 明确不走编排（兼容保留）

规则：
1) 信息不足时优先选 generic，不要轻易选 other。
2) 只有明确不适合编排时才选 other。

严格返回 JSON:
{
  "task_type": "email|code|writing|generic|other",
  "confidence": 0.0 到 1.0,
  "reason": "一句简短理由"
}
不要输出 JSON 以外的内容。"""


def route_task(text: str):
    # 优先尝试本地小模型（若已训练并可用），降低对在线 API 依赖。
    ml_pred = predict_task_type(text)
    if ml_pred and ml_pred.confidence >= 0.62:
        handler = get_handler(ml_pred.task_type)
        if handler:
            return ml_pred.task_type, handler, ml_pred.confidence

    # 先拿到现有分类器结果，作为上下文与 fallback。
    cls = classify_task(text)

    # 主路径：LLM 语义路由。
    llm_route = _route_with_llm(text, cls)
    if llm_route:
        task_type, conf = llm_route
        handler = get_handler(task_type)
        if handler:
            return task_type, handler, conf
        return "other", None, conf

    # 降级路径：使用分类器 + 少量规则映射。
    task_types = cls.get("task_types") or []
    if not task_types:
        return "generic", GENERIC_HANDLER, 0.25

    top = task_types[0]
    t = top.get("type")
    conf = float(top.get("confidence", 0.0) or 0.0)

    if t == "coding":
        if conf >= 0.2:
            return "code", CODE_HANDLER, conf
        return "generic", GENERIC_HANDLER, max(conf, 0.25)

    if t in {"writing", "academic", "business"}:
        # 降级模式下，用 email 专用检测兜底，避免“写邮件”被当普通写作。
        email_conf = EMAIL_HANDLER.detect(text)
        if email_conf >= 0.4:
            return "email", EMAIL_HANDLER, email_conf
        writing_conf = WRITING_HANDLER.detect(text)
        if writing_conf >= 0.35:
            return "writing", WRITING_HANDLER, max(conf, writing_conf)
        return "generic", GENERIC_HANDLER, max(conf, 0.25)

    if t in {"search", "reasoning"}:
        # 对未专门建 handler 的类型，走 generic clarifier 而不是直接回退。
        return "generic", GENERIC_HANDLER, max(conf, 0.25)

    return "generic", GENERIC_HANDLER, max(conf, 0.25)


def get_handler(task_type: str):
    for handler in HANDLERS:
        if handler.task_type == task_type:
            return handler
    return None


def _route_with_llm(text: str, classification: dict):
    if not llm_client.is_available():
        return None

    try:
        result = llm_client.chat_json(
            prompt=(
                "请根据用户请求和分类上下文做路由。\\n\\n"
                f"用户请求：{text}\\n\\n"
                f"分类上下文：{classification}"
            ),
            system_prompt=ROUTER_SYSTEM_PROMPT,
        )

        task_type = str(result.get("task_type", "")).strip().lower()
        confidence = float(result.get("confidence", 0.0) or 0.0)
        confidence = max(0.0, min(1.0, confidence))

        if task_type not in {"email", "code", "writing", "generic", "other"}:
            return None

        if task_type == "other":
            # 信息不足默认走 generic，除非置信度极低才保留 other 兼容。
            if confidence < 0.1:
                return "other", confidence
            return "generic", max(confidence, 0.3)

        if task_type == "generic":
            return "generic", max(confidence, 0.3)

        # email/code/writing
        if confidence < 0.2:
            return None
        return task_type, confidence
    except Exception:
        return None
