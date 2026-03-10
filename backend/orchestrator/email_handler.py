from __future__ import annotations

import re

from .base import TaskHandler
from .prompts import render_email_prompt
from .validator import validate_email_output


PURPOSE_TO_OBJECTIVE = {
    "chase_progress": "Draft a follow-up email to push progress and request immediate action.",
    "request_invoice": "Draft an email requesting the vendor invoice and a clear delivery timeline.",
    "follow_up": "Draft a follow-up email to move the thread forward with clear next steps.",
    "other": "Draft an email that achieves the user objective with clear action items.",
}


class EmailTaskHandler(TaskHandler):
    task_type = "email"

    def detect(self, text: str) -> float:
        lower = text.lower()
        score = 0.0

        # 强信号：明确提到“邮件/email/mail”，直接进入高置信度。
        if any(kw in lower for kw in ("邮件", "email", "e-mail")):
            score += 0.55
        if re.search(r"\bmail\b", lower):
            score += 0.45

        # 常见邮件场景词加权。
        weighted_keywords = {
            "subject": 0.2,
            "收件人": 0.2,
            "发票": 0.24,
            "供应商": 0.22,
            "follow up": 0.22,
            "follow-up": 0.22,
            "vendor": 0.22,
            "invoice": 0.24,
            "催": 0.2,
            "抄送": 0.2,
        }
        for kw, weight in weighted_keywords.items():
            if kw in lower:
                score += weight

        # 结构信号：如“写一封邮件 / draft an email”。
        if ("写" in text and "邮件" in text) or ("draft" in lower and "email" in lower):
            score += 0.25

        return min(score, 1.0)

    def clarify_schema(self, text: str) -> dict:
        return {
            "title": "邮件任务澄清",
            "description": "补充几个关键信息，我会按这些信息生成更准确的邮件草稿和提示词。",
            "fields": [
                {
                    "key": "recipient_type",
                    "label": "收件人类型",
                    "type": "single_choice",
                    "required": True,
                    "default": "vendor",
                    "help_text": "如果对方不在选项里，请选“其他”并填写具体身份。",
                    "options": [
                        {"value": "vendor", "label": "供应商"},
                        {"value": "client", "label": "客户"},
                        {"value": "manager", "label": "上级"},
                        {"value": "colleague", "label": "同事"},
                        {"value": "other", "label": "其他"},
                    ],
                },
                {
                    "key": "recipient_type_other",
                    "label": "请补充收件人类型",
                    "type": "short_text",
                    "required": False,
                    "required_when": {"recipient_type": "other"},
                    "show_when": {"recipient_type": "other"},
                    "placeholder": "例如：渠道合作方、财务联系人、法务顾问",
                },
                {
                    "key": "relationship",
                    "label": "你和收件人的关系",
                    "type": "single_choice",
                    "required": True,
                    "default": "existing",
                    "help_text": "用于控制邮件语气和礼貌程度。",
                    "options": [
                        {"value": "new", "label": "首次沟通"},
                        {"value": "existing", "label": "已有合作"},
                        {"value": "escalation", "label": "催办/升级"},
                    ],
                },
                {
                    "key": "purpose",
                    "label": "这封邮件的主要目的",
                    "type": "single_choice",
                    "required": True,
                    "default": "request_invoice",
                    "help_text": "如果你的目的不在选项里，请选“其他”并补充。",
                    "options": [
                        {"value": "chase_progress", "label": "催进度"},
                        {"value": "request_invoice", "label": "催发票"},
                        {"value": "follow_up", "label": "一般跟进"},
                        {"value": "other", "label": "其他"},
                    ],
                },
                {
                    "key": "purpose_other",
                    "label": "请补充邮件目的",
                    "type": "multiline_text",
                    "required": False,
                    "required_when": {"purpose": "other"},
                    "show_when": {"purpose": "other"},
                    "placeholder": "请写清楚你希望对方做什么、何时完成。",
                },
                {
                    "key": "order_or_po_number",
                    "label": "订单号 / PO号（可选）",
                    "type": "short_text",
                    "required": False,
                    "show_when": {"purpose": "request_invoice"},
                    "placeholder": "例如：PO-2026-018",
                    "help_text": "如果你要催某个具体订单的发票，填这个会更准确。",
                },
                {
                    "key": "invoice_type",
                    "label": "发票类型（可选）",
                    "type": "single_choice",
                    "required": False,
                    "show_when": {"purpose": "request_invoice"},
                    "options": [
                        {"value": "vat_special", "label": "增值税专票"},
                        {"value": "vat_normal", "label": "增值税普票"},
                        {"value": "receipt", "label": "收据"},
                        {"value": "other", "label": "其他"},
                    ],
                },
                {
                    "key": "invoice_type_other",
                    "label": "请补充发票类型",
                    "type": "short_text",
                    "required": False,
                    "required_when": {"invoice_type": "other"},
                    "show_when": {"invoice_type": "other"},
                    "placeholder": "例如：电子发票（服务费）",
                },
                {
                    "key": "current_blocker",
                    "label": "当前卡点（可选）",
                    "type": "multiline_text",
                    "required": False,
                    "show_when": {"purpose": "chase_progress"},
                    "placeholder": "例如：对方未确认交付时间，导致我方排期受阻。",
                },
                {
                    "key": "tone",
                    "label": "语气风格",
                    "type": "single_choice",
                    "required": True,
                    "default": "professional",
                    "help_text": "专业=中性正式；坚定=明确催办；友好=温和礼貌。",
                    "options": [
                        {"value": "professional", "label": "专业"},
                        {"value": "firm", "label": "坚定"},
                        {"value": "friendly", "label": "友好"},
                    ],
                },
                {
                    "key": "language",
                    "label": "邮件语言",
                    "type": "single_choice",
                    "required": True,
                    "default": "en",
                    "options": [
                        {"value": "en", "label": "英文"},
                        {"value": "zh", "label": "中文"},
                    ],
                },
                {
                    "key": "word_limit",
                    "label": "篇幅上限",
                    "type": "number",
                    "required": True,
                    "default": 200,
                    "min": 50,
                    "max": 500,
                    "help_text": "建议 120-220。数字越小，内容越简洁。",
                },
                {
                    "key": "include_deadline",
                    "label": "是否要写明确截止时间",
                    "type": "boolean",
                    "required": True,
                    "default": False,
                    "help_text": "“截止时间”就是具体日期/时间，例如“请于 3 月 8 日前回复”。",
                    "true_label": "需要",
                    "false_label": "不需要",
                },
                {
                    "key": "deadline_text",
                    "label": "请写明截止时间",
                    "type": "short_text",
                    "required": False,
                    "required_when": {"include_deadline": True},
                    "show_when": {"include_deadline": True},
                    "placeholder": "例如：请在 3 月 8 日 18:00 前回复",
                },
                {
                    "key": "include_bullets",
                    "label": "是否需要项目符号清单",
                    "type": "boolean",
                    "required": True,
                    "default": False,
                    "help_text": "例如列出待办项：- 发票编号 - 开票日期 - 回传时间。",
                    "true_label": "需要",
                    "false_label": "不需要",
                },
                {
                    "key": "bullet_focus",
                    "label": "希望清单里列什么（可选）",
                    "type": "short_text",
                    "required": False,
                    "show_when": {"include_bullets": True},
                    "placeholder": "例如：开票信息、寄送时间、回传节点",
                },
                {
                    "key": "must_include",
                    "label": "邮件中必须提到的信息（可选）",
                    "type": "multiline_text",
                    "required": False,
                    "help_text": "一行一条。只写关键点，不用写完整句子。",
                    "placeholder": "例如：\nPO 编号\n发票抬头\n希望回复时间",
                },
                {
                    "key": "must_avoid",
                    "label": "邮件中避免出现的内容（可选）",
                    "type": "multiline_text",
                    "required": False,
                    "help_text": "例如“不要太强硬”“不要提价格争议”。",
                    "placeholder": "例如：\n不要指责对方\n不要提内部流程细节",
                },
                {
                    "key": "background",
                    "label": "背景信息（必填）",
                    "type": "multiline_text",
                    "required": True,
                    "help_text": "写清楚来龙去脉：发生了什么、当前卡点、你希望对方下一步做什么。",
                    "placeholder": "例如：PO-2026-018 已交付，但我们还没收到发票，已影响本月对账。",
                },
            ],
        }

    def build_spec(self, text: str, answers: dict) -> dict:
        clarified_request = (answers.get("clarified_request") or "").strip()
        success_criteria = _lines_to_list(answers.get("success_criteria", ""))
        hard_constraints = _lines_to_list(answers.get("hard_constraints", ""))
        output_preference = answers.get("output_preference", "direct")
        intent_frame = _build_intent_frame(answers)

        purpose = answers.get("purpose", "other")
        custom_purpose = (answers.get("purpose_other") or "").strip()
        custom_recipient = (answers.get("recipient_type_other") or "").strip()
        background = (answers.get("background") or "").strip()
        order_or_po_number = (answers.get("order_or_po_number") or "").strip()
        current_blocker = (answers.get("current_blocker") or "").strip()
        deadline_text = (answers.get("deadline_text") or "").strip()
        bullet_focus = (answers.get("bullet_focus") or "").strip()
        invoice_type = answers.get("invoice_type")
        invoice_type_other = (answers.get("invoice_type_other") or "").strip()

        must_include = _lines_to_list(answers.get("must_include", ""))
        if order_or_po_number:
            must_include.append(f"订单号/PO号：{order_or_po_number}")
        if deadline_text:
            must_include.append(f"明确截止时间：{deadline_text}")
        if bullet_focus:
            must_include.append(f"清单重点：{bullet_focus}")
        if invoice_type:
            final_invoice_type = invoice_type_other if invoice_type == "other" else invoice_type
            if final_invoice_type:
                must_include.append(f"发票类型：{final_invoice_type}")
        if current_blocker:
            must_include.append(f"当前卡点：{current_blocker}")

        must_avoid = _lines_to_list(answers.get("must_avoid", ""))

        include_deadline = bool(answers.get("include_deadline", True))
        include_bullets = bool(answers.get("include_bullets", True))
        word_limit = int(answers.get("word_limit", 200) or 200)

        acceptance = [
            f"Keep the response within {word_limit} words/tokens.",
            "Match the requested tone and language.",
            "Address the recipient and include a clear action request.",
        ]
        if include_deadline:
            acceptance.append("Include an explicit deadline or due date.")
        if include_bullets:
            acceptance.append("Include at least one bullet list for action items.")
        acceptance = _merge_list_unique(acceptance, success_criteria)

        objective = clarified_request or (
            f"{PURPOSE_TO_OBJECTIVE.get(purpose, PURPOSE_TO_OBJECTIVE['other'])} "
            f"Custom purpose: {custom_purpose}. Original request: {text}"
        ).strip()

        spec = {
            "task_type": "email",
            "objective": objective,
            "context": {
                "background": background,
                "deadline_text": deadline_text,
                "order_or_po_number": order_or_po_number,
                "current_blocker": current_blocker,
                "intent_frame": intent_frame,
            },
            "audience": {
                "recipient_type": custom_recipient or answers.get("recipient_type", "vendor"),
                "relationship": answers.get("relationship", "existing"),
            },
            "language": answers.get("language", "en"),
            "tone": answers.get("tone", "professional"),
            "constraints": {
                "word_limit": word_limit,
                "must_include_deadline": include_deadline,
                "must_include_bullets": include_bullets,
                "hard_constraints": hard_constraints,
            },
            "must_include": must_include,
            "must_avoid": must_avoid,
            "output_format": {
                "sections": ["Subject", "Greeting", "Body", "Action Items", "Closing"],
                "bullet_list_required": include_bullets,
                "preference": output_preference,
            },
            "acceptance_criteria": acceptance,
        }
        return spec

    def prompts(self, spec: dict, route: dict) -> list[dict]:
        prompt = render_email_prompt(spec)
        rows = []
        for ex in route.get("recommended_executors", []):
            rows.append(
                {
                    "executor": ex,
                    "prompt": prompt,
                    "notes": "Use this prompt as-is for the selected executor.",
                }
            )
        return rows

    def validate(self, spec: dict, output: str) -> dict:
        return validate_email_output(spec, output)


def _lines_to_list(text: str) -> list[str]:
    lines = []
    for raw in (text or "").splitlines():
        s = raw.strip().lstrip("-*0123456789. ")
        if s:
            lines.append(s)
    return lines


def _merge_list_unique(base: list[str], extra: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in (base or []) + (extra or []):
        s = (item or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _build_intent_frame(answers: dict) -> dict:
    return {
        "motivation": (answers.get("motivation") or "").strip(),
        "primary_target": (answers.get("primary_target") or "").strip(),
        "stakeholders": (answers.get("stakeholders") or "").strip(),
        "style_modifiers": _lines_to_list(answers.get("style_modifiers", "")),
    }
