from __future__ import annotations

from .base import TaskHandler


class WritingTaskHandler(TaskHandler):
    task_type = "writing"

    def detect(self, text: str) -> float:
        lower = text.lower()
        score = 0.0
        keywords = {
            "文案": 0.45,
            "小红书": 0.6,
            "公众号": 0.55,
            "推文": 0.5,
            "标题": 0.35,
            "写一篇": 0.35,
            "写": 0.2,
            "文章": 0.28,
            "润色": 0.35,
            "改写": 0.35,
            "脚本": 0.25,
            "write": 0.2,
            "copy": 0.25,
            "post": 0.2,
            "content": 0.2,
        }
        for kw, w in keywords.items():
            if kw in lower:
                score += w
        return min(score, 1.0)

    def clarify_schema(self, text: str) -> dict:
        return {
            "title": "写作任务澄清",
            "description": "先把目标和风格对齐，再生成更可用的内容提示词。",
            "fields": [
                {
                    "key": "platform",
                    "label": "发布平台",
                    "type": "single_choice",
                    "required": True,
                    "default": "xiaohongshu",
                    "options": [
                        {"value": "xiaohongshu", "label": "小红书"},
                        {"value": "wechat", "label": "公众号"},
                        {"value": "douyin", "label": "抖音"},
                        {"value": "general", "label": "通用"},
                        {"value": "other", "label": "其他"},
                    ],
                },
                {
                    "key": "platform_other",
                    "label": "请补充平台",
                    "type": "short_text",
                    "required": False,
                    "required_when": {"platform": "other"},
                    "show_when": {"platform": "other"},
                    "placeholder": "例如：知乎、B站专栏",
                },
                {
                    "key": "goal",
                    "label": "核心目标",
                    "type": "single_choice",
                    "required": True,
                    "default": "conversion",
                    "options": [
                        {"value": "conversion", "label": "转化（下单/咨询）"},
                        {"value": "engagement", "label": "互动（点赞评论）"},
                        {"value": "awareness", "label": "种草/曝光"},
                        {"value": "other", "label": "其他"},
                    ],
                },
                {
                    "key": "goal_other",
                    "label": "请补充目标",
                    "type": "short_text",
                    "required": False,
                    "required_when": {"goal": "other"},
                    "show_when": {"goal": "other"},
                    "placeholder": "例如：引导私信领取资料",
                },
                {
                    "key": "audience",
                    "label": "目标人群",
                    "type": "short_text",
                    "required": True,
                    "placeholder": "例如：25-35岁职场女性，关注护肤与通勤妆",
                },
                {
                    "key": "tone",
                    "label": "语气风格",
                    "type": "single_choice",
                    "required": True,
                    "default": "friendly",
                    "options": [
                        {"value": "professional", "label": "专业"},
                        {"value": "friendly", "label": "亲切"},
                        {"value": "bold", "label": "强势"},
                        {"value": "storytelling", "label": "故事化"},
                    ],
                },
                {
                    "key": "length",
                    "label": "篇幅",
                    "type": "single_choice",
                    "required": True,
                    "default": "medium",
                    "options": [
                        {"value": "short", "label": "短（100字内）"},
                        {"value": "medium", "label": "中（100-300字）"},
                        {"value": "long", "label": "长（300字以上）"},
                    ],
                },
                {
                    "key": "must_include",
                    "label": "必须提到的信息（可选）",
                    "type": "multiline_text",
                    "required": False,
                    "placeholder": "一行一条，例如：产品名、优惠截止时间、行动号召",
                },
                {
                    "key": "must_avoid",
                    "label": "避免出现的内容（可选）",
                    "type": "multiline_text",
                    "required": False,
                    "placeholder": "一行一条，例如：夸大宣传、医学承诺",
                },
                {
                    "key": "background",
                    "label": "背景信息",
                    "type": "multiline_text",
                    "required": True,
                    "placeholder": "补充你已有素材、卖点、场景、限制条件",
                },
            ],
        }

    def build_spec(self, text: str, answers: dict) -> dict:
        clarified_request = (answers.get("clarified_request") or "").strip()
        success_criteria = _lines_to_list(answers.get("success_criteria", ""))
        hard_constraints = _lines_to_list(answers.get("hard_constraints", ""))
        output_preference = answers.get("output_preference", "direct")
        intent_frame = _build_intent_frame(answers)

        platform = answers.get("platform", "general")
        platform_other = (answers.get("platform_other") or "").strip()
        goal = answers.get("goal", "conversion")
        goal_other = (answers.get("goal_other") or "").strip()
        length = answers.get("length", "medium")

        length_map = {
            "short": "100字以内",
            "medium": "100-300字",
            "long": "300字以上",
        }

        must_include = _lines_to_list(answers.get("must_include", ""))
        must_avoid = _lines_to_list(answers.get("must_avoid", ""))

        acceptance = [
            "内容贴合目标人群和平台风格。",
            "语气和目标一致。",
            "篇幅符合要求。",
        ]
        acceptance = _merge_list_unique(acceptance, success_criteria)

        return {
            "task_type": "writing",
            "objective": clarified_request or f"围绕用户请求产出可直接使用的内容草稿。原始请求：{text}",
            "audience": {
                "target": answers.get("audience", ""),
            },
            "platform": platform_other or platform,
            "goal": goal_other or goal,
            "tone": answers.get("tone", "friendly"),
            "constraints": {
                "length": length,
                "length_hint": length_map.get(length, "100-300字"),
                "hard_constraints": hard_constraints,
            },
            "must_include": must_include,
            "must_avoid": must_avoid,
            "context": {
                "background": answers.get("background", ""),
                "intent_frame": intent_frame,
            },
            "output_format": {
                "sections": ["标题", "正文", "结尾行动号召"],
                "preference": output_preference,
            },
            "acceptance_criteria": acceptance,
        }

    def prompts(self, spec: dict, route: dict) -> list[dict]:
        prompt = _render_writing_prompt(spec)
        rows = []
        for ex in route.get("recommended_executors", []):
            rows.append(
                {
                    "executor": ex,
                    "prompt": prompt,
                    "notes": "按该提示词生成写作结果。",
                }
            )
        return rows

    def validate(self, spec: dict, output: str) -> dict:
        issues = []
        if not output.strip():
            issues.append({"type": "empty_output", "message": "没有返回内容。"})
        pass_check = len(issues) == 0
        return {
            "pass": pass_check,
            "issues": issues,
            "suggested_fix_prompt": "请按既定 spec 重新生成更完整的内容。" if not pass_check else "",
        }


def _render_writing_prompt(spec: dict) -> str:
    must_include = "\n".join(f"- {x}" for x in (spec.get("must_include") or [])) or "- (无)"
    must_avoid = "\n".join(f"- {x}" for x in (spec.get("must_avoid") or [])) or "- (无)"
    sections = "\n".join(f"- {x}" for x in ((spec.get("output_format") or {}).get("sections") or [])) or "- (无)"
    acceptance = "\n".join(f"- {x}" for x in (spec.get("acceptance_criteria") or [])) or "- (无)"
    hard_constraints = "\n".join(
        f"- {x}" for x in ((spec.get("constraints") or {}).get("hard_constraints") or [])
    ) or "- (无)"
    intent = ((spec.get("context") or {}).get("intent_frame") or {})
    intent_lines = []
    if intent.get("motivation"):
        intent_lines.append(f"- 前因/目的：{intent.get('motivation')}")
    if intent.get("primary_target"):
        intent_lines.append(f"- 主要作用对象：{intent.get('primary_target')}")
    if intent.get("stakeholders"):
        intent_lines.append(f"- 相关对象：{intent.get('stakeholders')}")
    if intent.get("style_modifiers"):
        intent_lines.append(f"- 风格修饰词：{', '.join(intent.get('style_modifiers') or [])}")
    intent_block = "\n".join(intent_lines) or "- (未提供)"

    return (
        "你是一位资深内容策略与文案专家，请输出可直接发布的高质量成稿。\n\n"
        "【创作目标】\n"
        f"- 目标：{spec.get('objective', '')}\n"
        f"- 平台：{spec.get('platform', '')}\n"
        f"- 受众：{(spec.get('audience') or {}).get('target', '')}\n"
        f"- 语气：{spec.get('tone', '')}\n"
        f"- 篇幅：{(spec.get('constraints') or {}).get('length_hint', '')}\n\n"
        "【创作上下文】\n"
        f"{intent_block}\n"
        f"- 背景信息：{(spec.get('context') or {}).get('background', '') or '未提供'}\n\n"
        "【内容约束】\n"
        f"- 必须提到：\n{must_include}\n"
        f"- 避免出现：\n{must_avoid}\n"
        f"- 硬性约束：\n{hard_constraints}\n\n"
        "【输出结构】\n"
        f"{sections}\n\n"
        "【验收标准】\n"
        f"{acceptance}\n\n"
        "请直接输出最终成稿，不要解释你的思考过程。"
    )


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
