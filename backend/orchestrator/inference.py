from __future__ import annotations

import re

import llm_client

from .ml_extractor import predict_slots


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ALLOWED_TIME_RANGE = {"today", "tomorrow", "next_3_days", "next_7_days", "custom"}
_ALLOWED_OUTPUT_PREFERENCE = {"direct", "outline_then_final", "options_then_pick"}
_ML_SLOT_THRESHOLDS = {
    "task_domain": 0.6,
    "language": 0.75,
    "output_preference": 0.72,
}


def infer_initial_answers(task_type: str, text: str, context: dict | None = None) -> dict:
    """Infer likely clarification answers from raw text.

    This is intentionally lightweight:
    - deterministic regex/keyword extraction first
    - optional LLM extraction to refine missing slots
    """
    base = _infer_common(text)
    if task_type == "email":
        base.update(_infer_email(text))
    elif task_type == "writing":
        base.update(_infer_writing(text))
    elif task_type == "code":
        base.update(_infer_code(text))
    elif task_type == "generic":
        base.update(_infer_generic(text))

    ml_slots = predict_slots(text)
    _merge_ml_slots_with_confidence(base, ml_slots)

    if context and isinstance(context, dict):
        for key in ("background", "target_audience", "language"):
            value = context.get(key)
            if value and not base.get(key):
                base[key] = value

    llm_guess = _infer_with_llm(task_type, text)
    for key, value in llm_guess.items():
        if _has_value(value) and not _has_value(base.get(key)):
            base[key] = value
    if task_type == "generic":
        _finalize_generic_with_intent(base, text, ml_slots)
    return _sanitize_inferred(base, task_type, text)


def apply_inferred_defaults(schema: dict, inferred: dict) -> dict:
    if not schema or not inferred:
        return schema
    fields = []
    for field in (schema.get("fields") or []):
        key = field.get("key")
        if key and key in inferred and _has_value(inferred.get(key)):
            field = {**field, "default": inferred.get(key)}
        fields.append(field)
    return {**schema, "fields": fields}


def _infer_common(text: str) -> dict:
    word_limit = _extract_word_limit(text)
    modifiers = _extract_style_modifiers(text)
    return {
        "clarified_request": text.strip(),
        "motivation": _extract_motivation(text),
        "primary_target": _extract_primary_target(text),
        "stakeholders": "",
        "style_modifiers": "\n".join(modifiers),
        "output_preference": _extract_output_preference(text),
        "word_limit": word_limit if word_limit else 200,
    }


def _infer_email(text: str) -> dict:
    t = text.lower()
    recipient_type = "vendor"
    if any(k in t for k in ("客户", "client")):
        recipient_type = "client"
    elif any(k in t for k in ("老板", "上级", "manager", "lead")):
        recipient_type = "manager"
    elif any(k in t for k in ("同事", "colleague")):
        recipient_type = "colleague"

    purpose = "follow_up"
    if any(k in t for k in ("发票", "invoice")):
        purpose = "request_invoice"
    elif any(k in t for k in ("催", "推进", "chase", "urgent")):
        purpose = "chase_progress"

    language = "zh" if _looks_chinese(text) else "en"
    tone = "professional"
    if any(k in t for k in ("强硬", "坚定", "firm", "严肃")):
        tone = "firm"
    elif any(k in t for k in ("友好", "friendly", "客气")):
        tone = "friendly"

    deadline_text = _extract_deadline_text(text)
    include_deadline = bool(deadline_text) or any(
        k in t for k in ("截止", "最晚", "deadline", "before", "by ")
    )
    include_bullets = any(k in t for k in ("bullet", "列表", "清单", "要点"))

    return {
        "recipient_type": recipient_type,
        "relationship": "existing",
        "purpose": purpose,
        "tone": tone,
        "language": language,
        "primary_target": "收件人/邮件接收方",
        "word_limit": _extract_word_limit(text) or 200,
        "include_deadline": include_deadline,
        "deadline_text": deadline_text,
        "include_bullets": include_bullets,
    }


def _infer_writing(text: str) -> dict:
    t = text.lower()
    platform = "general"
    if any(k in t for k in ("小红书", "xiaohongshu", "rednote")):
        platform = "xiaohongshu"
    elif any(k in t for k in ("公众号", "wechat")):
        platform = "wechat"
    elif any(k in t for k in ("抖音", "douyin", "tiktok")):
        platform = "douyin"

    goal = "awareness"
    if any(k in t for k in ("转化", "下单", "咨询", "conversion")):
        goal = "conversion"
    elif any(k in t for k in ("互动", "评论", "engagement")):
        goal = "engagement"

    tone = "friendly"
    if any(k in t for k in ("专业", "professional")):
        tone = "professional"
    elif any(k in t for k in ("故事", "story")):
        tone = "storytelling"
    elif any(k in t for k in ("强势", "bold")):
        tone = "bold"

    return {
        "platform": platform,
        "goal": goal,
        "tone": tone,
        "primary_target": "目标读者/受众",
        "length": _infer_length_bucket(text),
    }


def _infer_code(text: str) -> dict:
    t = text.lower()
    change_type = "feature"
    if any(k in t for k in ("bug", "修复", "fix", "报错")):
        change_type = "bugfix"
    elif any(k in t for k in ("重构", "refactor", "优化结构")):
        change_type = "refactor"

    language = ""
    if "react" in t:
        language = "React"
    elif "flask" in t:
        language = "Flask/Python"
    elif "python" in t:
        language = "Python"
    elif "typescript" in t or "ts" in t:
        language = "TypeScript"
    elif "javascript" in t or "js" in t:
        language = "JavaScript"

    return {
        "change_type": change_type,
        "desired_change": text.strip(),
        "language": language,
        "primary_target": "代码仓库/目标模块",
        "tests_constraint": "run_related_tests",
        "no_breaking_changes": True,
    }


def _infer_generic(text: str) -> dict:
    out = {
        "task_domain": "analysis",
        "expected_output_type": "structured",
    }
    return out


def _infer_with_llm(task_type: str, text: str) -> dict:
    if not llm_client.is_available():
        return {}
    system = (
        "你是任务参数抽取器。请从用户输入提取最可能的任务参数。"
        "只返回 JSON 对象，不要解释。"
    )
    prompt = (
        "请抽取如下键（缺失就返回空字符串或 false）：\n"
        "- clarified_request\n- target_audience\n- language(zh|en)\n- word_limit(number)\n"
        "- include_deadline(boolean)\n- deadline_text\n- location\n- time_range\n"
        "- motivation\n- primary_target\n- stakeholders\n- style_modifiers(array of short strings)\n\n"
        f"task_type={task_type}\n"
        f"text={text}"
    )
    try:
        data = llm_client.chat_json(prompt=prompt, system_prompt=system)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    allowed = {
        "clarified_request",
        "target_audience",
        "language",
        "word_limit",
        "include_deadline",
        "deadline_text",
        "location",
        "time_range",
        "motivation",
        "primary_target",
        "stakeholders",
        "style_modifiers",
    }
    if isinstance(data.get("style_modifiers"), list):
        data["style_modifiers"] = "\n".join(str(x).strip() for x in data["style_modifiers"] if str(x).strip())
    return {k: v for k, v in data.items() if k in allowed}


def _extract_word_limit(text: str) -> int | None:
    m = re.search(r"(\d{2,4})\s*(字|词|words?|tokens?)", text, flags=re.IGNORECASE)
    if not m:
        return None
    value = int(m.group(1))
    return max(50, min(1000, value))


def _extract_output_preference(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("先给大纲", "outline first", "先提纲")):
        return "outline_then_final"
    if any(k in t for k in ("多个方案", "2个方案", "3个方案", "multiple options")):
        return "options_then_pick"
    return "direct"


def _extract_motivation(text: str) -> str:
    patterns = [
        r"(为了[^，。；\n]{2,40})",
        r"(因为[^，。；\n]{2,40})",
        r"(so that[^,.\n]{2,40})",
        r"(because[^,.\n]{2,40})",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def _extract_primary_target(text: str) -> str:
    t = (text or "").strip()
    cn_patterns = [
        r"(关于[^，。；\n]{1,25})",
        r"(分析[^，。；\n]{1,25})",
        r"(解释[^，。；\n]{1,25})",
        r"(介绍[^，。；\n]{1,25})",
    ]
    for p in cn_patterns:
        m = re.search(p, t)
        if m:
            value = m.group(1).strip()
            if len(value) >= 2:
                return value

    en_patterns = [
        r"(for\s+[^,.\n]{2,80})",
        r"(about\s+[^,.\n]{2,80})",
        r"(analyze\s+[^,.\n]{2,80})",
        r"(explain\s+[^,.\n]{2,80})",
    ]
    for p in en_patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return _extract_target_by_object_span(t)


def _extract_style_modifiers(text: str) -> list[str]:
    t = (text or "").strip()
    found: list[str] = []

    # 显式风格描述（优先）
    for p in [
        r"(?:语气|风格|口吻|语调)\s*(?:是|为|要|:|：)?\s*([^\s，。；,.;]{1,18})",
        r"(?:tone|style)\s*(?:is|be|:)?\s*([a-zA-Z\-]{3,24})",
    ]:
        for m in re.finditer(p, t, flags=re.IGNORECASE):
            cand = _normalize_modifier(m.group(1).strip())
            if _is_likely_modifier(cand):
                found.append(cand)

    # 中文常见结构：形容词 + 的 + 名词
    for m in re.finditer(r"([\u4e00-\u9fff]{1,6})的[\u4e00-\u9fff]{1,10}", t):
        adj = _normalize_modifier(m.group(1).strip())
        # 控制误提取：只保留短修饰词或含否定/程度对比描述。
        if len(adj) > 4 and not any(x in adj for x in ("不", "更", "较", "偏")):
            continue
        if _is_likely_modifier(adj):
            found.append(adj)

    # 程度副词 + 形容词（如：非常新颖）
    for m in re.finditer(r"(很|非常|特别|更|较为|相对)([\u4e00-\u9fff]{1,6})", t):
        adj = _normalize_modifier(m.group(2).strip())
        if _is_likely_modifier(adj):
            found.append(adj)

    dedup = []
    seen = set()
    for x in found:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup[:6]


def _is_likely_modifier(word: str) -> bool:
    if not word:
        return False
    if len(word) == 1:
        return False
    # 过滤高频功能词，保持通用
    blocked = {"这个", "那个", "一些", "一种", "一个", "今天", "最近", "你要", "帮我"}
    if word in blocked:
        return False
    if re.search(r"\d", word):
        return False
    if word.endswith(("分钟", "小时", "天", "周", "月", "公里", "千米", "米", "kg", "w")):
        return False
    # 过滤明显名词/量词片段，避免把“生成一篇小红书”这类误判为风格词
    noun_like_patterns = [
        r"一篇", r"一条", r"一个", r"帖子", r"文章", r"文案", r"邮件", r"代码",
        r"小红书", r"天气", r"特斯拉", r"公司", r"介绍", r"生成", r"写",
    ]
    return not any(re.search(p, word) for p in noun_like_patterns)


def _normalize_modifier(word: str) -> str:
    if not word:
        return ""
    s = word.strip()
    s = re.sub(r"^(帮我|请|写|生成|做|来|给我)+", "", s)
    s = re.sub(r"^(一篇|一条|一个|一份|这篇|该)", "", s)
    # 如“生成一篇新颖”兜底提取尾部形容词片段
    if len(s) > 3 and re.search(r"(新颖|专业|简洁|清晰|详细|严谨|有趣|高级|吸引人)$", s):
        m = re.search(r"(新颖|专业|简洁|清晰|详细|严谨|有趣|高级|吸引人)$", s)
        if m:
            s = m.group(1)
    return s.strip()


def _extract_deadline_text(text: str) -> str:
    patterns = [
        r"(请于[^，。；\n]{2,30}(?:前|之前))",
        r"(在[^，。；\n]{2,30}(?:前|之前))",
        r"(by\s+[^,.\n]{2,30})",
        r"(before\s+[^,.\n]{2,30})",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def _infer_length_bucket(text: str) -> str:
    wl = _extract_word_limit(text)
    if wl is None:
        return "medium"
    if wl <= 100:
        return "short"
    if wl <= 300:
        return "medium"
    return "long"


def _extract_weather_query(text: str) -> dict:
    if not _looks_like_weather_query(text):
        return {}

    t = text.strip()
    location = ""

    m_cn = re.search(r"([\u4e00-\u9fffA-Za-z·\s]{1,25})(?:的)?天气", t)
    if m_cn:
        location = _clean_location(m_cn.group(1).strip())
    if not location:
        m_en = re.search(r"weather\s+(?:in|for)\s+([A-Za-z\s]{2,30})", t, flags=re.IGNORECASE)
        if m_en:
            location = _clean_location(m_en.group(1).strip())
    if not location:
        location = _extract_weather_location_heuristic(t)

    time_range = "today"
    if re.search(r"(最近|未来|接下来).{0,3}(三天|3天)|next\s*3\s*days", t, flags=re.IGNORECASE):
        time_range = "next_3_days"
    elif re.search(r"(最近|未来|接下来).{0,3}(七天|7天)|next\s*7\s*days|week", t, flags=re.IGNORECASE):
        time_range = "next_7_days"
    elif re.search(r"明天|tomorrow", t, flags=re.IGNORECASE):
        time_range = "tomorrow"

    return {"location": location, "time_range": time_range}


def _looks_like_weather_query(text: str) -> bool:
    t = text.lower()
    weather_words = ["天气", "气温", "降雨", "wind", "temperature", "forecast", "weather"]
    return any(w in t for w in weather_words)


def _clean_location(raw: str) -> str:
    if not raw:
        return raw
    s = raw.strip()
    s = re.sub(r"^(请|帮我|给我|给出|查询|查下|查一下|看看|看下|告诉我|显示|预测)", "", s)
    s = re.sub(r"(并提醒.*|并告诉.*|是否适合.*)$", "", s)
    s = re.sub(r"(最近|未来|接下来).{0,3}(三天|3天|七天|7天|一周)$", "", s)
    s = re.sub(r"(未来|最近|接下来)?\s*\d+\s*天$", "", s)
    s = re.sub(r"(today|tomorrow|next\s*3\s*days|next\s*7\s*days)$", "", s, flags=re.IGNORECASE)
    s = s.strip(" ，。;；:：")
    if s in {"天", "天气", "forecast", "weather", "city"}:
        return ""
    return s.strip()


def _looks_chinese(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _has_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    return True


def _merge_ml_slots_with_confidence(base: dict, ml_slots: dict) -> None:
    for key, value in (ml_slots or {}).items():
        if key.endswith("_confidence"):
            continue
        if _has_value(base.get(key)):
            continue
        conf = float(ml_slots.get(f"{key}_confidence", 0.0) or 0.0)
        threshold = _ML_SLOT_THRESHOLDS.get(key, 0.7)
        if conf >= threshold and _has_value(value):
            base[key] = value


def _finalize_generic_with_intent(base: dict, text: str, ml_slots: dict) -> None:
    """Use learned intent to decide whether weather slots should be extracted."""
    intent = str(ml_slots.get("query_intent", "") or "").strip()
    intent_conf = float(ml_slots.get("query_intent_confidence", 0.0) or 0.0)

    should_weather = False
    if intent == "weather_query" and intent_conf >= 0.75:
        should_weather = True
    elif intent == "general" and intent_conf < 0.75 and _looks_like_weather_query(text):
        # Small model is uncertain: allow conservative regex fallback.
        should_weather = True
    elif not intent and _looks_like_weather_query(text):
        # No model signal available: conservative fallback.
        should_weather = True

    if not should_weather:
        base.pop("location", None)
        base.pop("time_range", None)
        if base.get("task_domain") == "research":
            base["task_domain"] = "analysis"
        return

    weather = _extract_weather_query(text)
    if weather.get("location"):
        base["location"] = weather.get("location")
        base["primary_target"] = weather.get("location")
    if weather.get("time_range"):
        base["time_range"] = weather.get("time_range")
    base["task_domain"] = "research"


def _sanitize_inferred(inferred: dict, task_type: str, text: str) -> dict:
    out = dict(inferred or {})
    if out.get("word_limit"):
        try:
            out["word_limit"] = int(out["word_limit"])
            out["word_limit"] = max(50, min(1000, out["word_limit"]))
        except Exception:
            out.pop("word_limit", None)

    if out.get("language") and out.get("language") not in {"zh", "en"}:
        out.pop("language", None)

    if out.get("output_preference") and out.get("output_preference") not in _ALLOWED_OUTPUT_PREFERENCE:
        out["output_preference"] = "direct"

    if out.get("time_range") and out.get("time_range") not in _ALLOWED_TIME_RANGE:
        out.pop("time_range", None)

    if out.get("location") and not _is_valid_location(out.get("location", "")):
        out.pop("location", None)

    if out.get("primary_target") and not _is_valid_primary_target(out.get("primary_target", ""), text):
        out.pop("primary_target", None)

    if out.get("style_modifiers"):
        modifiers = _lines_to_modifiers(out.get("style_modifiers", ""))
        out["style_modifiers"] = "\n".join(modifiers)

    if task_type == "generic" and _looks_like_weather_query(text):
        # 天气任务如果地点无效，不做猜测，交给 clarifier 追问。
        if not _is_valid_location(out.get("location", "")):
            out.pop("location", None)

    return out


def _is_valid_location(location: str) -> bool:
    s = (location or "").strip()
    if len(s) < 2:
        return False
    banned = {"天", "天气", "weather", "forecast", "city", "未来", "最近"}
    if s.lower() in banned:
        return False
    if re.fullmatch(r"\d+\s*天", s):
        return False
    return True


def _is_valid_primary_target(target: str, text: str) -> bool:
    t = (target or "").strip()
    if len(t) < 2 or len(t) > 80:
        return False
    noisy_prefix = ("帮我", "请", "给我", "Tell me", "Explain", "Summarize")
    if any(t.startswith(p) for p in noisy_prefix):
        return False
    if t in text and len(t) > max(30, int(len(text) * 0.8)):
        return False
    # 避免明显截断残片（例如最后一个 token 只有 1-2 个字母）
    tokens = t.split()
    if tokens and len(tokens[-1]) <= 2 and re.search(r"[A-Za-z]", tokens[-1]):
        return False
    return True


def _lines_to_modifiers(text: str) -> list[str]:
    items = []
    for raw in (text or "").splitlines():
        s = _normalize_modifier(raw.strip())
        if _is_likely_modifier(s):
            items.append(s)
    dedup = []
    seen = set()
    for x in items:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup[:6]


def _extract_target_by_object_span(text: str) -> str:
    t = (text or "").strip()
    cn = re.search(r"(?:介绍|分析|解释|说明|总结)([^，。；\n]{2,24})", t)
    if cn:
        return cn.group(1).strip(" 的")
    en = re.search(
        r"(?:tell me more about|about|summarize|explain|analyze)\s+([A-Za-z0-9\-\s]{2,48})",
        t,
        flags=re.IGNORECASE,
    )
    if en:
        v = en.group(1).strip()
        v = re.sub(r"\b(in|with|for)\b.*$", "", v, flags=re.IGNORECASE).strip()
        return v
    return ""


def _extract_weather_location_heuristic(text: str) -> str:
    t = text or ""
    # 在“天气/forecast/weather”前取最近片段作为候选地点，并做清洗。
    idx = t.find("天气")
    if idx == -1:
        idx = t.lower().find("weather")
    if idx == -1:
        idx = t.lower().find("forecast")
    if idx == -1:
        return ""
    prefix = t[:idx]
    prefix = re.sub(r"(请|帮我|给我|给出|查询|查下|查一下|看看|看下|告诉我|显示|预测)", "", prefix)
    prefix = re.sub(r"(最近|未来|接下来)?\s*\d+\s*天", "", prefix)
    prefix = re.sub(r"(今天|明天|后天|today|tomorrow|next\s*3\s*days|next\s*7\s*days)", "", prefix, flags=re.IGNORECASE)
    m = re.search(r"([A-Za-z\u4e00-\u9fff·\s]{2,24})$", prefix)
    if not m:
        return ""
    return _clean_location(m.group(1).strip())
