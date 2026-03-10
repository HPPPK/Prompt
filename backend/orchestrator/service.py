from __future__ import annotations

import re

from recommender import recommend_models

from .executor import run_executor
from .inference import apply_inferred_defaults, infer_initial_answers
from .router import get_handler, route_task
from .store import store


class ClarifyValidationError(ValueError):
    """Raised when clarify answers fail schema validation."""


def start_workflow(
    text: str,
    preferred_executor: str | None = None,
    context: dict | None = None,
) -> dict:
    task_type, handler, confidence = route_task(text)
    inferred = infer_initial_answers(task_type, text, context)
    session = store.create(
        text=text,
        preferred_executor=preferred_executor,
        context=context,
        task_type=task_type,
    )

    if task_type == "other" or not handler:
        session = store.update(
            session["session_id"],
            state="spec_ready",
            clarify_form_schema=None,
            spec_draft=None,
            inferred_answers=inferred,
            routing_confidence=confidence,
        )
        return {
            "session_id": session["session_id"],
            "state": session["state"],
            "task_type": task_type,
            "clarify_form_schema": None,
            "spec_draft": None,
        }

    # generic 但语义已经很明确时，直接产出 spec 草案，避免让用户填一整页表单。
    if task_type == "generic" and _looks_specific_request(text) and _generic_can_skip_clarify(text, inferred):
        default_answers = _default_generic_answers(text, inferred)
        spec = handler.build_spec(text, default_answers)
        session = store.update(
            session["session_id"],
            state="spec_ready",
            clarify_form_schema=None,
            clarify_answers=default_answers,
            inferred_answers=inferred,
            spec_draft=spec,
            routing_confidence=confidence,
        )
        return {
            "session_id": session["session_id"],
            "state": session["state"],
            "task_type": task_type,
            "clarify_form_schema": None,
            "spec_draft": spec,
        }

    schema = handler.clarify_schema(text)
    schema = _with_common_clarify_fields(schema, task_type)
    schema = apply_inferred_defaults(schema, inferred)
    schema, missing_slots = _build_minimal_clarify_schema(schema, task_type, inferred, text)
    missing_slot_hints = _build_missing_slot_hints(missing_slots, task_type)
    session = store.update(
        session["session_id"],
        state="clarifying",
        clarify_form_schema=schema,
        inferred_answers=inferred,
        missing_slots=missing_slots,
        missing_slot_hints=missing_slot_hints,
        routing_confidence=confidence,
    )
    return {
        "session_id": session["session_id"],
        "state": session["state"],
        "task_type": task_type,
        "clarify_form_schema": schema,
        "missing_slots": missing_slots,
        "missing_slot_hints": missing_slot_hints,
        "spec_draft": None,
    }


def submit_clarifications(session_id: str, answers: dict) -> dict:
    session = _must_session(session_id)
    if session["state"] != "clarifying":
        raise ValueError(f"Invalid state transition: {session['state']} -> spec_ready")

    inferred = session.get("inferred_answers") or {}
    merged_answers = {**inferred, **(answers or {})}
    normalized_answers = _validate_and_normalize_answers(
        session.get("clarify_form_schema"),
        merged_answers,
    )
    handler = _must_handler(session["task_type"])
    spec = handler.build_spec(session["text"], normalized_answers)

    session = store.update(
        session_id,
        clarify_answers=normalized_answers,
        spec_draft=spec,
        state="spec_ready",
    )
    return {
        "session_id": session_id,
        "state": session["state"],
        "spec_draft": session["spec_draft"],
    }


def confirm_spec(session_id: str, spec: dict) -> dict:
    session = _must_session(session_id)
    if session["state"] not in {"spec_ready", "clarifying"}:
        raise ValueError(f"Invalid state transition: {session['state']} -> confirm_spec")

    handler = _must_handler(spec.get("task_type") or session["task_type"])

    recommended = ["prompt_only", "local_lmstudio", "openai_compatible"]
    selected = session.get("preferred_executor") or "prompt_only"
    if selected not in recommended:
        selected = "prompt_only"

    route = {
        "recommended_executors": recommended,
        "selected_executor": selected,
        "recommended_models": _recommend_models_for_spec(spec, session.get("text", "")),
    }
    prompts = handler.prompts(spec, route)
    new_state = "done" if selected == "prompt_only" else "executing"

    session = store.update(
        session_id,
        spec=spec,
        spec_draft=spec,
        route=route,
        generated_prompts=prompts,
        state=new_state,
        execution=None,
        validation=None,
        final_output=None,
    )

    return {
        "session_id": session_id,
        "state": session["state"],
        "route": route,
        "generated_prompts": prompts,
        "execution": session.get("execution"),
    }


def execute_session(session_id: str, executor: str, executor_config: dict | None) -> dict:
    session = _must_session(session_id)
    if session["task_type"] == "other":
        raise ValueError("Task type 'other' is not supported by workflow execution.")
    if not session.get("spec"):
        raise ValueError("Spec is not confirmed yet.")

    prompt = _select_prompt(session, executor)
    result = run_executor(executor, prompt, executor_config or {}, {"session_id": session_id})

    if result.get("error"):
        state = "done"
    elif executor == "prompt_only":
        state = "done"
    else:
        state = "validating"

    session = store.update(
        session_id,
        state=state,
        execution=result,
        executor_config=executor_config or {},
    )

    return {
        "session_id": session_id,
        "state": session["state"],
        "execution": result,
    }


def validate_session_output(
    session_id: str,
    output: str | None = None,
    auto_revise: bool = False,
) -> dict:
    session = _must_session(session_id)
    if session["task_type"] == "other":
        raise ValueError("Task type 'other' is not supported by workflow validation.")

    spec = session.get("spec") or session.get("spec_draft")
    if not spec:
        raise ValueError("No spec found for validation.")

    handler = _must_handler(spec.get("task_type") or session["task_type"])

    execution = session.get("execution") or {}
    current_output = output if output is not None else execution.get("raw_output", "")

    report = handler.validate(spec, current_output)
    final_output = current_output

    if (
        auto_revise
        and not report.get("pass")
        and execution
        and execution.get("executor") not in {None, "prompt_only"}
        and report.get("suggested_fix_prompt")
    ):
        revised = run_executor(
            execution.get("executor"),
            report["suggested_fix_prompt"],
            session.get("executor_config") or {},
            {"session_id": session_id, "mode": "auto_revise"},
        )
        if not revised.get("error") and revised.get("raw_output"):
            final_output = revised["raw_output"]
            execution = revised
            report = handler.validate(spec, final_output)

    session = store.update(
        session_id,
        state="done",
        execution=execution,
        validation=report,
        final_output=handler.postprocess(final_output),
    )

    return {
        "session_id": session_id,
        "state": session["state"],
        "validation": session["validation"],
        "final_output": session["final_output"],
    }


def get_session(session_id: str) -> dict | None:
    return store.get(session_id)


def _must_session(session_id: str) -> dict:
    session = store.get(session_id)
    if not session:
        raise KeyError(f"Session not found: {session_id}")
    return session


def _must_handler(task_type: str):
    handler = get_handler(task_type)
    if not handler:
        raise ValueError(f"No handler for task_type: {task_type}")
    return handler


def _select_prompt(session: dict, executor: str) -> str:
    prompts = session.get("generated_prompts") or []
    for row in prompts:
        if row.get("executor") == executor:
            return row.get("prompt", "")
    if prompts:
        return prompts[0].get("prompt", "")
    return ""


def _validate_and_normalize_answers(schema: dict | None, answers: dict | None) -> dict:
    if schema is None:
        return answers or {}
    if answers is None:
        answers = {}
    if not isinstance(answers, dict):
        raise ClarifyValidationError("answers must be a JSON object.")

    fields = schema.get("fields") or []
    normalized: dict = {}
    errors: list[str] = []

    for field in fields:
        key = field.get("key")
        field_type = field.get("type")
        default = field.get("default")
        options = [opt.get("value") for opt in (field.get("options") or [])]

        if not key:
            continue
        show_when = field.get("show_when")
        if show_when and not _field_condition_match(show_when, answers, normalized, fields):
            continue

        if key in answers:
            raw_value = answers.get(key)
        elif default is not None:
            raw_value = default
        else:
            raw_value = None

        required = bool(field.get("required", False)) or _field_condition_match(
            field.get("required_when"), answers, normalized, fields
        )
        if required and _is_empty(raw_value, field_type):
            errors.append(f"{key}: required field is missing.")
            continue

        if _is_empty(raw_value, field_type):
            # Optional empty field.
            if field_type == "multi_choice":
                normalized[key] = []
            continue

        ok, casted_or_err = _cast_value(raw_value, field_type, options, field)
        if not ok:
            errors.append(f"{key}: {casted_or_err}")
            continue

        normalized[key] = casted_or_err

    if errors:
        raise ClarifyValidationError("Clarify validation failed: " + "; ".join(errors))

    return normalized


def _is_empty(value, field_type: str | None) -> bool:
    if value is None:
        return True
    if field_type in {"short_text", "multiline_text", "single_choice"}:
        return str(value).strip() == ""
    if field_type == "multi_choice":
        return not isinstance(value, list) or len(value) == 0
    return False


def _cast_value(value, field_type: str | None, options: list, field: dict):
    if field_type in {"short_text", "multiline_text"}:
        if isinstance(value, str):
            return True, value.strip()
        return False, "must be a string."

    if field_type == "single_choice":
        if not isinstance(value, str):
            return False, "must be a string option."
        if options and value not in options:
            return False, f"must be one of: {options}."
        return True, value

    if field_type == "multi_choice":
        if not isinstance(value, list):
            return False, "must be an array."
        for item in value:
            if not isinstance(item, str):
                return False, "all options in array must be strings."
            if options and item not in options:
                return False, f"contains invalid option '{item}', valid: {options}."
        return True, value

    if field_type == "number":
        try:
            num = float(value)
        except (TypeError, ValueError):
            return False, "must be a number."

        min_v = field.get("min")
        max_v = field.get("max")
        if min_v is not None and num < float(min_v):
            return False, f"must be >= {min_v}."
        if max_v is not None and num > float(max_v):
            return False, f"must be <= {max_v}."

        if num.is_integer():
            return True, int(num)
        return True, num

    if field_type == "boolean":
        if isinstance(value, bool):
            return True, value
        if isinstance(value, (int, float)) and value in (0, 1):
            return True, bool(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"true", "1", "yes", "y"}:
                return True, True
            if s in {"false", "0", "no", "n"}:
                return True, False
        return False, "must be a boolean."

    # Unknown type: keep as-is for forward compatibility.
    return True, value


def _field_condition_match(condition, raw_answers: dict, normalized: dict, fields: list[dict]) -> bool:
    if not condition:
        return False
    if not isinstance(condition, dict):
        return False
    for dep_key, expected in condition.items():
        actual = _lookup_value(dep_key, raw_answers, normalized, fields)
        if actual != expected:
            return False
    return True


def _lookup_value(dep_key: str, raw_answers: dict, normalized: dict, fields: list[dict]):
    if dep_key in normalized:
        return normalized[dep_key]
    if dep_key in raw_answers:
        return raw_answers.get(dep_key)
    for field in fields:
        if field.get("key") == dep_key and field.get("default") is not None:
            return field.get("default")
    return None


def _with_common_clarify_fields(schema: dict, task_type: str) -> dict:
    """Inject a universal clarify layer before task-specific fields."""
    fields = list(schema.get("fields") or [])
    common_fields = [
        {
            "key": "clarified_request",
            "label": "你最终想交付的结果（必填）",
            "type": "multiline_text",
            "required": True,
            "placeholder": "用 1-3 句话写清楚最终要产出什么（例如：一封可直接发送的催发票邮件）。",
            "help_text": "先明确“做什么”，避免后续执行跑偏。",
        },
        {
            "key": "motivation",
            "label": "做这件事的前因/目的（可选）",
            "type": "multiline_text",
            "required": False,
            "placeholder": "例如：月底对账要完成，发票未到导致财务无法入账。",
            "help_text": "说明“为什么现在做”，有助于系统判断优先级和语气。",
        },
        {
            "key": "primary_target",
            "label": "主要作用对象（可选）",
            "type": "short_text",
            "required": False,
            "placeholder": "例如：供应商A、特斯拉商业模式、纽约天气",
            "help_text": "说明“对谁/对什么做”，避免内容太空泛。",
        },
        {
            "key": "stakeholders",
            "label": "还涉及哪些对象（可选）",
            "type": "short_text",
            "required": False,
            "placeholder": "例如：财务、法务、老板、客户",
            "help_text": "如果有次要相关方，写出来便于平衡表达。",
        },
        {
            "key": "style_modifiers",
            "label": "风格修饰词（可选）",
            "type": "multiline_text",
            "required": False,
            "placeholder": "一行一个，例如：新颖\n简洁\n可信",
            "help_text": "系统会自动抽取你原话中的形容词，你也可以手动补充。",
        },
        {
            "key": "success_criteria",
            "label": "你怎么判断结果“够好”（可选）",
            "type": "multiline_text",
            "required": False,
            "placeholder": "一行一条，例如：\n结论要有依据\n语气不能太强硬\n控制在200字",
            "help_text": "这是验收标准，不填也可以。",
        },
        {
            "key": "hard_constraints",
            "label": "硬性约束（可选）",
            "type": "multiline_text",
            "required": False,
            "placeholder": "一行一条，例如：\n不能提竞品名\n必须用中文\n不能编造数据",
            "help_text": "任何不能违反的限制都写这里。",
        },
        {
            "key": "output_preference",
            "label": "输出方式偏好",
            "type": "single_choice",
            "required": True,
            "default": "direct",
            "options": [
                {"value": "direct", "label": "直接给最终结果"},
                {"value": "outline_then_final", "label": "先大纲再最终结果"},
                {"value": "options_then_pick", "label": "先给多个方案再细化"},
            ],
            "help_text": "决定执行器生成内容的组织方式。",
        },
    ]

    return {
        **schema,
        "description": (
            f"{schema.get('description', '')}\n"
            "先补齐“前因-对象-交付-验收”四要素，再填写任务专用问题。"
        ),
        "fields": common_fields + fields,
    }


def _looks_specific_request(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 4:
        return False

    specific_patterns = [
        r"解释.+(意思|含义)",
        r"什么是.+",
        r".+是什么意思",
        r"请说明.+",
        r"define\s+.+",
        r"explain\s+.+",
        r"what\s+is\s+.+",
    ]
    if any(re.search(p, t, flags=re.IGNORECASE) for p in specific_patterns):
        return True

    # 含明确宾语且非泛泛请求，通常可直接进入 spec_ready。
    has_explicit_object = len(t) >= 8 and (" " in t or any("\u4e00" <= c <= "\u9fff" for c in t))
    vague_patterns = [r"帮我想想", r"给点建议", r"随便写", r"不知道怎么做"]
    is_vague = any(re.search(p, t) for p in vague_patterns)
    return has_explicit_object and not is_vague


def _default_generic_answers(text: str, inferred: dict | None = None) -> dict:
    defaults = {
        "clarified_request": text.strip(),
        "motivation": "",
        "primary_target": "",
        "stakeholders": "",
        "style_modifiers": "",
        "success_criteria": "",
        "hard_constraints": "",
        "output_preference": "direct",
        "task_domain": "analysis",
        "target_audience": "",
        "expected_output_type": "structured",
        "background": "用户未提供额外背景信息。",
    }
    inferred = inferred or {}
    defaults.update({k: v for k, v in inferred.items() if v not in (None, "")})
    return defaults


def _build_minimal_clarify_schema(
    schema: dict,
    task_type: str,
    inferred: dict,
    text: str,
) -> tuple[dict, list[str]]:
    """Reduce clarify schema to only currently-missing key questions."""
    fields = list(schema.get("fields") or [])
    key_to_field = {f.get("key"): f for f in fields if f.get("key")}

    required_missing = []
    for field in fields:
        key = field.get("key")
        if not key:
            continue
        if not _field_active(field, inferred):
            continue
        required_now = bool(field.get("required")) or _field_condition_match(
            field.get("required_when"), inferred, inferred, fields
        )
        if not required_now:
            continue
        val = inferred.get(key, field.get("default"))
        if _is_empty(val, field.get("type")):
            required_missing.append(key)

    target_missing = []
    if task_type == "generic":
        weather_like = _looks_like_weather_like_text(text)
        if weather_like:
            for k in ("location", "time_range"):
                if _is_empty(inferred.get(k), (key_to_field.get(k) or {}).get("type")):
                    target_missing.append(k)
        else:
            if _is_empty(inferred.get("primary_target"), "short_text"):
                target_missing.append("primary_target")

    prioritized = []
    for k in required_missing + target_missing:
        if k not in prioritized and k in key_to_field:
            prioritized.append(k)

    if not prioritized:
        # 没有缺失关键槽位时，保留原 schema（避免回归）。
        return schema, []

    selected = []
    added = set()

    # 始终包含目标字段，便于用户快速确认。
    for key in ("clarified_request",):
        if key in key_to_field and key not in added:
            selected.append(key_to_field[key])
            added.add(key)

    for key in prioritized[:3]:
        if key in key_to_field and key not in added:
            f = dict(key_to_field[key])
            if key in target_missing:
                f["required"] = True
            selected.append(f)
            added.add(key)

    minimal = {
        **schema,
        "description": "只补齐当前缺失的关键信息（最少问题），即可进入下一步。",
        "fields": selected,
    }
    return minimal, prioritized


def _build_missing_slot_hints(missing_slots: list[str], task_type: str) -> dict:
    common = {
        "clarified_request": "明确最终交付物，避免输出方向跑偏。",
        "primary_target": "明确作用对象，避免结果过于泛化。",
        "background": "补充上下文，避免事实缺失导致内容空泛。",
        "audience": "明确受众，便于控制表达深度和语气。",
        "location": "地点是查询环境信息（如天气）的关键参数。",
        "time_range": "时间范围会直接影响结果内容与有效性。",
    }
    task_specific = {
        "email": {
            "background": "邮件场景需要背景，才能写出可执行且有说服力的正文。",
        },
        "writing": {
            "audience": "写作任务必须有受众，才能确定语气、角度和信息密度。",
        },
        "generic": {
            "primary_target": "通用任务先明确对象，再决定后续分析或执行路径。",
        },
    }.get(task_type, {})

    hints = {}
    for slot in missing_slots or []:
        hints[slot] = task_specific.get(slot) or common.get(slot) or "这是推进下一步所需的关键参数。"
    return hints


def _field_active(field: dict, answers: dict) -> bool:
    cond = field.get("show_when")
    if not cond:
        return True
    if not isinstance(cond, dict):
        return True
    for dep_key, expected in cond.items():
        actual = answers.get(dep_key)
        if actual != expected:
            return False
    return True


def _looks_like_weather_like_text(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in ("天气", "forecast", "weather", "temperature", "降雨", "气温"))


def _generic_can_skip_clarify(text: str, inferred: dict) -> bool:
    """Only skip clarify when inferred slots are reliable enough."""
    t = (text or "").lower()
    weather_like = any(w in t for w in ("天气", "forecast", "weather", "temperature"))
    if weather_like:
        location = (inferred or {}).get("location", "")
        time_range = (inferred or {}).get("time_range", "")
        return bool(str(location).strip()) and bool(str(time_range).strip())

    # 通用任务至少要有较可信的目标描述，才可直接跳过澄清。
    target = (inferred or {}).get("primary_target", "")
    objective = (inferred or {}).get("clarified_request", "")
    return bool(str(target).strip()) and len(str(objective).strip()) >= 8


def _recommend_models_for_spec(spec: dict, original_text: str) -> list[dict]:
    task_type = spec.get("task_type")
    map_type = {
        "email": "writing",
        "writing": "writing",
        "code": "coding",
        "generic": "reasoning",
    }.get(task_type, "writing")

    classification = {
        "task_types": [{"type": map_type, "confidence": 0.8}],
        "complexity": "medium",
        "intent": spec.get("objective") or original_text,
        "key_entities": [],
        "source": "workflow",
    }
    recs = recommend_models(classification, top_n=3)
    return [
        {
            "name": r.get("name"),
            "provider": r.get("provider"),
            "reason": r.get("reason"),
            "match_pct": r.get("match_pct"),
        }
        for r in recs
    ]
