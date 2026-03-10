from __future__ import annotations


def render_email_prompt(spec: dict) -> str:
    audience = spec.get("audience", {})
    constraints = spec.get("constraints", {})
    output_format = spec.get("output_format", {})
    context = spec.get("context", {})

    sections = "\n".join(f"- {s}" for s in output_format.get("sections", []))
    must_include = "\n".join(f"- {x}" for x in spec.get("must_include", [])) or "- (none)"
    must_avoid = "\n".join(f"- {x}" for x in spec.get("must_avoid", [])) or "- (none)"
    acceptance = "\n".join(f"- {x}" for x in spec.get("acceptance_criteria", []))
    background = context.get("background", "")
    context_lines = []
    intent = context.get("intent_frame", {}) or {}
    if intent.get("motivation"):
        context_lines.append(f"- Why now: {intent['motivation']}")
    if intent.get("primary_target"):
        context_lines.append(f"- Primary target: {intent['primary_target']}")
    if intent.get("stakeholders"):
        context_lines.append(f"- Related stakeholders: {intent['stakeholders']}")
    if intent.get("style_modifiers"):
        context_lines.append(f"- Style modifiers: {', '.join(intent['style_modifiers'])}")
    if background:
        context_lines.append(f"- Background: {background}")
    if context.get("order_or_po_number"):
        context_lines.append(f"- Order/PO: {context['order_or_po_number']}")
    if context.get("current_blocker"):
        context_lines.append(f"- Current blocker: {context['current_blocker']}")
    if context.get("deadline_text"):
        context_lines.append(f"- Deadline detail: {context['deadline_text']}")
    context_text = "\n".join(context_lines) or "- (none)"

    return (
        "You are an expert business writing assistant. Draft ONE final email exactly matching the spec.\n\n"
        f"Objective:\n{spec.get('objective', '')}\n\n"
        f"Language: {spec.get('language', 'en')}\n"
        f"Tone: {spec.get('tone', 'professional')}\n"
        f"Recipient type: {audience.get('recipient_type', '')}\n"
        f"Relationship: {audience.get('relationship', '')}\n"
        f"Word limit: {constraints.get('word_limit', 200)}\n"
        f"Must include deadline: {constraints.get('must_include_deadline', False)}\n"
        f"Must include bullets: {constraints.get('must_include_bullets', False)}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Must include:\n{must_include}\n\n"
        f"Must avoid:\n{must_avoid}\n\n"
        f"Output sections:\n{sections}\n\n"
        f"Acceptance criteria:\n{acceptance}\n\n"
        "Return only the email content."
    )


def render_code_prompt(spec: dict) -> str:
    constraints = spec.get("constraints", {})
    context = spec.get("context", {}) or {}
    intent = context.get("intent_frame", {}) or {}
    files = spec.get("files_affected", [])
    files_text = "\n".join(f"- {f}" for f in files) or "- (not specified)"
    acceptance = "\n".join(f"- {x}" for x in spec.get("acceptance_criteria", []))
    intent_lines = []
    if intent.get("motivation"):
        intent_lines.append(f"- Why now: {intent['motivation']}")
    if intent.get("primary_target"):
        intent_lines.append(f"- Primary target: {intent['primary_target']}")
    if intent.get("stakeholders"):
        intent_lines.append(f"- Related stakeholders: {intent['stakeholders']}")
    if intent.get("style_modifiers"):
        intent_lines.append(f"- Style modifiers: {', '.join(intent['style_modifiers'])}")
    intent_text = "\n".join(intent_lines) or "- (not provided)"

    return (
        "You are a senior software engineer. Implement the requested code change in an existing repository.\n\n"
        f"Objective:\n{spec.get('objective', '')}\n\n"
        f"Change type: {spec.get('change_type', 'feature')}\n"
        f"Preferred language/framework: {constraints.get('language', 'unspecified')}\n"
        f"Test requirement: {constraints.get('tests', 'run relevant tests')}\n"
        f"No breaking changes: {constraints.get('no_breaking_changes', True)}\n\n"
        f"Intent frame:\n{intent_text}\n\n"
        f"Potential files/areas:\n{files_text}\n\n"
        f"Acceptance criteria:\n{acceptance}\n\n"
        "Output format priority: 1) unified diff/patch, 2) file-by-file change list with exact code blocks, 3) precise implementation steps."
    )
