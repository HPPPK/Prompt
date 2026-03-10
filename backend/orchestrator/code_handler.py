from __future__ import annotations

from .base import TaskHandler
from .prompts import render_code_prompt
from .validator import validate_code_output


class CodeTaskHandler(TaskHandler):
    task_type = "code"

    def detect(self, text: str) -> float:
        lower = text.lower()
        score = 0.0
        keywords = [
            "代码", "前端", "后端", "api", "接口", "按钮", "bug", "修复",
            "feature", "refactor", "test", "lint", "file", "repo", "react", "flask", "python", "js",
        ]
        for kw in keywords:
            if kw in lower:
                score += 0.14
        return min(score, 1.0)

    def clarify_schema(self, text: str) -> dict:
        return {
            "title": "Code Change Clarification",
            "description": "Collect minimal implementation details for code orchestration.",
            "fields": [
                {
                    "key": "repo_area_or_paths",
                    "label": "Repo Area / Paths",
                    "type": "multiline_text",
                    "required": False,
                    "placeholder": "e.g. frontend/src/App.js\\nbackend/app.py",
                },
                {
                    "key": "change_type",
                    "label": "Change Type",
                    "type": "single_choice",
                    "required": True,
                    "default": "feature",
                    "options": [
                        {"value": "feature", "label": "Feature"},
                        {"value": "bugfix", "label": "Bugfix"},
                        {"value": "refactor", "label": "Refactor"},
                    ],
                },
                {
                    "key": "desired_change",
                    "label": "Desired Change",
                    "type": "multiline_text",
                    "required": True,
                    "placeholder": "Describe what to implement",
                },
                {
                    "key": "language",
                    "label": "Language/Framework",
                    "type": "short_text",
                    "required": False,
                    "placeholder": "e.g. React + Flask",
                },
                {
                    "key": "tests_constraint",
                    "label": "Test Constraint",
                    "type": "single_choice",
                    "required": True,
                    "default": "run_related_tests",
                    "options": [
                        {"value": "run_related_tests", "label": "Run related tests"},
                        {"value": "run_full_tests", "label": "Run full test suite"},
                        {"value": "no_tests", "label": "No tests required"},
                    ],
                },
                {
                    "key": "no_breaking_changes",
                    "label": "No Breaking Changes",
                    "type": "boolean",
                    "required": True,
                    "default": True,
                },
            ],
        }

    def build_spec(self, text: str, answers: dict) -> dict:
        clarified_request = (answers.get("clarified_request") or "").strip()
        success_criteria = _lines_to_list(answers.get("success_criteria", ""))
        hard_constraints = _lines_to_list(answers.get("hard_constraints", ""))
        output_preference = answers.get("output_preference", "direct")
        intent_frame = _build_intent_frame(answers)

        files_affected = _lines_to_list(answers.get("repo_area_or_paths", ""))
        desired_change = (answers.get("desired_change") or "").strip()
        change_type = answers.get("change_type", "feature")
        tests = answers.get("tests_constraint", "run_related_tests")

        acceptance = [
            "Implementation addresses the requested change.",
            "Output includes patch/diff or file-by-file code edits.",
            "Testing/lint expectations are clearly stated.",
            "No breaking change introduced unless explicitly requested.",
        ]
        acceptance = _merge_list_unique(acceptance, success_criteria)

        return {
            "task_type": "code",
            "objective": clarified_request or desired_change or text,
            "change_type": change_type,
            "files_affected": files_affected,
            "constraints": {
                "language": answers.get("language", ""),
                "tests": tests,
                "no_breaking_changes": bool(answers.get("no_breaking_changes", True)),
                "hard_constraints": hard_constraints,
            },
            "context": {"intent_frame": intent_frame},
            "must_include": ["Provide implementation details with concrete code edits."],
            "must_avoid": ["Ambiguous pseudo-code without actionable changes."],
            "output_format": {
                "sections": ["Plan", "Patch/Diff", "Verification"],
                "prefer_diff": True,
                "preference": output_preference,
            },
            "acceptance_criteria": acceptance,
        }

    def prompts(self, spec: dict, route: dict) -> list[dict]:
        prompt = render_code_prompt(spec)
        rows = []
        for ex in route.get("recommended_executors", []):
            rows.append(
                {
                    "executor": ex,
                    "prompt": prompt,
                    "notes": "Codex-style code-change prompt. Prefer patch/diff output.",
                }
            )
        return rows

    def validate(self, spec: dict, output: str) -> dict:
        return validate_code_output(spec, output)


def _lines_to_list(text: str) -> list[str]:
    lines = []
    for raw in (text or "").splitlines():
        s = raw.strip()
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
