from __future__ import annotations

import os
from dataclasses import dataclass


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "small_orchestrator")


@dataclass
class TaskTypePrediction:
    task_type: str
    confidence: float


_router_bundle = None
_slot_bundle = None


def is_small_model_ready() -> bool:
    return os.path.exists(os.path.join(MODEL_DIR, "router.joblib"))


def predict_task_type(text: str) -> TaskTypePrediction | None:
    bundle = _load_router_bundle()
    if not bundle:
        return None
    try:
        clf = bundle["classifier"]
        labels = bundle["labels"]
        probs = clf.predict_proba([text])[0]
        idx = int(probs.argmax())
        return TaskTypePrediction(task_type=labels[idx], confidence=float(probs[idx]))
    except Exception:
        return None


def predict_slots(text: str) -> dict:
    """Predict lightweight slot values using local small models.

    Returned keys are optional and can be merged with rule/LLM extraction.
    """
    bundle = _load_slot_bundle()
    if not bundle:
        return {}
    out = {}
    try:
        for key, model in (bundle.get("models") or {}).items():
            labels = (bundle.get("labels") or {}).get(key) or []
            probs = model.predict_proba([text])[0]
            idx = int(probs.argmax())
            if idx < len(labels):
                out[key] = labels[idx]
                out[f"{key}_confidence"] = float(probs[idx])
    except Exception:
        return {}
    return out


def _load_router_bundle():
    global _router_bundle
    if _router_bundle is not None:
        return _router_bundle
    path = os.path.join(MODEL_DIR, "router.joblib")
    if not os.path.exists(path):
        _router_bundle = {}
        return _router_bundle
    try:
        import joblib  # type: ignore

        _router_bundle = joblib.load(path)
    except Exception:
        _router_bundle = {}
    return _router_bundle


def _load_slot_bundle():
    global _slot_bundle
    if _slot_bundle is not None:
        return _slot_bundle
    path = os.path.join(MODEL_DIR, "slots.joblib")
    if not os.path.exists(path):
        _slot_bundle = {}
        return _slot_bundle
    try:
        import joblib  # type: ignore

        _slot_bundle = joblib.load(path)
    except Exception:
        _slot_bundle = {}
    return _slot_bundle
