from __future__ import annotations

import argparse
import json
import os
from typing import Any


def main():
    parser = argparse.ArgumentParser(description="Train lightweight local models for routing + slot priors.")
    parser.add_argument("--train", default="backend/data/train_small_model.jsonl")
    parser.add_argument("--out-dir", default="backend/models/small_orchestrator")
    args = parser.parse_args()

    rows = _load_rows(args.train)
    if not rows:
        raise RuntimeError(f"empty training data: {args.train}")

    X = [r["text"] for r in rows]
    y_task = [r["task_type"] for r in rows]
    y_lang = [r.get("language", "en") for r in rows]
    y_pref = [r.get("output_preference", "direct") for r in rows]
    y_domain = [r.get("task_domain", "analysis") for r in rows]
    y_query_intent = [r.get("query_intent", "general") for r in rows]

    try:
        import joblib  # type: ignore
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.dummy import DummyClassifier  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.metrics import classification_report  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Training dependencies missing. Install backend/training/requirements.txt first."
        ) from exc

    os.makedirs(args.out_dir, exist_ok=True)

    # Router model
    x_tr, x_te, y_tr, y_te = train_test_split(X, y_task, test_size=0.2, random_state=42, stratify=y_task)
    router_clf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )
    router_clf.fit(x_tr, y_tr)
    pred = router_clf.predict(x_te)
    print("\n[router report]")
    print(classification_report(y_te, pred))

    task_labels = sorted(set(y_task))
    router_path = os.path.join(args.out_dir, "router.joblib")
    joblib.dump({"classifier": router_clf, "labels": task_labels}, router_path)

    # Slot prior models
    slot_models: dict[str, Any] = {}
    slot_labels: dict[str, list[str]] = {}
    for key, y in [
        ("language", y_lang),
        ("output_preference", y_pref),
        ("task_domain", y_domain),
        ("query_intent", y_query_intent),
    ]:
        unique_labels = sorted(set(y))
        if len(unique_labels) == 1:
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X, y)
            slot_models[key] = dummy
            slot_labels[key] = unique_labels
            continue

        clf = Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)),
                ("clf", LogisticRegression(max_iter=1500)),
            ]
        )
        clf.fit(X, y)
        slot_models[key] = clf
        slot_labels[key] = unique_labels

    slots_path = os.path.join(args.out_dir, "slots.joblib")
    joblib.dump({"models": slot_models, "labels": slot_labels}, slots_path)

    print(f"saved router model -> {router_path}")
    print(f"saved slot models  -> {slots_path}")


def _load_rows(path: str) -> list[dict]:
    rows = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not row.get("text") or not row.get("task_type"):
                continue
            rows.append(row)
    if bad:
        print(f"[warn] skipped {bad} malformed rows from {path}")
    return rows


if __name__ == "__main__":
    main()
