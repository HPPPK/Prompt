from __future__ import annotations

import argparse
import json
import random
import re


def main():
    parser = argparse.ArgumentParser(description="Build supervised dataset for small orchestrator model.")
    parser.add_argument("--raw", default="backend/data/raw_corpus.jsonl")
    parser.add_argument("--out", default="backend/data/train_small_model.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    entities = _load_entities(args.raw)
    samples = []

    # Generic / analysis
    generic_templates = [
        "Tell me more about {x}",
        "Introduce {x} to a general reader",
        "请介绍一下{x}",
        "帮我分析{x}的商业模式",
        "Explain what {x} is",
    ]
    for x in _pick(entities, 220):
        t = random.choice(generic_templates).format(x=x)
        samples.append(_row(t, "generic", _lang(t), "direct", "analysis", "general"))

    # Weather / research
    cities = ["New York", "London", "Tokyo", "上海", "北京", "深圳", "广州"]
    weather_templates = [
        "What is the weather in {city} for next 3 days?",
        "{city}最近三天天气怎么样",
        "Give me a 7-day forecast for {city}",
        "帮我查一下{city}明天的天气",
    ]
    for city in cities:
        for _ in range(25):
            t = random.choice(weather_templates).format(city=city)
            samples.append(_row(t, "generic", _lang(t), "direct", "research", "weather_query"))

    # Writing
    writing_templates = [
        "帮我生成一篇{adj}的小红书帖子",
        "Write a {adj} social media post about {x}",
        "Help me generate a {adj} Xiaohongshu post about {x}",
        "请写一篇{adj}公众号文章，主题是{x}",
        "Create a {adj} marketing copy for {x}",
    ]
    adjs = ["新颖", "简洁", "专业", "有趣", "可信", "creative", "concise", "engaging"]
    for x in _pick(entities, 220):
        t = random.choice(writing_templates).format(x=x, adj=random.choice(adjs))
        samples.append(_row(t, "writing", _lang(t), "direct", "writing", "general"))
    writing_seed_queries = [
        "Help me generate a novel Xiaohongshu post",
        "Generate a Xiaohongshu post for my product",
        "Write a creative RedNote post",
        "写一篇新颖的小红书帖子",
        "帮我写一条小红书文案",
    ]
    for _ in range(140):
        t = random.choice(writing_seed_queries)
        samples.append(_row(t, "writing", _lang(t), "direct", "writing", "general"))

    # Email
    email_templates = [
        "帮我写一封邮件催供应商给发票",
        "Draft a follow-up email to vendor about invoice for {x}",
        "Write an email to client about project delay",
        "写一封邮件给同事，跟进{x}进度",
    ]
    for x in _pick(entities, 180):
        t = random.choice(email_templates).format(x=x)
        samples.append(_row(t, "email", _lang(t), "direct", "writing", "general"))

    # Code
    code_templates = [
        "帮我给前端加一个按钮并接API",
        "Fix bug in {x} and add tests",
        "Refactor backend endpoint for {x}",
        "Implement feature in React for {x}",
        "修复{x}相关报错并补测试",
    ]
    for x in _pick(entities, 220):
        t = random.choice(code_templates).format(x=_sanitize(x))
        samples.append(_row(t, "code", _lang(t), "direct", "planning", "general"))

    random.shuffle(samples)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved {len(samples)} labeled rows -> {args.out}")


def _row(
    text: str,
    task_type: str,
    language: str,
    output_preference: str,
    task_domain: str,
    query_intent: str,
) -> dict:
    return {
        "text": text.strip(),
        "task_type": task_type,
        "language": language,
        "output_preference": output_preference,
        "task_domain": task_domain,
        "query_intent": query_intent,
    }


def _load_entities(path: str) -> list[str]:
    entities = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = (row.get("text") or "").strip()
                if 2 <= len(text) <= 80:
                    entities.append(text)
    except FileNotFoundError:
        pass
    if not entities:
        entities = ["Tesla", "Amazon", "Apple", "OpenAI", "新能源汽车", "跨境电商", "供应链管理"]
    return list(dict.fromkeys(entities))


def _pick(items: list[str], n: int) -> list[str]:
    if not items:
        return []
    return [random.choice(items) for _ in range(n)]


def _lang(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text or "") else "en"


def _sanitize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    main()
