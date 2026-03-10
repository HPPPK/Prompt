from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request


WIKI_API = "https://en.wikipedia.org/w/api.php"
ZH_WIKI_API = "https://zh.wikipedia.org/w/api.php"
STACK_API = "https://api.stackexchange.com/2.3/questions"


def fetch_wikipedia_titles(api_url: str, query: str, limit: int = 30) -> list[str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "format": "json",
    }
    url = f"{api_url}?{urllib.parse.urlencode(params)}"
    data = _get_json(url)
    return [item.get("title", "").strip() for item in data.get("query", {}).get("search", []) if item.get("title")]


def fetch_stackexchange_titles(tag: str, pagesize: int = 50) -> list[str]:
    params = {
        "order": "desc",
        "sort": "votes",
        "tagged": tag,
        "site": "stackoverflow",
        "pagesize": str(pagesize),
    }
    url = f"{STACK_API}?{urllib.parse.urlencode(params)}"
    data = _get_json(url)
    return [item.get("title", "").strip() for item in data.get("items", []) if item.get("title")]


def main():
    parser = argparse.ArgumentParser(description="Collect legally accessible seed corpus via public APIs.")
    parser.add_argument("--out", default="backend/data/raw_corpus.jsonl")
    parser.add_argument("--max-per-source", type=int, default=80)
    args = parser.parse_args()

    records = []
    sources = [
        ("en_wiki", lambda: fetch_wikipedia_titles(WIKI_API, "technology company", args.max_per_source)),
        ("en_wiki", lambda: fetch_wikipedia_titles(WIKI_API, "business model", args.max_per_source)),
        ("zh_wiki", lambda: fetch_wikipedia_titles(ZH_WIKI_API, "公司 商业模式", args.max_per_source)),
        ("zh_wiki", lambda: fetch_wikipedia_titles(ZH_WIKI_API, "天气 城市", args.max_per_source)),
        ("stack_overflow", lambda: fetch_stackexchange_titles("python", min(args.max_per_source, 100))),
        ("stack_overflow", lambda: fetch_stackexchange_titles("reactjs", min(args.max_per_source, 100))),
    ]

    for source, fn in sources:
        try:
            titles = fn()
        except Exception as exc:
            print(f"[warn] source={source} failed: {exc}")
            titles = []
        for title in titles:
            records.append({
                "source": source,
                "text": title,
                "license_note": "Public API content; check platform ToS before redistribution.",
            })
        time.sleep(0.4)

    # de-dup
    seen = set()
    dedup = []
    for row in records:
        key = row["text"].strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(row)

    with open(args.out, "w", encoding="utf-8") as f:
        for row in dedup:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved {len(dedup)} rows to {args.out}")


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "ai-prompt-manager/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data)


if __name__ == "__main__":
    main()
