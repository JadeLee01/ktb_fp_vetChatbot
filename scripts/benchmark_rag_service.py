import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_BASELINE = {
    "tag": "baseline_ko_sroberta",
    "success_rate": 1.0,
    "avg_latency_ms": 1987.72,
    "p95_latency_ms": 2560.69,
    "citation_keyword_hit_rate": 0.125,
    "answer_keyword_hit_rate": 1.0,
    "avg_citation_count": 3.0,
}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def contains_keywords(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return all(keyword.lower() in lowered for keyword in keywords)


def load_eval_set(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def http_json(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def compare_with_baseline(metrics: dict, baseline: dict) -> dict:
    comparison = {}
    for key, baseline_value in baseline.items():
        if key == "tag" or key not in metrics or not isinstance(baseline_value, (int, float)):
            continue
        comparison[key] = {
            "baseline": baseline_value,
            "candidate": metrics[key],
            "delta": round(metrics[key] - baseline_value, 6),
        }
    return comparison


def run_benchmark(base_url: str, eval_path: Path, output_path: Path, baseline_path: Path | None):
    health = http_json(f"{base_url}/health")
    eval_set = load_eval_set(eval_path)

    results = []
    latencies = []
    success_count = 0
    citation_hits = 0
    answer_hits = 0
    citation_counts = []

    for item in eval_set:
        started = time.perf_counter()
        try:
            response = http_json(
                f"{base_url}/chat",
                method="POST",
                payload={"question": item["question"]},
            )
            latency_ms = (time.perf_counter() - started) * 1000
            latencies.append(latency_ms)
            success = True
            success_count += 1
        except urllib.error.URLError as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            success = False
            response = {"error": str(exc)}

        citations = response.get("citations", []) if success else []
        answer = response.get("answer", "") if success else ""
        citation_text = "\n".join(
            f"{citation.get('title', '')}\n{citation.get('snippet', '')}" for citation in citations
        )
        citation_count = len(citations)
        citation_counts.append(citation_count)

        citation_hit = contains_keywords(citation_text, item.get("citation_keywords", []))
        answer_hit = contains_keywords(answer, item.get("answer_keywords", []))
        citation_hits += int(citation_hit)
        answer_hits += int(answer_hit)

        results.append(
            {
                "question": item["question"],
                "latency_ms": round(latency_ms, 2),
                "success": success,
                "citation_hit": citation_hit,
                "answer_hit": answer_hit,
                "citation_count": citation_count,
                "response": response,
            }
        )

    metrics = {
        "tag": output_path.stem,
        "sample_count": len(eval_set),
        "success_rate": round(success_count / len(eval_set), 6) if eval_set else 0.0,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p50_latency_ms": round(percentile(latencies, 0.50), 2) if latencies else 0.0,
        "p95_latency_ms": round(percentile(latencies, 0.95), 2) if latencies else 0.0,
        "p99_latency_ms": round(percentile(latencies, 0.99), 2) if latencies else 0.0,
        "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
        "citation_keyword_hit_rate": round(citation_hits / len(eval_set), 6) if eval_set else 0.0,
        "answer_keyword_hit_rate": round(answer_hits / len(eval_set), 6) if eval_set else 0.0,
        "avg_citation_count": round(statistics.mean(citation_counts), 2) if citation_counts else 0.0,
    }

    baseline = DEFAULT_BASELINE
    if baseline_path:
        with baseline_path.open("r", encoding="utf-8") as handle:
            baseline = json.load(handle)

    report = {
        "health": health,
        "metrics": metrics,
        "baseline_comparison": compare_with_baseline(metrics, baseline),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved benchmark report to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark the local RAG chat service and compare against baseline metrics.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--eval-file", default="temp/rag_benchmark_eval.json")
    parser.add_argument("--output-file", default="temp/benchmark_e5_base.json")
    parser.add_argument("--baseline-file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        base_url=args.base_url.rstrip("/"),
        eval_path=Path(args.eval_file),
        output_path=Path(args.output_file),
        baseline_path=Path(args.baseline_file) if args.baseline_file else None,
    )
