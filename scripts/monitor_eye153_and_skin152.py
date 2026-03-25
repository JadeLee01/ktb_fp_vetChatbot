from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor eye153 training and skin152 dataset build progress.")
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=30,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--status-path",
        default="runs/monitor_152_153_status.json",
        help="Path to the latest JSON status file.",
    )
    parser.add_argument(
        "--log-path",
        default="runs/monitor_152_153.log",
        help="Path to the append-only monitor log.",
    )
    parser.add_argument(
        "--done-path",
        default="runs/monitor_152_153.done",
        help="Path written when both tasks are complete.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_cmd(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return result.stdout.strip()


def matching_processes(pattern: str) -> list[str]:
    output = run_cmd(["pgrep", "-af", pattern])
    if not output:
        return []
    lines = []
    for line in output.splitlines():
        if "pgrep -af" in line:
            continue
        lines.append(line)
    return lines


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_eye153(root: Path) -> dict:
    run_dir = root / "runs/eye153_efficientnet_b0"
    history_path = run_dir / "history.json"
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best.pt"
    eyebest_path = run_dir / "eyeBest.pt"
    run_config_path = run_dir / "run_config.json"

    processes = matching_processes("train_eye153_efficientnet_b0.py --data-dir datasets/eye153_general_classifier")
    history = load_json(history_path)
    summary = load_json(summary_path)
    run_config = load_json(run_config_path)

    last_epoch = None
    last_metrics = None
    if isinstance(history, list) and history:
        last_metrics = history[-1]
        last_epoch = last_metrics.get("epoch")

    return {
        "running": bool(processes),
        "processes": processes,
        "run_dir": str(run_dir),
        "run_config_exists": run_config_path.exists(),
        "history_exists": history_path.exists(),
        "summary_exists": summary_path.exists(),
        "best_exists": best_path.exists(),
        "eyeBest_exists": eyebest_path.exists(),
        "epochs": run_config.get("epochs") if isinstance(run_config, dict) else None,
        "last_epoch": last_epoch,
        "last_metrics": last_metrics,
        "summary": summary if isinstance(summary, dict) else None,
        "done": summary_path.exists() and eyebest_path.exists(),
    }


def collect_split_classes(dataset_dir: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        if split_dir.exists():
            result[split] = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        else:
            result[split] = []
    return result


def summarize_skin152(root: Path) -> dict:
    dataset_dir = root / "datasets/skin152_dog_general_classifier"
    manifest_path = dataset_dir / "manifest.json"
    processes = matching_processes("build_skin152_dog_general_classifier_dataset.py")
    manifest = load_json(manifest_path)
    split_classes = collect_split_classes(dataset_dir) if dataset_dir.exists() else {"train": [], "val": [], "test": []}

    return {
        "running": bool(processes),
        "processes": processes,
        "dataset_dir": str(dataset_dir),
        "manifest_exists": manifest_path.exists(),
        "split_classes": split_classes,
        "manifest": manifest if isinstance(manifest, dict) else None,
        "done": manifest_path.exists(),
    }


def append_log(log_path: Path, payload: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    status_path = (root / args.status_path).resolve()
    log_path = (root / args.log_path).resolve()
    done_path = (root / args.done_path).resolve()

    status_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        payload = {
            "timestamp_utc": utc_now(),
            "eye153": summarize_eye153(root),
            "skin152": summarize_skin152(root),
        }
        payload["all_done"] = payload["eye153"]["done"] and payload["skin152"]["done"]

        status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        append_log(log_path, payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)

        if payload["all_done"]:
            done_path.write_text(payload["timestamp_utc"] + "\n", encoding="utf-8")
            break

        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
