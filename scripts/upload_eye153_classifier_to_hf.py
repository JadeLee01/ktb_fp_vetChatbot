import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, create_repo

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


DEFAULT_RUN_DIR = Path("runs/eye153_efficientnet_b0_hardlink")
DEFAULT_REPO_ID = "20-team-daeng-ddang-ai/vet-chat"
DEFAULT_REPO_FOLDER_PATH = "eye_disease_classifier"
DEFAULT_REQUIRED_FILES = (
    "eyeBest.pt",
    "best.pt",
    "run_config.json",
    "history.json",
    "summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the trained eye disease classifier artifacts to Hugging Face Hub."
    )
    parser.add_argument(
        "--run-dir",
        default=str(DEFAULT_RUN_DIR),
        help="Training output directory containing checkpoints and metadata files.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Target Hugging Face repo id.",
    )
    parser.add_argument(
        "--repo-folder-path",
        default=DEFAULT_REPO_FOLDER_PATH,
        help="Destination folder path inside the Hugging Face repo.",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Hugging Face repository type.",
    )
    parser.add_argument(
        "--include-last",
        action="store_true",
        help="Also upload last.pt in addition to the best checkpoint artifacts.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload eye disease classifier artifacts",
        help="Commit message to use for the Hub upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare the staging folder and print what would be uploaded without uploading.",
    )
    return parser.parse_args()


def require_hf_token() -> str:
    if load_dotenv is not None:
        load_dotenv(override=False)
    else:
        load_dotenv_fallback(Path(".env"))
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Add it to .env or export it in the shell.")
    return token


def load_dotenv_fallback(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def ensure_run_dir(run_dir: Path) -> None:
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")


def collect_artifact_paths(run_dir: Path, include_last: bool) -> list[Path]:
    required_names = list(DEFAULT_REQUIRED_FILES)
    if include_last:
        required_names.append("last.pt")

    artifact_paths: list[Path] = []
    for name in required_names:
        path = run_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")
        artifact_paths.append(path)
    return artifact_paths


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_upload_manifest(run_dir: Path, artifact_paths: list[Path]) -> dict:
    run_config = load_json(run_dir / "run_config.json")
    summary = load_json(run_dir / "summary.json")
    manifest = {
        "model_name": "eye153_efficientnet_b0",
        "task": "dog eye disease image classification",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir.resolve()),
        "artifacts": [
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
        "num_classes": run_config["num_classes"],
        "class_names": run_config["class_names"],
        "image_size": run_config["image_size"],
        "train_size": run_config["train_size"],
        "val_size": run_config["val_size"],
        "test_size": run_config["test_size"],
        "best_epoch": summary["best_epoch"],
        "best_val_accuracy": summary["best_val_accuracy"],
        "test_accuracy": summary["test_accuracy"],
        "test_loss": summary["test_loss"],
    }
    return manifest


def build_readme_text(run_config: dict, summary: dict, artifact_paths: list[Path]) -> str:
    artifact_lines = "\n".join(f"- `{path.name}`" for path in artifact_paths)
    class_lines = "\n".join(f"- {name}" for name in run_config["class_names"])
    return f"""# Eye Disease Classifier

This folder contains the trained EfficientNet-B0 checkpoint and training metadata for the 153 eye disease classifier.

## Summary

- Task: dog eye disease image classification
- Architecture: EfficientNet-B0
- Num classes: {run_config["num_classes"]}
- Image size: {run_config["image_size"]}
- Train / Val / Test: {run_config["train_size"]} / {run_config["val_size"]} / {run_config["test_size"]}
- Best epoch: {summary["best_epoch"]}
- Best validation accuracy: {summary["best_val_accuracy"]:.6f}
- Test accuracy: {summary["test_accuracy"]:.6f}
- Test loss: {summary["test_loss"]:.6f}

## Included Files

{artifact_lines}
- `upload_manifest.json`
- `README.md`

## Class Names

{class_lines}
"""


def stage_artifacts(run_dir: Path, artifact_paths: list[Path]) -> Path:
    staging_dir = Path(tempfile.mkdtemp(prefix="eye153_hf_upload_"))

    for artifact_path in artifact_paths:
        shutil.copy2(artifact_path, staging_dir / artifact_path.name)

    run_config = load_json(run_dir / "run_config.json")
    summary = load_json(run_dir / "summary.json")
    manifest = build_upload_manifest(run_dir, artifact_paths)

    with (staging_dir / "upload_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    readme_text = build_readme_text(run_config, summary, artifact_paths)
    (staging_dir / "README.md").write_text(readme_text, encoding="utf-8")

    return staging_dir


def print_staged_files(staging_dir: Path) -> None:
    print("Prepared upload folder:")
    for path in sorted(staging_dir.iterdir()):
        print(f"- {path.name} ({path.stat().st_size} bytes)")


def upload_to_hf(
    staging_dir: Path,
    repo_id: str,
    repo_folder_path: str,
    repo_type: str,
    commit_message: str,
    token: str,
) -> None:
    create_repo(repo_id=repo_id, repo_type=repo_type, token=token, exist_ok=True)
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        path_in_repo=repo_folder_path,
        repo_type=repo_type,
        commit_message=commit_message,
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()

    ensure_run_dir(run_dir)
    artifact_paths = collect_artifact_paths(run_dir, include_last=args.include_last)
    staging_dir = stage_artifacts(run_dir, artifact_paths)

    try:
        print_staged_files(staging_dir)
        print(f"Target repo: {args.repo_id}")
        print(f"Target folder: {args.repo_folder_path}")

        if args.dry_run:
            print("Dry run enabled. Nothing was uploaded.")
            return

        token = require_hf_token()
        upload_to_hf(
            staging_dir=staging_dir,
            repo_id=args.repo_id,
            repo_folder_path=args.repo_folder_path,
            repo_type=args.repo_type,
            commit_message=args.commit_message,
            token=token,
        )
        print("Upload complete.")
        print(f"https://huggingface.co/{args.repo_id}/tree/main/{args.repo_folder_path}")
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
