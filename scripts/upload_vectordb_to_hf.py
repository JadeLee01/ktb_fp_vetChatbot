import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sharing.embedding_utils import get_chroma_db_dir, get_embedding_model_id, should_use_e5_prefix


def write_manifest(db_dir: Path) -> Path:
    manifest_path = db_dir / "embedding_manifest.json"
    model_id = get_embedding_model_id()
    manifest = {
        "embedding_model_id": model_id,
        "use_e5_prefix": should_use_e5_prefix(model_id),
        "query_prefix": "query: " if should_use_e5_prefix(model_id) else "",
        "passage_prefix": "passage: " if should_use_e5_prefix(model_id) else "",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chroma_collection_name": "vet_qa_collection",
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest_path


def upload_chroma_db(local_folder_path: str, repo_id: str, repo_folder_path: str, repo_type: str):
    load_dotenv(override=False)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not set. Add it to .env or export it in the shell.")

    local_path = Path(local_folder_path).resolve()
    if not local_path.exists() or not local_path.is_dir():
        raise FileNotFoundError(f"Vector DB directory not found: {local_path}")

    manifest_path = write_manifest(local_path)
    print(f"Prepared manifest: {manifest_path}")

    create_repo(repo_id=repo_id, token=hf_token, repo_type=repo_type, exist_ok=True)
    api = HfApi(token=hf_token)

    print(f"🚀 Uploading vector DB: {local_path} -> {repo_id}/{repo_folder_path}")
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        path_in_repo=repo_folder_path,
        repo_type=repo_type,
        commit_message=f"Upload vector DB for {get_embedding_model_id()}",
    )
    print("✅ Upload complete")
    print(f"https://huggingface.co/{repo_id}/tree/main/{repo_folder_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Upload a Chroma vector DB folder to Hugging Face.")
    parser.add_argument(
        "--local-folder-path",
        default=get_chroma_db_dir(),
        help="Local Chroma DB directory to upload.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repo id, e.g. org/repo-name",
    )
    parser.add_argument(
        "--repo-folder-path",
        default="chroma_db_e5_base",
        help="Destination folder inside the HF repo.",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="HF repository type.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    upload_chroma_db(
        local_folder_path=args.local_folder_path,
        repo_id=args.repo_id,
        repo_folder_path=args.repo_folder_path,
        repo_type=args.repo_type,
    )
