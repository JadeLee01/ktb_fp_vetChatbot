from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}
SOURCE_DIRS = [
    Path("152.반려동물_피부질환_데이터/01.데이터/1.Training/1_원천데이터_240422_add/반려견/피부/일반카메라"),
    Path("152.반려동물_피부질환_데이터/01.데이터/2.Validation/1_원천데이터_240422_add/반려견/피부/일반카메라"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an ImageFolder-style 7-class skin classifier dataset from 152 dog/general-camera data."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing 152.반려동물_피부질환_데이터",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/skin152_dog_general_classifier",
        help="Output dataset root. train/val/test directories will be created here.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio per class.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio per class.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to place files into the output dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic split seed.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before rebuilding.",
    )
    return parser.parse_args()


def normalize_disease_name(raw_name: str) -> str:
    if raw_name.endswith("_잔여"):
        raw_name = raw_name[: -len("_잔여")]
    return raw_name


def resolve_label(image_path: Path) -> str:
    parts = image_path.parts
    if "무증상" in parts:
        return "정상"
    if "유증상" in parts:
        disease_index = parts.index("유증상") + 1
        if disease_index >= len(parts):
            raise ValueError(f"Disease directory missing for symptomatic path: {image_path}")
        return normalize_disease_name(parts[disease_index])
    raise ValueError(f"Unsupported path outside 무증상/유증상: {image_path}")


def resolve_group_id(image_path: Path) -> str:
    return image_path.stem


def iter_skin_images(root: Path):
    for rel_dir in SOURCE_DIRS:
        source_dir = (root / rel_dir).resolve()
        if not source_dir.exists():
            raise SystemExit(f"Source directory not found: {source_dir}")
        for dirpath, _, filenames in os.walk(source_dir):
            base_dir = Path(dirpath)
            for filename in filenames:
                path = base_dir / filename
                if path.suffix.lower() not in VALID_SUFFIXES:
                    continue
                yield path


def assign_split(group_id: str, label: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha1(f"{seed}:{label}:{group_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def place_file(src: Path, dst: Path, copy_mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_mode == "symlink":
        os.symlink(src, dst)
        return
    if copy_mode == "hardlink":
        os.link(src, dst)
        return
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()

    repo_root = Path(args.root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("train_ratio and val_ratio must be > 0 and train_ratio + val_ratio must be < 1")

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val", "test"):
        ensure_dir(output_dir / split)

    counts = defaultdict(Counter)
    label_to_groups = defaultdict(set)
    num_samples = 0

    for image_path in iter_skin_images(repo_root):
        label = resolve_label(image_path)
        group_id = resolve_group_id(image_path)
        split = assign_split(group_id, label, args.seed, args.train_ratio, args.val_ratio)
        destination_dir = output_dir / split / label
        ensure_dir(destination_dir)
        destination_path = destination_dir / image_path.name
        place_file(image_path, destination_path, args.copy_mode)

        counts[split][label] += 1
        label_to_groups[label].add(group_id)
        num_samples += 1

    if num_samples == 0:
        raise SystemExit("No eligible dog/general-camera skin images found under 152 dataset.")

    manifest = {
        "source_dirs": [str((repo_root / rel_dir).resolve()) for rel_dir in SOURCE_DIRS],
        "output_dir": str(output_dir),
        "copy_mode": args.copy_mode,
        "seed": args.seed,
        "classes": sorted(label_to_groups.keys()),
        "counts": {split: dict(counter) for split, counter in counts.items()},
        "group_counts": {label: len(groups) for label, groups in sorted(label_to_groups.items())},
        "num_samples": num_samples,
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
