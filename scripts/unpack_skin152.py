from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


ZIP_DIRS = [
    Path("152.반려동물_피부질환_데이터/01.데이터/1.Training/1_원천데이터_240422_add"),
    Path("152.반려동물_피부질환_데이터/01.데이터/1.Training/2_라벨링데이터_240422_add"),
    Path("152.반려동물_피부질환_데이터/01.데이터/2.Validation/1_원천데이터_240422_add"),
    Path("152.반려동물_피부질환_데이터/01.데이터/2.Validation/2_라벨링데이터_240422_add"),
]


def extract_zip(zip_path: Path) -> None:
    target_dir = zip_path.parent
    print(f"[UNZIP] {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            zf.extract(member, path=target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpack 152 skin-disease dataset zips with Python zipfile.")
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing 152.반려동물_피부질환_데이터",
    )
    args = parser.parse_args()

    repo_root = Path(args.root).resolve()
    zip_paths: list[Path] = []
    for rel_dir in ZIP_DIRS:
        abs_dir = repo_root / rel_dir
        zip_paths.extend(sorted(abs_dir.glob("*.zip")))

    if not zip_paths:
        raise SystemExit("No zip files found for 152 dataset.")

    for zip_path in zip_paths:
        extract_zip(zip_path)

    print("[DONE] skin152 unpack finished")


if __name__ == "__main__":
    main()
