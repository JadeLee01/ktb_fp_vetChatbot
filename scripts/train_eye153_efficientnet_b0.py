from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import ImageFile
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights


# Some source images are slightly truncated; allow PIL to decode them instead of
# crashing a dataloader worker mid-epoch.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an EfficientNet-B0 eye disease classifier on the processed 153 eye dataset."
    )
    parser.add_argument(
        "--data-dir",
        default="datasets/eye153_general_classifier",
        help="ImageFolder dataset root containing train/val/test directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/eye153_efficientnet_b0",
        help="Directory where checkpoints and logs will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size for EfficientNet-B0.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing for cross entropy loss.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device. Use 'auto', 'cpu', or e.g. 'cuda:0'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="If > 0, limit each epoch/split pass to this many batches for smoke testing.",
    )
    parser.add_argument(
        "--final-best-name",
        default="eyeBest.pt",
        help="Extra filename to export the best checkpoint under inside output_dir.",
    )
    parser.add_argument(
        "--log-interval-batches",
        type=int,
        default=100,
        help="If > 0, print progress every N batches within each split pass.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, eval_tf


def build_datasets(data_dir: Path, image_size: int) -> tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    train_tf, eval_tf = build_transforms(image_size)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)
    return train_ds, val_ds, test_ds


def build_dataloader(dataset: datasets.ImageFolder, batch_size: int, workers: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    counts = Counter(dataset.targets)
    total = sum(counts.values())
    num_classes = len(dataset.classes)
    weights = []
    for class_index in range(num_classes):
        class_count = counts[class_index]
        weights.append(total / (num_classes * class_count))
    return torch.tensor(weights, dtype=torch.float32)


def create_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    max_batches: int,
    phase_name: str,
    epoch: int,
    log_interval_batches: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_batches = len(loader)

    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None and device.type == "cuda"
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_train:
                assert optimizer is not None
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == targets).sum().item()
        total_examples += images.size(0)

        if log_interval_batches > 0 and (
            batch_index == 1
            or batch_index % log_interval_batches == 0
            or batch_index == total_batches
            or (max_batches > 0 and batch_index == max_batches)
        ):
            progress = {
                "event": "batch_progress",
                "phase": phase_name,
                "epoch": epoch,
                "batch_index": batch_index,
                "total_batches": min(total_batches, max_batches) if max_batches > 0 else total_batches,
                "avg_loss": total_loss / total_examples,
                "avg_accuracy": total_correct / total_examples,
                "seen_examples": total_examples,
            }
            print(json.dumps(progress, ensure_ascii=False), flush=True)

        if max_batches > 0 and batch_index >= max_batches:
            break

    if total_examples == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
        }

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


@torch.inference_mode()
def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    correct = Counter()
    total = Counter()

    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)

        for target, pred in zip(targets.tolist(), preds.tolist()):
            total[target] += 1
            if target == pred:
                correct[target] += 1

        if max_batches > 0 and batch_index >= max_batches:
            break

    return {
        class_names[index]: (correct[index] / total[index] if total[index] else 0.0)
        for index in range(len(class_names))
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    metrics: dict[str, float],
    class_names: list[str],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "class_names": class_names,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / "train").exists():
        raise SystemExit(f"Dataset not found: {data_dir}")

    device = resolve_device(args.device)
    train_ds, val_ds, test_ds = build_datasets(data_dir, args.image_size)
    train_loader = build_dataloader(train_ds, args.batch_size, args.workers, shuffle=True, device=device)
    val_loader = build_dataloader(val_ds, args.batch_size, args.workers, shuffle=False, device=device)
    test_loader = build_dataloader(test_ds, args.batch_size, args.workers, shuffle=False, device=device)

    class_weights = compute_class_weights(train_ds).to(device)
    model = create_model(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    run_info = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "workers": args.workers,
        "image_size": args.image_size,
        "label_smoothing": args.label_smoothing,
        "max_batches": args.max_batches,
        "num_classes": len(train_ds.classes),
        "class_names": train_ds.classes,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "class_weights": class_weights.detach().cpu().tolist(),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(run_info, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "event": "run_start",
                "train_batches": len(train_loader),
                "val_batches": len(val_loader),
                "test_batches": len(test_loader),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_epoch = -1
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(json.dumps({"event": "epoch_start", "epoch": epoch}, ensure_ascii=False), flush=True)
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.max_batches,
            phase_name="train",
            epoch=epoch,
            log_interval_batches=args.log_interval_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            None,
            device,
            args.max_batches,
            phase_name="val",
            epoch=epoch,
            log_interval_batches=args.log_interval_batches,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        current = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": current_lr,
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(current)
        print(json.dumps(current, ensure_ascii=False), flush=True)

        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, current, train_ds.classes)
        (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

        if current["val_accuracy"] > best_val_acc:
            best_val_acc = current["val_accuracy"]
            best_epoch = epoch
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, current, train_ds.classes)

    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        None,
        None,
        device,
        args.max_batches,
        phase_name="test",
        epoch=best_epoch if best_epoch > 0 else args.epochs,
        log_interval_batches=args.log_interval_batches,
    )
    per_class_accuracy = evaluate_per_class(model, test_loader, device, train_ds.classes, args.max_batches)

    summary = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "per_class_test_accuracy": per_class_accuracy,
        "total_training_seconds": time.time() - start_time,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.final_best_name:
        shutil.copy2(output_dir / "best.pt", output_dir / args.final_best_name)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
