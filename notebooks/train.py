"""
Standalone training script for the Pascal VOC 2007 segmentation experiments.

This script is the practical retraining path when the notebook results are not
good enough: it preserves the notebook's checkpoint format so the notebook can
load the refreshed `.pt` files for analysis without retraining.

Usage examples:
    # smoke test a single experiment on a tiny subset
    /Users/mac/miniconda3/envs/shbt-seg/bin/python notebooks/train.py \\
        --force-retrain --experiments unet_small_ce_only --epochs-override 1 \\
        --train-subset 32 --val-subset 32

    # full retrain for selected experiments
    /Users/mac/miniconda3/envs/shbt-seg/bin/python notebooks/train.py \\
        --force-retrain --experiments unet_small_ce_only unet_small_dice_only deeplabv3_resnet50

    # continue an existing checkpoint for 10 more epochs on HPC
    python notebooks/train.py \\
        --resume-training --additional-epochs 10 --experiments unet_small_ce_only
"""

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCSegmentation
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
from torchvision.transforms import functional as TF


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Using device:", device)

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
NUM_CLASSES = len(VOC_CLASSES)
PERSON_CLASS_ID = VOC_CLASSES.index("person")
IGNORE_INDEX = 255
CHECKPOINT_VERSION = 2


@dataclass
class SegConfig:
    dataset_root: str = "./VOCtrainval_06-Nov-2007"
    image_size: Tuple[int, int] = (256, 256)
    batch_size: int = 4
    num_workers: int = 0
    train_subset: Optional[int] = None
    val_subset: Optional[int] = None
    epochs: int = 15
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    mixed_precision: bool = True
    grad_clip_norm: float = 1.0
    save_dir: str = "./artifacts"


CONFIG = SegConfig()


def resolve_script_relative_path(path_str: str, must_exist: bool = False) -> Path:
    raw = Path(path_str).expanduser()
    if raw.is_absolute():
        return raw

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = [
        (script_dir / raw).resolve(),
        (cwd / raw).resolve(),
        (script_dir.parent / raw).resolve(),
    ]

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    if must_exist:
        for candidate in unique_candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not resolve existing path for {path_str}")

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate
    return unique_candidates[0]


def resolve_voc_root(dataset_root: str) -> Path:
    raw = Path(dataset_root).expanduser()
    candidates = [resolve_script_relative_path(dataset_root)]
    raw_resolved = raw.resolve() if raw.is_absolute() else candidates[0]

    for candidate in candidates:
        jpeg_dir = candidate / "VOCdevkit" / "VOC2007" / "JPEGImages"
        split_dir = candidate / "VOCdevkit" / "VOC2007" / "ImageSets" / "Segmentation"
        if jpeg_dir.exists() and split_dir.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate Pascal VOC 2007 files from {raw_resolved}. "
        "Expected VOCdevkit/VOC2007/JPEGImages and ImageSets/Segmentation."
    )


class SegmentationPairTransform:
    def __init__(self, image_size=(256, 256), augment: bool = False):
        self.image_size = image_size
        self.augment = augment
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = image.convert("RGB")
        mask = mask.copy()

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() < 0.6:
                angle = random.uniform(-10.0, 10.0)
                max_dx = int(0.05 * image.size[0])
                max_dy = int(0.05 * image.size[1])
                translate = (random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy))
                scale = random.uniform(0.9, 1.1)
                shear = random.uniform(-5.0, 5.0)
                image = TF.affine(
                    image,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=shear,
                    interpolation=Image.BILINEAR,
                    fill=0,
                )
                mask = TF.affine(
                    mask,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=shear,
                    interpolation=Image.NEAREST,
                    fill=IGNORE_INDEX,
                )

            if random.random() < 0.35:
                image = TF.adjust_brightness(image, 1.0 + random.uniform(-0.2, 0.2))
            if random.random() < 0.35:
                image = TF.adjust_contrast(image, 1.0 + random.uniform(-0.2, 0.2))
            if random.random() < 0.2:
                image = TF.adjust_saturation(image, 1.0 + random.uniform(-0.15, 0.15))

        image = TF.resize(image, self.image_size, interpolation=Image.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=Image.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)

        mask_np = np.array(mask, dtype=np.int64)
        mask_tensor = torch.as_tensor(mask_np.copy(), dtype=torch.long)
        return image, mask_tensor


class PascalVOCSegmentationDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        image_set: str,
        image_size=(256, 256),
        augment: bool = False,
        max_samples: Optional[int] = None,
    ):
        resolved_root = resolve_voc_root(dataset_root)
        self.base = VOCSegmentation(
            root=str(resolved_root),
            year="2007",
            image_set=image_set,
            download=False,
        )
        self.transform = SegmentationPairTransform(image_size=image_size, augment=augment)
        self.indices = list(range(len(self.base)))
        if max_samples is not None:
            self.indices = self.indices[:max_samples]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        image, mask = self.base[base_idx]
        image, mask = self.transform(image, mask)
        image_id = Path(self.base.images[base_idx]).stem
        return {"image": image, "mask": mask, "image_id": image_id}


def build_dataloaders(config: SegConfig, augment_train: bool = False):
    train_dataset = PascalVOCSegmentationDataset(
        dataset_root=config.dataset_root,
        image_set="train",
        image_size=config.image_size,
        augment=augment_train,
        max_samples=config.train_subset,
    )
    val_dataset = PascalVOCSegmentationDataset(
        dataset_root=config.dataset_root,
        image_set="val",
        image_size=config.image_size,
        augment=False,
        max_samples=config.val_subset,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dataset, val_dataset, train_loader, val_loader


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 21, features=(32, 64, 128, 256)):
        super().__init__()
        self.down_blocks = nn.ModuleList()
        self.up_transpose = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        channels = in_channels
        for feature in features:
            self.down_blocks.append(DoubleConv(channels, feature))
            channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout=0.1)

        for feature in reversed(features):
            self.up_transpose.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.up_blocks.append(DoubleConv(feature * 2, feature))

        self.classifier = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx, up in enumerate(self.up_transpose):
            x = up(x)
            skip = skips[idx]
            if x.shape[-2:] != skip.shape[-2:]:
                x = TF.resize(x, size=skip.shape[-2:])
            x = torch.cat([skip, x], dim=1)
            x = self.up_blocks[idx](x)
        return self.classifier(x)


def build_torchvision_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    use_pretrained_backbone: bool = False,
):
    model_name = model_name.lower()
    kwargs = {"weights": None, "num_classes": num_classes}
    if use_pretrained_backbone:
        kwargs["weights_backbone"] = ResNet50_Weights.IMAGENET1K_V2
    else:
        kwargs["weights_backbone"] = None

    try:
        if model_name == "deeplabv3_resnet50":
            return deeplabv3_resnet50(**kwargs)
        if model_name == "fcn_resnet50":
            return fcn_resnet50(**kwargs)
    except Exception as exc:
        if use_pretrained_backbone:
            print(f"  [WARN] pretrained backbone unavailable for {model_name}: {exc}")
            kwargs["weights_backbone"] = None
            if model_name == "deeplabv3_resnet50":
                return deeplabv3_resnet50(**kwargs)
            if model_name == "fcn_resnet50":
                return fcn_resnet50(**kwargs)
        raise

    raise ValueError(f"Unsupported torchvision segmentation model: {model_name}")


def build_model(experiment_cfg: Dict):
    model_type = experiment_cfg["model_type"]
    if model_type == "unet":
        return UNet(num_classes=NUM_CLASSES, features=experiment_cfg.get("features", (32, 64, 128, 256)))
    return build_torchvision_model(
        model_type,
        num_classes=NUM_CLASSES,
        use_pretrained_backbone=experiment_cfg.get("use_pretrained_backbone", False),
    )


def extract_logits(model_output):
    return model_output["out"] if isinstance(model_output, dict) else model_output


def compute_class_weights(dataset: Dataset, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX):
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for idx in range(len(dataset)):
        mask = dataset[idx]["mask"].view(-1)
        valid_mask = mask[mask != ignore_index]
        if valid_mask.numel() == 0:
            continue
        counts += torch.bincount(valid_mask, minlength=num_classes).double()

    if counts.sum() == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    freqs = counts / counts.sum()
    weights = 1.0 / torch.sqrt(freqs.clamp_min(1e-6))
    weights = weights / weights.mean()
    return weights.float()


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    smooth: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
):
    valid_mask = target != ignore_index
    if not valid_mask.any():
        return logits.new_tensor(0.0)

    probs = torch.softmax(logits, dim=1)
    safe_target = target.clamp(0, num_classes - 1)
    target_one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    valid_mask = valid_mask.unsqueeze(1)
    probs = probs * valid_mask
    target_one_hot = target_one_hot * valid_mask

    dims = (0, 2, 3)
    intersection = (probs * target_one_hot).sum(dims)
    union = probs.sum(dims) + target_one_hot.sum(dims)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    present_classes = target_one_hot.sum(dims) > 0
    dice = dice[present_classes] if present_classes.any() else dice
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_name: str = "ce_dice",
    class_weights: Optional[torch.Tensor] = None,
    ignore_index: int = IGNORE_INDEX,
    label_smoothing: float = 0.0,
):
    loss_name = loss_name.lower()
    ce = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    if loss_name == "ce":
        return ce
    if loss_name == "dice":
        return multiclass_dice_loss(logits, target, ignore_index=ignore_index)
    if loss_name == "ce_dice":
        return ce + multiclass_dice_loss(logits, target, ignore_index=ignore_index)
    raise ValueError(f"Unknown loss: {loss_name}")


def confusion_matrix_from_predictions(pred, target, num_classes=NUM_CLASSES):
    pred = pred.detach().view(-1).cpu()
    target = target.detach().view(-1).cpu()
    valid = (target >= 0) & (target < num_classes)
    encoded = target[valid] * num_classes + pred[valid]
    return torch.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def metrics_from_confusion_matrix(hist, human_class_id=PERSON_CLASS_ID):
    hist = hist.float()
    tp = hist.diag()
    gt_count = hist.sum(dim=1)
    pred_count = hist.sum(dim=0)
    union = gt_count + pred_count - tp

    per_class_iou = torch.where(union > 0, tp / union, torch.nan)
    per_class_acc = torch.where(gt_count > 0, tp / gt_count, torch.nan)
    per_class_dice = torch.where(gt_count + pred_count > 0, 2 * tp / (gt_count + pred_count), torch.nan)

    return {
        "pixel_accuracy": (tp.sum() / hist.sum()).item() if hist.sum() > 0 else float("nan"),
        "mIoU": torch.nanmean(per_class_iou).item(),
        "mean_dice": torch.nanmean(per_class_dice).item(),
        "human_iou": per_class_iou[human_class_id].item(),
        "human_accuracy": per_class_acc[human_class_id].item(),
        "per_class_iou": {VOC_CLASSES[i]: float(per_class_iou[i]) for i in range(NUM_CLASSES)},
        "per_class_accuracy": {VOC_CLASSES[i]: float(per_class_acc[i]) for i in range(NUM_CLASSES)},
    }


def binary_hd95(pred_mask, true_mask):
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        return float("nan")
    pred_points = np.argwhere(pred_mask > 0)
    true_points = np.argwhere(true_mask > 0)
    if len(pred_points) == 0 or len(true_points) == 0:
        return float("nan")
    d_pred = cdist(pred_points, true_points).min(axis=1)
    d_true = cdist(true_points, pred_points).min(axis=1)
    return float(max(np.percentile(d_pred, 95), np.percentile(d_true, 95)))


def batch_hd95(pred, target, class_id=PERSON_CLASS_ID):
    scores = []
    for pred_mask, true_mask in zip(pred.cpu().numpy(), target.cpu().numpy()):
        scores.append(binary_hd95(pred_mask == class_id, true_mask == class_id))
    return scores


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    loss_name="ce_dice",
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    grad_clip_norm: Optional[float] = None,
):
    model.train()
    loss_meter = AverageMeter()
    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader, 1):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)

        autocast_enabled = scaler is not None and device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=autocast_enabled):
            logits = extract_logits(model(images))
            loss = segmentation_loss(
                logits,
                masks,
                loss_name=loss_name,
                class_weights=class_weights,
                label_smoothing=label_smoothing,
            )

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        print(f"  train [{batch_idx:3d}/{total_batches}]  loss={loss_meter.avg:.4f}", end="\r", flush=True)

    print()
    return {"train_loss": loss_meter.avg}


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    loss_name="ce_dice",
    compute_hd95=True,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
):
    model.eval()
    hist = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    loss_meter = AverageMeter()
    hd95_scores = []
    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader, 1):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = extract_logits(model(images))
        loss = segmentation_loss(
            logits,
            masks,
            loss_name=loss_name,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
        preds = logits.argmax(dim=1)
        hist += confusion_matrix_from_predictions(preds, masks)
        loss_meter.update(loss.item(), images.size(0))
        if compute_hd95:
            hd95_scores.extend(batch_hd95(preds, masks))

        print(f"  val   [{batch_idx:3d}/{total_batches}]  loss={loss_meter.avg:.4f}", end="\r", flush=True)

    print()
    metrics = metrics_from_confusion_matrix(hist)
    metrics["val_loss"] = loss_meter.avg
    metrics["hd95_person"] = float(np.nanmean(hd95_scores)) if hd95_scores else float("nan")
    return metrics, hist


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += value * n
        self.count += n

    @property
    def avg(self):
        return self.total / max(self.count, 1)


def run_experiment(
    name: str,
    config: SegConfig,
    experiment_cfg: Dict,
    force_retrain: bool = False,
    resume_training: bool = False,
    additional_epochs: int = 0,
):
    save_dir = resolve_script_relative_path(config.save_dir, must_exist=False)
    ckpt_path = save_dir / f"{name}.pt"

    print(f"\n{'=' * 62}")
    print(f"  Experiment : {name}")
    print(f"{'=' * 62}")

    ckpt = None
    start_epoch = 1
    history: List[Dict] = []
    best_state = None
    best_metrics = None
    best_hist = None
    best_miou = -float("inf")

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_version = ckpt.get("checkpoint_version", 0)
        is_legacy_resume_compatible = (
            resume_training
            and ckpt.get("model_state") is not None
            and ckpt.get("history") is not None
            and ckpt.get("metrics") is not None
        )
        if ckpt_version != CHECKPOINT_VERSION and not is_legacy_resume_compatible:
            print(
                f"  [STALE] checkpoint version {ckpt_version} != {CHECKPOINT_VERSION}; "
                "retraining with the improved pipeline."
            )
            ckpt = None
        elif ckpt_version != CHECKPOINT_VERSION and is_legacy_resume_compatible:
            print(
                f"  [LEGACY] checkpoint version {ckpt_version} detected; "
                "resuming from saved weights/history without optimizer state."
            )
        elif not force_retrain and not resume_training:
            print(f"  [LOADED]  checkpoint: {ckpt_path}")
            print(f"  Best mIoU  : {ckpt['metrics']['mIoU']:.4f}")
            print(f"{'=' * 62}\n")
            return ckpt["metrics"]

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        config,
        augment_train=experiment_cfg.get("augment", False),
    )
    class_weights = None
    if experiment_cfg.get("use_class_weights", True) and experiment_cfg.get("loss_name", "ce_dice") != "dice":
        class_weights = compute_class_weights(train_dataset).to(device)

    model = build_model(experiment_cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Model      : {experiment_cfg['model_type']}  ({num_params:,} params)")
    print(f"  Train/Val  : {len(train_dataset):,} / {len(val_dataset):,} samples")
    print(f"  Loss       : {experiment_cfg.get('loss_name', 'ce_dice')}")
    print(f"  Augment    : {experiment_cfg.get('augment', False)}")
    if experiment_cfg.get("use_pretrained_backbone", False):
        print("  Backbone   : ImageNet-pretrained if locally available")
    if class_weights is not None:
        print(
            f"  Class wts  : background={class_weights[0].item():.3f}  "
            f"person={class_weights[PERSON_CLASS_ID].item():.3f}"
        )
    print(f"  Device     : {device}")
    print(f"  Save dir   : {save_dir}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment_cfg.get("lr", config.learning_rate),
        weight_decay=config.weight_decay,
    )
    target_epochs = experiment_cfg.get("epochs", config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")

    if ckpt is not None:
        print(f"  [RESUME] checkpoint: {ckpt_path}")
        if ckpt.get("model_state") is not None:
            model.load_state_dict(ckpt["model_state"])
        history = list(ckpt.get("history", []))
        best_metrics = ckpt.get("metrics")
        best_hist = ckpt.get("confusion_matrix")
        best_state = ckpt.get("model_state")
        if best_metrics is not None:
            best_miou = float(best_metrics.get("mIoU", -float("inf")))
        start_epoch = len(history) + 1

        if ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        else:
            print("  [INFO] optimizer state missing; resuming from weights/history only.")

        if ckpt.get("scaler_state") is not None and device.type == "cuda":
            scaler.load_state_dict(ckpt["scaler_state"])

        if additional_epochs > 0:
            target_epochs = len(history) + additional_epochs
        elif target_epochs <= len(history):
            target_epochs = len(history) + 1

        print(f"  Resume     : epoch {start_epoch} -> {target_epochs}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(target_epochs, 1))
    if ckpt is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    exp_start = time.time()

    for epoch in range(start_epoch, target_epochs + 1):
        print(f"  Epoch {epoch}/{target_epochs}  |  lr={optimizer.param_groups[0]['lr']:.6f}")
        epoch_start = time.time()

        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            loss_name=experiment_cfg.get("loss_name", "ce_dice"),
            class_weights=class_weights,
            label_smoothing=experiment_cfg.get("label_smoothing", 0.0),
            grad_clip_norm=config.grad_clip_norm,
        )
        val_metrics, val_hist = evaluate_model(
            model,
            val_loader,
            device,
            loss_name=experiment_cfg.get("loss_name", "ce_dice"),
            class_weights=class_weights,
            label_smoothing=experiment_cfg.get("label_smoothing", 0.0),
        )
        scheduler.step()

        elapsed = time.time() - epoch_start
        epoch_stats = {
            "epoch": epoch,
            **train_stats,
            **{k: v for k, v in val_metrics.items() if not isinstance(v, dict)},
            "epoch_minutes": elapsed / 60.0,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_stats)

        is_best = epoch_stats["mIoU"] > best_miou
        best_tag = "  <- best" if is_best else ""
        print(
            f"    loss  train={epoch_stats['train_loss']:.4f}  val={epoch_stats['val_loss']:.4f}"
            f"  |  mIoU={epoch_stats['mIoU']:.4f}  px_acc={epoch_stats['pixel_accuracy']:.4f}"
            f"  |  human_iou={epoch_stats['human_iou']:.4f}"
            f"  |  {elapsed:.0f}s{best_tag}"
        )

        if is_best:
            best_miou = epoch_stats["mIoU"]
            best_metrics = val_metrics
            best_hist = val_hist
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    total_time = time.time() - exp_start
    print(f"\n  Done  best mIoU={best_miou:.4f}  |  total time: {total_time / 60:.1f} min")
    print(f"{'=' * 62}\n")

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_metrics = val_metrics
        best_hist = val_hist

    history_df = pd.DataFrame(history)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_state": best_state,
            "history": history_df.to_dict(orient="records"),
            "metrics": best_metrics,
            "confusion_matrix": best_hist,
            "class_weights": class_weights.detach().cpu() if class_weights is not None else None,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if device.type == "cuda" else None,
        },
        ckpt_path,
    )
    print(f"  [SAVED]  {ckpt_path}")
    return best_metrics


EXPERIMENTS = {
    "unet_small_no_aug": {
        "model_type": "unet",
        "features": (32, 64, 128, 256),
        "augment": False,
        "loss_name": "ce_dice",
        "epochs": 15,
        "lr": 5e-4,
        "use_class_weights": True,
        "label_smoothing": 0.0,
    },
    "unet_small_with_aug": {
        "model_type": "unet",
        "features": (32, 64, 128, 256),
        "augment": True,
        "loss_name": "ce_dice",
        "epochs": 15,
        "lr": 5e-4,
        "use_class_weights": True,
        "label_smoothing": 0.0,
    },
    "unet_small_ce_only": {
        "model_type": "unet",
        "features": (32, 64, 128, 256),
        "augment": True,
        "loss_name": "ce",
        "epochs": 15,
        "lr": 5e-4,
        "use_class_weights": True,
        "label_smoothing": 0.05,
    },
    "unet_small_dice_only": {
        "model_type": "unet",
        "features": (32, 64, 128, 256),
        "augment": True,
        "loss_name": "dice",
        "epochs": 15,
        "lr": 5e-4,
        "use_class_weights": False,
        "label_smoothing": 0.0,
    },
    "unet_wider_with_aug": {
        "model_type": "unet",
        "features": (64, 128, 256, 512),
        "augment": True,
        "loss_name": "ce_dice",
        "epochs": 18,
        "lr": 3e-4,
        "use_class_weights": True,
        "label_smoothing": 0.0,
    },
    "deeplabv3_resnet50": {
        "model_type": "deeplabv3_resnet50",
        "augment": True,
        "loss_name": "ce_dice",
        "epochs": 18,
        "lr": 3e-4,
        "use_class_weights": True,
        "use_pretrained_backbone": True,
        "label_smoothing": 0.0,
    },
}


DEFAULT_EXPERIMENTS = [
    "unet_small_no_aug",
    "unet_small_with_aug",
    "unet_small_ce_only",
    "unet_small_dice_only",
    "unet_wider_with_aug",
    "deeplabv3_resnet50",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Retrain Pascal VOC segmentation experiments.")
    parser.add_argument("--dataset-root", default=CONFIG.dataset_root)
    parser.add_argument("--save-dir", default=CONFIG.save_dir)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--additional-epochs", type=int, default=0)
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--val-subset", type=int, default=None)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    return parser.parse_args()


def main():
    args = parse_args()

    CONFIG.dataset_root = args.dataset_root
    CONFIG.save_dir = args.save_dir
    CONFIG.train_subset = args.train_subset
    CONFIG.val_subset = args.val_subset
    CONFIG.batch_size = args.batch_size

    selected_experiments = args.experiments or DEFAULT_EXPERIMENTS
    for name in selected_experiments:
        if name not in EXPERIMENTS:
            raise KeyError(f"Unknown experiment: {name}")

    if args.epochs_override is not None:
        for name in selected_experiments:
            EXPERIMENTS[name]["epochs"] = args.epochs_override

    resolved_save_dir = resolve_script_relative_path(CONFIG.save_dir, must_exist=False)
    resolved_save_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.save_dir = str(resolved_save_dir)
    results = {}
    for experiment_name in selected_experiments:
        results[experiment_name] = run_experiment(
            experiment_name,
            CONFIG,
            EXPERIMENTS[experiment_name],
            force_retrain=args.force_retrain,
            resume_training=args.resume_training,
            additional_epochs=args.additional_epochs,
        )

    print("\nAll requested experiments complete.")
    for name, metrics in results.items():
        print(
            f"- {name}: mIoU={metrics['mIoU']:.4f}, "
            f"person_iou={metrics['human_iou']:.4f}, "
            f"pixel_accuracy={metrics['pixel_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
