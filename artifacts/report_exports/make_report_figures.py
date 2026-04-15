"""Generate PNG figures for Mini-Project 2 report from CSV exports."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# Also mirror into project root so pandoc/pdflatex resolves paths reliably.
FIG_DIR = HERE / "figures"
REPORT_FIG = HERE.parent.parent / "report_figures"
for d in (FIG_DIR, REPORT_FIG):
    d.mkdir(exist_ok=True)


def main() -> None:
    exp = pd.read_csv(HERE / "experiment_summary.csv")
    abl = pd.read_csv(HERE / "ablation_summary.csv")
    per = pd.read_csv(HERE / "per_class_iou.csv")

    # --- Figure 1: mIoU and mean Dice by experiment ---
    exp_sorted = exp.sort_values("mIoU", ascending=False)
    x = np.arange(len(exp_sorted))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - w / 2, exp_sorted["mIoU"], width=w, label="mIoU", color="#2c6cb0")
    ax.bar(x + w / 2, exp_sorted["mean_dice"], width=w, label="Mean Dice", color="#b85c38")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_sorted["experiment"], rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(exp_sorted["mIoU"].max(), exp_sorted["mean_dice"].max()) * 1.25)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Global overlap metrics by experiment")
    fig.tight_layout()
    for d in (FIG_DIR, REPORT_FIG):
        fig.savefig(d / "fig01_miou_dice_by_experiment.png", dpi=200)
    plt.close(fig)

    # --- Figure 2: Person-centric metrics ---
    fig, ax1 = plt.subplots(figsize=(7.2, 3.4))
    ax2 = ax1.twinx()
    x = np.arange(len(exp_sorted))
    labs = list(exp_sorted["experiment"])
    ax1.bar(x - w / 2, exp_sorted["human_iou"], width=w, label="Person IoU", color="#1a6b3a")
    ax1.bar(x + w / 2, exp_sorted["human_accuracy"], width=w, label="Person pixel acc.", color="#6b8e23")
    ax2.plot(x, exp_sorted["hd95_person"], "o-", color="#5c2d91", ms=5, lw=1.2, label="HD95 person")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labs, rotation=22, ha="right", fontsize=8)
    ax1.set_ylabel("IoU / accuracy")
    ax2.set_ylabel("HD95 (person, lower better)")
    ax1.set_ylim(0, max(exp_sorted["human_iou"].max(), exp_sorted["human_accuracy"].max()) * 1.15)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7)
    ax1.set_title("Person class: IoU, accuracy, and boundary distance")
    fig.tight_layout()
    for d in (FIG_DIR, REPORT_FIG):
        fig.savefig(d / "fig02_person_metrics.png", dpi=200)
    plt.close(fig)

    # --- Figure 3: Ablation delta bars (mIoU and person IoU) ---
    labels = abl["ablation"].tolist()
    d_miou = abl["delta_mIoU"].astype(float).tolist()
    d_h = abl["delta_human_iou"].astype(float).tolist()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(labels))
    h = 0.35
    ax.barh(y - h / 2, d_miou, height=h, label="Delta mIoU", color="#2c6cb0")
    ax.barh(y + h / 2, d_h, height=h, label="Delta person IoU", color="#b85c38")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Change vs baseline (comparison - baseline)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Ablation summary: paired comparisons from the experiment grid")
    fig.tight_layout()
    for d in (FIG_DIR, REPORT_FIG):
        fig.savefig(d / "fig03_ablation_deltas.png", dpi=200)
    plt.close(fig)

    # --- Figure 4: DeepLab per-class IoU (non-background) ---
    s = (
        per.loc[per["experiment"] == "deeplabv3_resnet50"]
        .drop(columns=["experiment"])
        .iloc[0]
    )
    s = s.drop(labels=["background"])
    s = s[s.astype(float) > 1e-6].sort_values()
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.barh(s.index.tolist(), s.astype(float).values, color="#2c6cb0")
    ax.set_xlabel("IoU")
    ax.set_title("DeepLabV3-ResNet50: per-class IoU (excluding background)")
    ax.set_xlim(0, float(s.max()) * 1.15)
    fig.tight_layout()
    for d in (FIG_DIR, REPORT_FIG):
        fig.savefig(d / "fig04_deeplab_per_class_iou.png", dpi=200)
    plt.close(fig)

    # --- Figure 5: Per-class IoU heatmap (DeepLab vs representative U-Nets) ---
    class_cols = [c for c in per.columns if c != "experiment"]
    models = ["deeplabv3_resnet50", "unet_small_with_aug", "unet_small_dice_only"]
    mat = np.stack(
        [
            per.loc[per["experiment"] == m, class_cols].iloc[0].astype(float).values
            for m in models
        ],
        axis=0,
    )  # shape (3, 21)
    fig, ax = plt.subplots(figsize=(6.2, 7.0))
    im = ax.imshow(mat.T, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.2, float(mat.max()) * 1.05))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(["DeepLab\nR50", "U-Net\n+aug", "U-Net\nDice"], fontsize=8)
    ax.set_yticks(range(len(class_cols)))
    ax.set_yticklabels(class_cols, fontsize=7)
    ax.set_title("Per-class IoU: DeepLab vs U-Net (aug) vs U-Net (Dice only)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("IoU", fontsize=8)
    fig.tight_layout()
    for d in (FIG_DIR, REPORT_FIG):
        fig.savefig(d / "fig05_perclass_heatmap_models.png", dpi=200)
    plt.close(fig)

    # --- Figure 6: Pixel accuracy vs mIoU (decoupling under imbalance) ---
    short = {
        "deeplabv3_resnet50": "DL-R50",
        "unet_small_dice_only": "UN-Dice",
        "unet_small_ce_only": "UN-CE",
        "unet_small_no_aug": "UN-noaug",
        "unet_wider_with_aug": "UN-wide",
        "unet_small_with_aug": "UN-aug",
    }
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    for _, row in exp.iterrows():
        ax.scatter(row["mIoU"], row["pixel_accuracy"], s=55, alpha=0.85)
        lab = short.get(row["experiment"], row["experiment"][:10])
        ax.annotate(lab, (row["mIoU"], row["pixel_accuracy"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mIoU")
    ax.set_ylabel("Pixel accuracy")
    ax.set_title("Pixel accuracy versus mIoU by experiment")
    fig.tight_layout()
    for d in (FIG_DIR, REPORT_FIG):
        fig.savefig(d / "fig06_pixelacc_vs_miou.png", dpi=200)
    plt.close(fig)

    for name in (
        "fig01_miou_dice_by_experiment.png",
        "fig02_person_metrics.png",
        "fig03_ablation_deltas.png",
        "fig04_deeplab_per_class_iou.png",
        "fig05_perclass_heatmap_models.png",
        "fig06_pixelacc_vs_miou.png",
    ):
        print("Wrote:", FIG_DIR / name, "&", REPORT_FIG / name)


if __name__ == "__main__":
    main()
