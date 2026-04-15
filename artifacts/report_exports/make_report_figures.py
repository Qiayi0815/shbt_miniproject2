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

    for name in (
        "fig01_miou_dice_by_experiment.png",
        "fig02_person_metrics.png",
        "fig03_ablation_deltas.png",
        "fig04_deeplab_per_class_iou.png",
    ):
        print("Wrote:", FIG_DIR / name, "&", REPORT_FIG / name)


if __name__ == "__main__":
    main()
