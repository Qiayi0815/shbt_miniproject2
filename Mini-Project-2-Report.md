---
title: "Mini-Project 2: Semantic Segmentation on Pascal VOC 2007"
author: "Qiayi Zha"
date: "April 2026"
subtitle: "SHBT -- AI in Medicine (course project)"
documentclass: article
fontsize: 10pt
geometry: "margin=0.75in"
linestretch: 1
toc: false
header-includes:
  - \usepackage{setspace}
  - \singlespacing
  - \usepackage{caption}
  - \captionsetup{font=small,skip=4pt}
  - \usepackage{graphicx}
  - \usepackage{booktabs}
---

# Introduction

Semantic segmentation assigns each pixel to one of a fixed set of semantic classes. This work reports experiments on the Pascal VOC 2007 segmentation benchmark (twenty foreground classes plus background). Following the course protocol, the official validation split is used as the held-out test set. We compare a custom U-Net to DeepLabV3-ResNet50 from torchvision and summarize ablations on the U-Net recipe plus an architecture-level comparison. All quantitative tables in Results were generated from the same exported evaluation CSVs as the companion notebook. The Methods section below is organized as a **single pipeline description**, a **transparent hyperparameter registry**, and an explicit **fairness critique** so readers can judge how strongly each headline comparison supports a causal claim.

# Methods

## End-to-end pipeline (concise flow)

**Data in.** Pascal VOC 2007 semantic masks are loaded with `torchvision.datasets.VOCSegmentation` under `VOCdevkit/VOC2007/` (209 training and 213 validation pairs in our logs). Each sample provides RGB JPEGs and a single-channel mask with labels in $\{0,\ldots,20\}$ (zero is background).

**Preprocess and augment.** Training pairs optionally pass through stochastic augmentation (geometry on image and mask jointly; photometric jitter on RGB only). Image and mask are then resized to $256 \times 256$: **bilinear** for images, **nearest-neighbor** for masks. RGB tensors receive ImageNet normalization; masks become `int64` then `long` for dense supervision. Affine padding uses ignore index **255** where supported.

**Model and loss.** The network outputs twenty-one logits per spatial location. The training objective is selected per experiment (joint cross-entropy with soft Dice, CE-only with label smoothing, or Dice-only); CE-type runs can use **class reweighting** from training-mask frequencies.

**Optimize and select.** We optimize with weight decay $10^{-4}$, mixed precision when available, gradient clipping (norm one), and **cosine decay** of the learning rate over the scheduled epoch horizon implemented in the repository. The checkpoint **maximizing validation mIoU** is retained; training may **stop early** if validation mIoU fails to improve for a patience window defined in the notebook.

**Evaluate.** All reported metrics use the **same** validation dataloader **without** augmentation.

## Architectures

**U-Net:** Encoder-decoder with skip connections; narrow (`32,64,128,256`) versus wide (`64,128,256,512`) channel schedules.

**DeepLabV3-ResNet50:** ResNet-50 + ASPP + dense head. Runs in our registry enable **ImageNet-pretrained backbone weights** when the environment can download or resolve them, which is a deliberate advantage relative to the **from-scratch** U-Net encoder.

**SAM/SAM2:** Optional prompted baseline in the notebook only; not part of the quantitative tables.

## Hyperparameter registry (reproducibility)

The table matches the **`EXPERIMENTS` dictionary** in the companion notebook at export time. **Nominal epoch budget** is 30 for each row; **actual** parameter updates can be fewer if early stopping fires. Initial learning rates are **pre-decay** values; optimization uses AdamW with cosine annealing over the remaining budget per implementation. Batch size is four unless locally overridden.

**Configuration summary.** Abbreviations: CE+Dice = sum of cross-entropy and soft Dice (implementation detail in code); ``Class reweight'' applies inverse-frequency style weights for CE components when enabled.

| Experiment | Epoch budget | Initial LR | Training loss | Train augment | Class reweight (CE path) | Label smoothing (CE) | ImageNet-pretrained backbone |
|:--|--:|--:|:--|:--|:--|:--|:--|
| unet_small_no_aug | 30 | $5\times10^{-4}$ | CE + Dice | no | yes | 0 | no |
| unet_small_with_aug | 30 | $5\times10^{-4}$ | CE + Dice | yes | yes | 0 | no |
| unet_small_ce_only | 30 | $5\times10^{-4}$ | CE only | yes | yes | 0.05 | no |
| unet_small_dice_only | 30 | $5\times10^{-4}$ | Dice only | yes | no | 0 | no |
| unet_wider_with_aug | 30 | $3\times10^{-4}$ | CE + Dice | yes | yes | 0 | no |
| deeplabv3_resnet50 | 30 | $2\times10^{-4}$ | CE + Dice | yes | yes | 0 | yes |

**Reproducibility artifacts.** Exact seeds, optimizer $\beta$ values, dataloader worker counts, and early-stopping counters are defined in `notebooks/train.py` and the training notebook cells. **Checkpoints and raw VOC archives are not stored in git** (see `.gitignore`); rerunning the notebook or `train.py` after downloading VOC reproduces weights up to hardware nondeterminism.

## Comparator fairness (critical evaluation)

**What is legitimately aligned:** identical split policy, evaluation resolution, metric implementations, and (by default) batch size and global weight decay from `SegConfig`.

**What is not a single-factor study:** (1) **DeepLab versus augmented narrow U-Net** differs in **architecture, parameter count, receptive field, and ImageNet backbone initialization**, not only ``which head''; (2) **initial learning rates are not uniform** (DeepLab $2\times10^{-4}$, most small U-Nets $5\times10^{-4}$, wider U-Net $3\times10^{-4}$), so the **width** row confounds **capacity with optimizer tuning**; (3) **loss recipes differ** (CE+Dice versus CE-only with smoothing versus Dice-only with class weights disabled); (4) **early stopping** means two runs with the same nominal budget can see different numbers of updates. Pairwise rows in Table 2 remain informative where **architecture and LR are shared** (augmentation toggle on the small U-Net; CE-only versus Dice-only on the small U-Net), but those comparisons still inherit loss-form differences other than the intended CE versus Dice contrast.

## Ablation protocol (intent versus confounds)

**Augmentation:** `unet_small_no_aug` versus `unet_small_with_aug` (same loss CE+Dice, same nominal LR schedule family).

**Loss (CE versus Dice):** `unet_small_ce_only` versus `unet_small_dice_only` --- **both use training augmentation** in the registry; the intended factor is the **objective**, acknowledging auxiliary differences (label smoothing and class reweighting on CE-only, disabled reweighting on Dice-only).

**Width:** `unet_small_with_aug` versus `unet_wider_with_aug` --- same augmentation and loss form, **but different initial LR** (see table); interpret as **``wider + retuned LR''** rather than width alone.

**Architecture:** `unet_small_with_aug` versus `deeplabv3_resnet50` --- interpret as **strong engineering baseline comparison** under the caveats above.

## Evaluation metrics

Mean intersection-over-union (mIoU) and mean Dice (class support handled as in code); pixel accuracy; per-class IoU; person-class IoU and person pixel accuracy; **HD95** (95th percentile Hausdorff distance) on binary person masks, lower is better. Qualitative mosaics and best/worst person cases are prepared in the notebook.

## Software

https://github.com/Qiayi0815/shbt_miniproject2 --- primary paths: `notebooks/mini_project_2_pascal_voc_segmentation (1) (4).ipynb`, `notebooks/train.py`, `artifacts/report_exports/`.

# Results

All metrics below are computed on the Pascal VOC 2007 validation masks used as test data. Figures and tables appear in reading order; each caption states the **main takeaway** without requiring the body text.

![**DeepLab leads global overlap among all runs.** Bar chart of mean IoU (blue) and mean Dice (orange) for every experiment, sorted by decreasing mIoU in the plotting script. DeepLabV3-ResNet50 is tallest on both axes; U-Net variants cluster at lower mIoU with heterogeneous Dice. Use this figure as a one-glance ranking of global segmentation quality under the registry in Methods.](./report_figures/fig01_miou_dice_by_experiment.png){width=95%}

Figure 1 supports the headline that DeepLab attains the highest mIoU and mean Dice in Table 1 before reading numeric cells.

![**High pixel accuracy does not imply high mIoU.** Scatter of overall pixel accuracy (vertical) versus mIoU (horizontal) for each experiment; labels abbreviate model IDs (DL-R50 = DeepLabV3-ResNet50; UN-* = U-Net variants). Several U-Nets lie **above** DeepLab on pixel accuracy while remaining **left** on mIoU, illustrating majority-class agreement without balanced per-class overlap.](./report_figures/fig06_pixelacc_vs_miou.png){width=62%}

Figure 2 visualizes the decoupling that Table 1 quantifies.

**Table 1.** Validation metrics on the held-out split. Pixel-type accuracies are in $[0,1]$; HD95 (person) is a distance (lower better). Val loss scales differ between Dice-only and CE-type objectives.

| Experiment | Pixel acc. | mIoU | Mean Dice | HD95 (person) | Person IoU | Person acc. | Val loss |
|:--|--:|--:|--:|--:|--:|--:|--:|
| deeplabv3_resnet50 | 0.653 | **0.0655** | **0.0978** | 97.91 | 0.024 | 0.027 | 2.57 |
| unet_small_dice_only | 0.671 | 0.0450 | 0.0587 | 89.86 | **0.210** | **0.479** | **0.82** |
| unet_small_ce_only | **0.707** | 0.0450 | 0.0585 | **88.38** | 0.132 | 0.323 | 2.57 |
| unet_small_no_aug | 0.656 | 0.0440 | 0.0591 | 95.89 | 0.123 | 0.374 | 3.41 |
| unet_wider_with_aug | **0.714** | 0.0423 | 0.0538 | 99.32 | 0.137 | 0.285 | 3.40 |
| unet_small_with_aug | 0.714 | 0.0418 | 0.0530 | 88.74 | 0.135 | 0.270 | 3.44 |

Table 1 states the same rankings numerically. DeepLab leads mIoU and mean Dice. The Dice-only U-Net attains the largest person IoU (0.210) and person pixel accuracy (0.479), whereas the cross-entropy-only U-Net attains the lowest HD95 among U-Net variants (88.38). DeepLab records a low person IoU (0.024) in this training snapshot despite leading global overlap, which motivates joint reporting of summary and class-specific evidence. Validation loss is lowest for Dice-only training (0.82) but is not comparable across loss families.

![**Dice training prioritizes the person class more than CE-only here.** Bars: person IoU (dark green) and person pixel accuracy (olive). Purple line with markers: HD95 on the person mask (lower is better). Dice-only U-Net peaks person IoU and person accuracy; CE-only U-Net achieves the best U-Net HD95; DeepLab is strong on global metrics (Table 1) but not on these person-centric scores in this log.](./report_figures/fig02_person_metrics.png){width=95%}

Figure 3 isolates the person axis of Table 1.

**Table 2.** Paired differences (comparison minus baseline). HD95: lower $\Delta$ is better.

| Ablation | Baseline | Comparison | $\Delta$ mIoU | $\Delta$ person IoU | $\Delta$ HD95 (person) |
|:--|:--|:--|--:|--:|--:|
| Augmentation | unet_small_no_aug | unet_small_with_aug | -0.0022 | +0.0121 | -7.15 |
| Loss (CE vs Dice) | unet_small_ce_only | unet_small_dice_only | 0.0000 | +0.0783 | +1.47 |
| Model width | unet_small_with_aug | unet_wider_with_aug | +0.0006 | +0.0013 | +10.58 |
| Architecture | unet_small_with_aug | deeplabv3_resnet50 | **+0.0237** | -0.1111 | +9.17 |

Augmentation reduces mIoU slightly ($-0.0022$) while improving person IoU ($+0.0121$) and HD95 ($-7.15$). Replacing CE-only with Dice-only leaves printed mIoU unchanged but raises person IoU by $0.0783$ with a small HD95 penalty ($+1.47$). Width scaling is nearly neutral on mIoU and person IoU yet raises HD95 by $10.58$, consistent with the Methods note that **LR differs** between narrow and wide runs. The architecture row shows the largest mIoU gain ($+0.0237$) together with a person IoU drop ($-0.1111$) and higher HD95 ($+9.17$); interpret together with the **pretraining and LR** caveats above.

![**Ablation directions for global versus person overlap.** Horizontal bars: $\Delta$mIoU (blue) and $\Delta$person IoU (orange) for each paired study in Table 2; zero is vertical. Takeaway: augmentation and architecture rows move mIoU and person IoU in opposite directions; the loss swap moves person IoU strongly with near-zero $\Delta$mIoU at three decimals.](./report_figures/fig03_ablation_deltas.png){width=95%}

Figure 4 mirrors Table 2 for the two overlap summaries.

![**DeepLab activates many VOC classes, not only background.** Horizontal bar chart of per-class IoU for DeepLabV3-ResNet50, **excluding** background, sorted ascending. Non-zero bars indicate where the model assigns measurable mass beyond chance; read as evidence of **breadth** of multi-class predictions.](./report_figures/fig04_deeplab_per_class_iou.png){width=72%}

Figure 5 lists every non-background class with non-negligible IoU for DeepLab in our export.

![**U-Net columns are sparse; DeepLab is dense across classes.** Heatmap of per-class IoU (rows = all twenty-one VOC labels including background; columns = DeepLab, U-Net+augmentation, U-Net Dice-only). Brighter cells mean higher IoU. Takeaway: DeepLab shows energy across many semantic rows, whereas U-Net columns concentrate brightness near **person** and a few incidental classes, matching a background-heavy failure mode for rare categories.](./report_figures/fig05_perclass_heatmap_models.png){width=62%}

Figure 6 contrasts dense versus sparse per-class structure under identical color scaling.

Dense prediction **mosaics** (image, ground truth, prediction) and **best- versus worst-case** person examples ranked by per-image person IoU remain in the companion notebook; export them for the submission PDF if required as qualitative figures.

# Discussion

Pixel accuracy is dominated by correct background predictions; consequently, it can increase when the model becomes more confident on easy majority regions even as rare-class IoU stagnates. Mean IoU penalizes such imbalance because it averages overlap across categories. Figure 2 visualizes this decoupling directly; Table 1 supplies the underlying numbers. DeepLabV3-ResNet50 widens the effective receptive field and fuses multi-scale context prior to upsampling, which favors recovery of diverse object categories and aligns with the spread of non-zero per-class IoU in Figures 5 and 6. Under the same evaluation resolution, compact U-Nets more often concentrate evidence in background and a limited set of foreground modes, as shown by the sparse columns in Figure 6. **Cross-model rankings should be read together with the fairness paragraph in Methods:** the architecture row is informative but not a pure isolation of ``DeepLab versus U-Net'' because pretraining and learning-rate schedules differ.

Absolute mIoU values remain modest at $256 \times 256$ with the documented training budget. Downsampling removes fine structures that remain semantically salient at full scale, and class imbalance continues to bias gradients toward safe dominant-class predictions. Within this constrained regime, relative ordering between objectives and augmentations remains informative where shared factors dominate.

The Dice-trained U-Net improves person IoU relative to cross-entropy at essentially fixed mIoU, consistent with overlap-based losses emphasizing foreground agreement. Cross-entropy with class reweighting optimizes a per-pixel softmax and, in these logs, attains the strongest person HD95 among U-Nets, indicating complementary behavior between overlap-based and likelihood-based training signals. Hausdorff metrics are sensitive to thin structures; small boundary displacements can alter HD95 substantially.

Augmentation slightly reduces mIoU in the paired comparison while improving person-centric measures. Geometric perturbations reduce reliance on fixed poses and crops yet can increase effective difficulty for infrequent classes that already receive limited supervision. Given finite epochs and early stopping, mean summaries can move in opposite directions to class-specific indices without inconsistency.

Increasing channel width without matching learning-rate protocol leaves mIoU and person IoU nearly unchanged while degrading HD95, which is compatible with optimization difficulty or texture overfitting when capacity grows **under a lower initial LR** than the narrow baseline. Architectural substitution from the augmented small U-Net to DeepLabV3-ResNet50 produces the dominant mIoU improvement, supporting the primacy of inductive bias and receptive field **when also granting ImageNet initialization**; the concurrent drop in person IoU for DeepLab relative to the Dice-trained U-Net reflects competition for representational capacity across twenty foreground classes within the same nominal budget.

Nearest-neighbor mask resizing preserves label integrity but can alias thin objects at coarse resolution, contributing to missing rare classes in mIoU rather than indicating label corruption. ImageNet normalization stabilizes optimization for architectures designed around pretrained statistics.

This study does not claim state-of-the-art VOC performance; it documents controlled comparisons under a transparent registry with explicit limits. Optional SAM-based pipelines are not evaluated here as dense twenty-one-way semantic competitors. Natural extensions include matched-LR sweeps for DeepLab and U-Net, width-only sweeps at fixed LR, higher-resolution training, and longer budgets without early stopping.

# Conclusion

We presented a reproducible comparison of U-Net variants and DeepLabV3-ResNet50 on Pascal VOC 2007 with paired preprocessing, a **documented hyperparameter registry**, and an explicit discussion of **which comparisons are and are not fair**. DeepLabV3-ResNet50 achieved the strongest mIoU and mean Dice, with broad non-background activation in per-class IoU. Dice-only U-Net training yielded the strongest person IoU and person pixel accuracy, whereas augmentation improved person HD95 despite a small negative change in mIoU in the logged pair. Architecture and objective choices materially shaped outcomes; **width results must be interpreted alongside learning-rate differences.** Qualitative material in the companion notebook completes the empirical record.

```{=latex}
\newpage
```

# References

1. Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., Zisserman, A. The PASCAL Visual Object Classes Challenge 2007 (VOC2007) results.

2. Ronneberger, O., Fischer, P., Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015.

3. Chen, L.-C., Papandreou, G., Schroff, F., Adam, H. Rethinking Atrous Convolution for Semantic Image Segmentation. arXiv:1706.05587, 2017.

4. Kirillov, A., et al. Segment Anything. In: IEEE/CVF International Conference on Computer Vision (ICCV), 2023.
