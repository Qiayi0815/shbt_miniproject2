# Segmentation experiment summary

> Synced to the rendered outputs currently stored in `notebooks/mini_project_2_pascal_voc_segmentation.ipynb`.

## Key findings

- Best overall overlap: deeplabv3_resnet50 with mIoU=0.0655 and mean Dice=0.0978.
- Best pixel accuracy: unet_wider_with_aug at 0.7141.
- Best person-class IoU: unet_small_dice_only at 0.2098.
- Best person HD95 (lower is better): unet_small_ce_only at 88.38.
- Augmentation: ΔmIoU=-0.0022, Δhuman_iou=+0.0121, Δhd95_person=-7.15.
- Loss function (CE vs Dice): ΔmIoU=+0.0000, Δhuman_iou=+0.0783, Δhd95_person=+1.47.
- Model size (small vs wide): ΔmIoU=+0.0006, Δhuman_iou=+0.0013, Δhd95_person=+10.58.
- Architecture (U-Net vs DeepLab): ΔmIoU=+0.0237, Δhuman_iou=-0.1111, Δhd95_person=+9.17.

## Ranked summary (by mIoU)

| experiment | pixel_accuracy | mIoU | mean_dice | hd95_person | human_iou | human_accuracy | val_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deeplabv3_resnet50 | 0.6526 | 0.0655 | 0.0978 | 97.9131 | 0.0241 | 0.0272 | 2.5729 |
| unet_small_dice_only | 0.6711 | 0.0450 | 0.0587 | 89.8553 | 0.2098 | 0.4787 | 0.8211 |
| unet_small_ce_only | 0.7069 | 0.0450 | 0.0585 | 88.3839 | 0.1315 | 0.3231 | 2.5740 |
| unet_small_no_aug | 0.6560 | 0.0440 | 0.0591 | 95.8874 | 0.1232 | 0.3741 | 3.4112 |
| unet_wider_with_aug | 0.7141 | 0.0423 | 0.0538 | 99.3223 | 0.1366 | 0.2849 | 3.3952 |
| unet_small_with_aug | 0.7138 | 0.0418 | 0.0530 | 88.7402 | 0.1353 | 0.2696 | 3.4392 |

## Best-by-metric table

| metric | direction | best_experiment | value |
| --- | --- | --- | --- |
| mIoU | max | deeplabv3_resnet50 | 0.0655 |
| mean_dice | max | deeplabv3_resnet50 | 0.0978 |
| pixel_accuracy | max | unet_wider_with_aug | 0.7141 |
| human_iou | max | unet_small_dice_only | 0.2098 |
| human_accuracy | max | unet_small_dice_only | 0.4787 |
| hd95_person | min | unet_small_ce_only | 88.3839 |
| val_loss | min | unet_small_dice_only | 0.8211 |

## Person-class ranking

| experiment | human_iou | human_accuracy | hd95_person |
| --- | --- | --- | --- |
| unet_small_dice_only | 0.2098 | 0.4787 | 89.8553 |
| unet_wider_with_aug | 0.1366 | 0.2849 | 99.3223 |
| unet_small_with_aug | 0.1353 | 0.2696 | 88.7402 |
| unet_small_ce_only | 0.1315 | 0.3231 | 88.3839 |
| unet_small_no_aug | 0.1232 | 0.3741 | 95.8874 |
| deeplabv3_resnet50 | 0.0241 | 0.0272 | 97.9131 |

## Per-class IoU

| experiment | background | aeroplane | bicycle | bird | boat | bottle | bus | car | cat | chair | diningtable | dog | horse | motorbike | person | pottedplant | sheep | sofa | train | tvmonitor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unet_small_no_aug | 0.692 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.028 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.081 | 0.123 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| unet_small_with_aug | 0.726 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.016 | 0.135 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| unet_small_ce_only | 0.733 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.072 | 0.131 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| unet_small_dice_only | 0.704 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | 0.210 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| unet_wider_with_aug | 0.729 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 | 0.000 | 0.000 | 0.018 | 0.137 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| deeplabv3_resnet50 | 0.712 | 0.186 | 0.031 | 0.024 | 0.000 | 0.000 | 0.032 | 0.000 | 0.045 | 0.004 | 0.084 | 0.048 | 0.048 | 0.062 | 0.024 | 0.000 | 0.014 | 0.000 | 0.048 | 0.012 |

## Ablation deltas

| ablation | baseline | comparison | baseline_mIoU | comparison_mIoU | delta_mIoU | baseline_mean_dice | comparison_mean_dice | delta_mean_dice | baseline_pixel_accuracy | delta_pixel_accuracy | baseline_human_iou | comparison_human_iou | delta_human_iou | baseline_hd95_person | comparison_hd95_person | delta_hd95_person | baseline_val_loss | comparison_val_loss | delta_val_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Augmentation | unet_small_no_aug | unet_small_with_aug | 0.0440 | 0.0418 | -0.0022 | 0.0591 | 0.0530 | -0.0061 | 0.6560 | 0.0578 | 0.1232 | 0.1353 | 0.0121 | 95.8874 | 88.7402 | -7.1472 | 3.4112 | 3.4392 | 0.0279 |
| Loss function (CE vs Dice) | unet_small_ce_only | unet_small_dice_only | 0.0450 | 0.0450 | 0.0000 | 0.0585 | 0.0587 | 0.0002 | 0.7069 | -0.0358 | 0.1315 | 0.2098 | 0.0783 | 88.3839 | 89.8553 | 1.4714 | 2.5740 | 0.8211 | -1.7529 |
| Model size (small vs wide) | unet_small_with_aug | unet_wider_with_aug | 0.0418 | 0.0423 | 0.0006 | 0.0530 | 0.0538 | 0.0009 | 0.7138 | 0.0003 | 0.1353 | 0.1366 | 0.0013 | 88.7402 | 99.3223 | 10.5821 | 3.4392 | 3.3952 | -0.0439 |
| Architecture (U-Net vs DeepLab) | unet_small_with_aug | deeplabv3_resnet50 | 0.0418 | 0.0655 | 0.0237 | 0.0530 | 0.0978 | 0.0449 | 0.7138 | -0.0612 | 0.1353 | 0.0241 | -0.1111 | 88.7402 | 97.9131 | 9.1729 | 3.4392 | 2.5729 | -0.8662 |

## Notebook ablation interpretation

```text

=== Augmentation ===
Baseline   : unet_small_no_aug
Comparison : unet_small_with_aug
- mIoU delta        : -0.0022
- mean Dice delta   : -0.0061
- pixel accuracy Δ  : +0.0578
- person IoU delta  : +0.0121
- HD95 delta        : -7.1472 (lower is better)
- val loss delta    : +0.0279 (lower is better)

=== Loss function (CE vs Dice) ===
Baseline   : unet_small_ce_only
Comparison : unet_small_dice_only
- mIoU delta        : +0.0000
- mean Dice delta   : +0.0002
- pixel accuracy Δ  : -0.0358
- person IoU delta  : +0.0783
- HD95 delta        : +1.4714 (lower is better)
- val loss delta    : -1.7529 (lower is better)

=== Model size (small vs wide) ===
Baseline   : unet_small_with_aug
Comparison : unet_wider_with_aug
- mIoU delta        : +0.0006
- mean Dice delta   : +0.0009
- pixel accuracy Δ  : +0.0003
- person IoU delta  : +0.0013
- HD95 delta        : +10.5821 (lower is better)
- val loss delta    : -0.0439 (lower is better)

=== Architecture (U-Net vs DeepLab) ===
Baseline   : unet_small_with_aug
Comparison : deeplabv3_resnet50
- mIoU delta        : +0.0237
- mean Dice delta   : +0.0449
- pixel accuracy Δ  : -0.0612
- person IoU delta  : -0.1111
- HD95 delta        : +9.1729 (lower is better)
- val loss delta    : -0.8662 (lower is better)
```

