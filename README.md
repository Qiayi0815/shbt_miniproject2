# SHBT Mini-Project 2: Pascal VOC 2007 segmentation

Code and report for semantic segmentation experiments (U-Net, DeepLabV3-ResNet50, ablations).

## Contents

- `notebooks/` -- main Jupyter notebook and `train.py` for scripted training
- `artifacts/report_exports/` -- CSV summaries and `make_report_figures.py`
- `report_figures/` -- PNG plots used in the PDF report
- `Mini-Project-2-Report.md` / `Mini-Project-2-Report.pdf` -- write-up

## Data

Download Pascal VOC 2007 (segmentation) and point `SegConfig.dataset_root` / notebook paths at your local `VOCtrainval_06-Nov-2007` tree. Archive files are gitignored by default.

## Build report PDF

```bash
pandoc Mini-Project-2-Report.md -o Mini-Project-2-Report.pdf --pdf-engine=pdflatex --from=markdown+raw_tex
```

Regenerate figures (requires matplotlib, pandas):

```bash
export MPLCONFIGDIR="$PWD/notebooks/.mplconfig"
python artifacts/report_exports/make_report_figures.py
```
