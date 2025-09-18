# Project Folder Guide

## boxai/
Python package with reusable code.
- `__init__.py`: exposes `FinalTotalPredictor` (and version).
- `models/`: model classes (currently `final_total_predictor.py`).

## notebooks/
- `XGB_Model.ipynb`: training + artifact saving.
- `FinalTotal_Inference.ipynb`: minimal inference usage.

## Data Wrangling.ipynb
Cleaning work 

## requirements.txt
Pinned runtime dependencies for training & inference.

## GUIDE.md (this file)
Explains structure for quick onboarding.

---
### Quick Start
1. Train: open `notebooks/XGB_Model.ipynb` and run all cells (creates/overwrites `artifacts/final_total`).
2. Infer: open `notebooks/FinalTotal_Inference.ipynb`, edit `wk1_value` or list and run

