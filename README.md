# Data Smell Detector

CLI that scans a dataset (CSV/Parquet) for common "data smells" and outputs an **HTML + JSON** report with actionable fixes.

## Smells Checked
- Missingness (overall + per-column)
- Duplicates
- Skew/Kurtosis (numeric)
- Outliers (IQR rule, >10% flagged)
- High-cardinality categoricals
- Constant columns
- (Optional) Target checks:
  - Class imbalance
  - Potential target leakage (high corr / 1–1 mappings)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m smell_detector.cli examples/sample.csv --out out/
# with target column
python -m smell_detector.cli examples/sample.csv --out out/ --target label
