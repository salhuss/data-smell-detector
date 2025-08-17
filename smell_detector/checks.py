from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
import os

# ---------- helpers: column types ----------
def _num_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def _cat_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

# ---------- target-aware checks ----------
def _class_imbalance(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    vc = df[target].value_counts(dropna=False)
    total = int(vc.sum())
    ratios = {str(k): float(v) / total for k, v in vc.items()}
    worst = float(vc.max()) / max(1, total)
    return {"counts": {str(k): int(v) for k, v in vc.items()}, "ratios": ratios, "max_ratio": worst}

def _target_leakage(df: pd.DataFrame, target: str) -> List[Dict[str, Any]]:
    leaks = []
    y = df[target]
    # numeric correlation ~1
    for c in _num_cols(df.drop(columns=[target], errors="ignore")):
        try:
            corr = abs(df[c].corr(y))
            if pd.notna(corr) and corr >= 0.95:
                leaks.append({"column": c, "type": "numeric_corr", "score": float(corr)})
        except Exception:
            pass
    # categorical one-to-one mapping
    for c in _cat_cols(df.drop(columns=[target], errors="ignore")):
        try:
            pairs = df[[c, target]].dropna().drop_duplicates()
            if pairs[c].nunique() == pairs[target].nunique() and pairs.shape[0] <= df.shape[0] * 0.6:
                leaks.append({"column": c, "type": "1-1_mapping", "score": 1.0})
        except Exception:
            pass
    return leaks

# ---------- distribution checks ----------
def _skew_kurtosis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    for c in _num_cols(df):
        s = df[c].dropna()
        if len(s) < 20:
            continue
        try:
            sk = stats.skew(s)
            ku = stats.kurtosis(s, fisher=True)
            if abs(sk) > 1 or abs(ku) > 3:
                out.append({"column": c, "skew": float(sk), "kurtosis": float(ku)})
        except Exception:
            pass
    return out

def _outlier_cols(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    for c in _num_cols(df):
        s = df[c].dropna()
        if len(s) < 20:
            continue
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        frac = float(((s < lo) | (s > hi)).mean())
        if frac > 0.1:
            out.append({"column": c, "outlier_frac": round(frac, 3)})
    return out

# ---------- schema & quality ----------
def _high_cardinality(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    n = len(df)
    for c in _cat_cols(df):
        uniq = df[c].nunique(dropna=True)
        if uniq > 0.5 * n or uniq > 5000:
            out.append({"column": c, "unique": int(uniq), "rows": int(n)})
    return out

def _constant_cols(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [{"column": c} for c in df.columns if df[c].nunique(dropna=False) <= 1]

def _missingness(df: pd.DataFrame) -> Dict[str, Any]:
    miss = df.isna().mean().sort_values(ascending=False)
    top = miss[miss > 0].head(50)
    return {"overall_frac": float(df.isna().mean().mean()), "by_col": {str(k): float(v) for k, v in top.items()}}

def _dupes(df: pd.DataFrame) -> Dict[str, Any]:
    dup = df.duplicated().sum()
    return {"duplicates": int(dup), "frac": float(dup) / max(1, len(df))}

def _dtype_summary(df: pd.DataFrame) -> Dict[str, Any]:
    return {str(c): str(df[c].dtype) for c in df.columns}

def _basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    nums = df[_num_cols(df)]
    desc = nums.describe().to_dict() if not nums.empty else {}
    return {"shape": [int(df.shape[0]), int(df.shape[1])], "describe": desc}

# ---------- correlation heatmap ----------
def _corr_heatmap(df: pd.DataFrame, out_dir: Optional[str]) -> Optional[str]:
    nums = df[_num_cols(df)]
    if nums.empty:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        corr = nums.corr()
        fig = plt.figure(figsize=(6, 5))
        im = plt.imshow(corr, aspect="auto", interpolation="nearest")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
        plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
        plt.tight_layout()
        if out_dir:
            path = os.path.join(out_dir, "corr_heatmap.png")
            plt.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            return path
        plt.close(fig)
    except Exception:
        return None
    return None

# ---------- reference comparison ----------
def _schema_diff(curr: pd.DataFrame, ref: pd.DataFrame) -> Dict[str, Any]:
    curr_cols = set(curr.columns)
    ref_cols = set(ref.columns)
    added = sorted(list(curr_cols - ref_cols))
    removed = sorted(list(ref_cols - curr_cols))
    dtype_changes = []
    for c in sorted(curr_cols & ref_cols):
        dc = str(curr[c].dtype)
        dr = str(ref[c].dtype)
        if dc != dr:
            dtype_changes.append({"column": c, "current": dc, "reference": dr})
    return {"added": added, "removed": removed, "dtype_changes": dtype_changes}

def _drift_numeric(curr: pd.Series, ref: pd.Series) -> Optional[Dict[str, Any]]:
    curr = curr.dropna()
    ref = ref.dropna()
    if len(curr) < 20 or len(ref) < 20:
        return None
    try:
        ks = stats.ks_2samp(curr, ref)
        return {"p_value": float(ks.pvalue)}
    except Exception:
        return None

def _drift_categorical(curr: pd.Series, ref: pd.Series) -> Optional[Dict[str, Any]]:
    try:
        curr_counts = curr.fillna("__NA__").value_counts()
        ref_counts = ref.fillna("__NA__").value_counts()
        cats = sorted(set(curr_counts.index) | set(ref_counts.index))
        curr_vec = np.array([curr_counts.get(c, 0) for c in cats])
        ref_vec = np.array([ref_counts.get(c, 0) for c in cats])
        if curr_vec.sum() == 0 or ref_vec.sum() == 0:
            return None
        # normalize
        curr_vec = curr_vec / curr_vec.sum()
        ref_vec = ref_vec / ref_vec.sum()
        from scipy.stats import chisquare
        eps = 1e-9
        ref_vec = np.clip(ref_vec, eps, None)
        cs = chisquare(curr_vec, ref_vec)
        return {"p_value": float(cs.pvalue)}
    except Exception:
        return None

def _drift_report(curr: pd.DataFrame, ref: pd.DataFrame) -> Dict[str, Any]:
    report = {"numeric": [], "categorical": []}
    shared = [c for c in curr.columns if c in ref.columns]
    for c in shared:
        if pd.api.types.is_numeric_dtype(curr[c]) and pd.api.types.is_numeric_dtype(ref[c]):
            r = _drift_numeric(curr[c], ref[c])
            if r and r["p_value"] < 0.01:
                report["numeric"].append({"column": c, "p_value": r["p_value"]})
        else:
            r = _drift_categorical(curr[c], ref[c])
            if r and r["p_value"] < 0.01:
                report["categorical"].append({"column": c, "p_value": r["p_value"]})
    return report

# ---------- suggestions ----------
def _suggestions(report: Dict[str, Any]) -> List[str]:
    sug = []
    if report["missingness"]["overall_frac"] > 0.05:
        sug.append("High missingness: consider imputation (median/most-frequent) or dropping problematic columns.")
    if report["dupes"]["frac"] > 0.02:
        sug.append("Many duplicate rows: review deduping keys or upstream data collection.")
    if report["constant_cols"]:
        sug.append("Constant columns detected: drop them to reduce noise.")
    if report["high_cardinality"]:
        sug.append("High-cardinality categoricals: consider hashing trick or target encoding carefully (avoid leakage).")
    if report["outliers"]:
        sug.append("Significant outliers: consider robust scaling or winsorization.")
    if report["skew_kurt"]:
        sug.append("Skewed distributions: consider log/Box-Cox transforms for affected features.")
    if report.get("target") and report["target"].get("imbalance", {}).get("max_ratio", 0) > 0.9:
        sug.append("Severe class imbalance: consider stratified sampling, class weights, or resampling (SMOTE).")
    if report.get("target") and report["target"].get("leakage"):
        sug.append("Potential target leakage: remove flagged columns or perform leakage-safe CV pipeline.")
    if report.get("schema_diff"):
        if report["schema_diff"]["added"] or report["schema_diff"]["removed"] or report["schema_diff"]["dtype_changes"]:
            sug.append("Schema drift detected: review added/removed columns and dtype changes.")
    if report.get("drift"):
        if report["drift"]["numeric"] or report["drift"]["categorical"]:
            sug.append("Statistical drift detected: consider recalibration or model retraining.")
    return sug

# ---------- entrypoint ----------
def run_checks(
    df: pd.DataFrame,
    target: str | None = None,
    ref_df: Optional[pd.DataFrame] = None,
    out_dir: Optional[str] = None,
    save_plots: bool = True,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "basic": _basic_stats(df),
        "dtypes": _dtype_summary(df),
        "missingness": _missingness(df),
        "dupes": _dupes(df),
        "constant_cols": _constant_cols(df),
        "high_cardinality": _high_cardinality(df),
        "skew_kurt": _skew_kurtosis(df),
        "outliers": _outlier_cols(df),
        "target": None,
        "schema_diff": None,
        "drift": None,
        "plots": {},
    }

    if target and target in df.columns:
        tgt = {"name": target, "imbalance": _class_imbalance(df, target)}
        try:
            leaks = _target_leakage(
                df.select_dtypes(include=[np.number]).join(df[target], how="inner"), target
            )
        except Exception:
            leaks = _target_leakage(df, target)
        tgt["leakage"] = leaks
        report["target"] = tgt

    if ref_df is not None:
        report["schema_diff"] = _schema_diff(df, ref_df)
        report["drift"] = _drift_report(df, ref_df)

    if save_plots:
        path = _corr_heatmap(df, out_dir)
        if path:
            report["plots"]["corr_heatmap_png"] = path

    report["preview"] = df.head(10).to_dict(orient="records")
    report["suggestions"] = _suggestions(report)
    return report
