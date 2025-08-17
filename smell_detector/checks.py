from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder

def _num_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def _cat_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

def _class_imbalance(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    vc = df[target].value_counts(dropna=False)
    total = int(vc.sum())
    ratios = {str(k): float(v)/total for k, v in vc.items()}
    worst = float(vc.max()) / max(1, total)
    return {"counts": vc.to_dict(), "ratios": ratios, "max_ratio": worst}

def _target_leakage(df: pd.DataFrame, target: str) -> List[Dict[str, Any]]:
    # Heuristic: high correlation (numeric) or too-perfect label encoding (categorical)
    leaks = []
    y = df[target]
    numeric = _num_cols(df.drop(columns=[target], errors="ignore"))
    # numeric corr
    for c in numeric:
        try:
            corr = abs(df[c].corr(y))
            if pd.notna(corr) and corr >= 0.95:
                leaks.append({"column": c, "type": "numeric_corr", "score": float(corr)})
        except Exception:
            pass
    # categorical near-unique mapping
    for c in _cat_cols(df.drop(columns=[target], errors="ignore")):
        try:
            # If cardinality equals target cardinality and rows map 1-1, flag
            pairs = df[[c, target]].dropna().drop_duplicates()
            if pairs[c].nunique() == pairs[target].nunique() and pairs.shape[0] <= df.shape[0]*0.6:
                leaks.append({"column": c, "type": "1-1_mapping", "score": 1.0})
        except Exception:
            pass
    return leaks

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
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        frac = float(((s < lo) | (s > hi)).mean())
        if frac > 0.1:
            out.append({"column": c, "outlier_frac": round(frac, 3)})
    return out

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
    return {"overall_frac": float(df.isna().mean().mean()), "by_col": top.to_dict()}

def _dupes(df: pd.DataFrame) -> Dict[str, Any]:
    dup = df.duplicated().sum()
    return {"duplicates": int(dup), "frac": float(dup) / max(1, len(df))}

def _dtype_summary(df: pd.DataFrame) -> Dict[str, Any]:
    return {c: str(df[c].dtype) for c in df.columns}

def _basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    nums = df[_num_cols(df)]
    desc = nums.describe().to_dict() if not nums.empty else {}
    return {"shape": list(df.shape), "describe": desc}

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
        sug.append("Significant outliers: consider robust scaling, winsorization, or model choices tolerant to outliers.")
    if report["skew_kurt"]:
        sug.append("Skewed distributions: consider log/Box-Cox transforms for affected features.")
    if report["target"] and report["target"].get("imbalance", {}).get("max_ratio", 0) > 0.9:
        sug.append("Severe class imbalance: consider stratified sampling, class weights, or resampling (SMOTE).")
    if report["target"] and report["target"].get("leakage"):
        sug.append("Potential target leakage: remove flagged columns or perform leakage-safe CV pipeline.")
    return sug

def run_checks(df: pd.DataFrame, target: str | None = None) -> Dict[str, Any]:
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
    }

    if target and target in df.columns:
        tgt = {"name": target, "imbalance": _class_imbalance(df, target)}
        try:
            leaks = _target_leakage(df.select_dtypes(include=[np.number]).join(df[target], how="inner"), target)
        except Exception:
            leaks = _target_leakage(df, target)
        tgt["leakage"] = leaks
        report["target"] = tgt

    report["suggestions"] = _suggestions(report)
    return report
