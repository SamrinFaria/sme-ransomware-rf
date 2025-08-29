import io
import os
import argparse
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RAW_BASE = "https://raw.githubusercontent.com/subiksha03/dataset/main"
OUT_CSV = "dataset.csv"
METRICS_CSV = "metrics.csv"

def _fetch(path: str) -> bytes:
    url = f"{RAW_BASE}/{path}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def fetch_to_csv() -> pd.DataFrame:
    # 1) Try Excel
    try:
        excel_bytes = _fetch("final%20ds.xlsx")
        xls = pd.ExcelFile(io.BytesIO(excel_bytes))
        sheet_name = None
        for s in xls.sheet_names:
            if str(s).strip().lower() == "dataset":
                sheet_name = s
                break
        if sheet_name is None:
            sheet_name = xls.sheet_names[0]
        df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=sheet_name)
    except Exception as e:
        print(f"[INFO] Excel path failed ({e}). Trying CSV candidates...")
        df = None
        csv_candidates = ["api.csv", "ent.csv", "exe.csv", "strings2.csv"]
        dfs = []
        for name in csv_candidates:
            try:
                b = _fetch(name)
                part = pd.read_csv(io.BytesIO(b))
                cols_l = [str(c).strip().lower() for c in part.columns]
                if "label" in cols_l:
                    dfs.append(part)
                    print(f"[INFO] loaded {name} with shape {part.shape}")
                else:
                    print(f"[INFO] {name} has no 'label' column; skipping")
            except Exception as ee:
                print(f"[INFO] skip {name}: {ee}")
        if not dfs:
            raise RuntimeError("Could not load any CSVs from the dataset repo.")
        # Align columns & concatenate
        all_cols = sorted(set().union(*[set(d.columns) for d in dfs]))
        dfs = [d.reindex(columns=all_cols, fill_value=0) for d in dfs]
        df = pd.concat(dfs, axis=0, ignore_index=True).drop_duplicates()

    # 2) Normalize label column name safely
    label_col = None
    lower_map = {c: str(c).strip().lower() for c in df.columns}
    for c, lc in lower_map.items():
        if lc in ("label", "class", "target", "y"):
            label_col = c
            break
    if label_col is None:
        raise RuntimeError("No label/target column found.")

    df = df.rename(columns={label_col: "label"})

    # 3) Keep only numeric + label (label can be non-numeric initially)
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if "label" not in num_cols:
        num_cols.append("label")
    df = df[num_cols].copy()

    # 4) Clean & coerce label to binary {0,1}
    df = df.dropna(subset=["label"])
    feat_cols = [c for c in df.columns if c != "label"]
    df = df.dropna(subset=feat_cols, how="all").fillna(0)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["label"] = df["label"].clip(0, 1)

    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] wrote {OUT_CSV} with shape {df.shape}")
    return df

def plot_cm(cm: np.ndarray, classes, title: str, outpath: str):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def run_rf(df: pd.DataFrame, runs: int = 5, test_size: float = 0.25):
    X = df.drop(columns=["label"])
    y = df["label"]
    rows = []
    for run in range(1, runs + 1):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=run, stratify=y
        )
        rf = RandomForestClassifier(
            n_estimators=100 + 20 * run,
            max_depth=None if run % 2 else 12,
            min_samples_leaf=1 if run < 3 else 2,
            random_state=run, n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        y_pr = rf.predict(X_te)
        acc = accuracy_score(y_te, y_pr)
        pre = precision_score(y_te, y_pr, zero_division=0)
        rec = recall_score(y_te, y_pr, zero_division=0)
        f1  = f1_score(y_te, y_pr, zero_division=0)
        print(f"\n=== Run {run} ===")
        print(classification_report(y_te, y_pr, digits=3))
        cm = confusion_matrix(y_te, y_pr)
        cm_file = f"confusion_matrix_run{run}.png"
        plot_cm(cm, ["benign(0)", "ransom(1)"], f"RF Confusion Matrix (run {run})", cm_file)
        print(f"[OK] saved {cm_file}")
        rows.append({
            "run": run, "n_estimators": rf.n_estimators,
            "max_depth": rf.max_depth if rf.max_depth is not None else "None",
            "min_samples_leaf": rf.min_samples_leaf,
            "accuracy": round(acc, 4),
            "precision": round(pre, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        })
    res = pd.DataFrame(rows)
    res.to_csv(METRICS_CSV, index=False)
    print("\nSummary:")
    print(res)
    print(f"\n[OK] wrote {METRICS_CSV}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    if args.skip-download and os.path.exists(OUT_CSV):
        df = pd.read_csv(OUT_CSV)
        print(f"[INFO] loaded existing {OUT_CSV} with shape {df.shape}")
    else:
        df = fetch_to_csv()
    run_rf(df, runs=args.runs, test_size=args.test_size)

if __name__ == "__main__":
    main()