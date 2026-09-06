"""Quantitative evaluation: purity, ARI, and comparison table."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, confusion_matrix


def purity_score(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)


def eval_file(path: str, pred_col: str, label_col: str = "rubrique") -> dict:
    df = pd.read_csv(path).dropna(subset=[label_col, pred_col])
    return {
        "fichier": path,
        "algorithme": pred_col,
        "nb_articles": len(df),
        "nb_gold_labels": df[label_col].nunique(),
        "nb_clusters_topics": df[pred_col].nunique(),
        "purete": purity_score(df[label_col].astype(str), df[pred_col].astype(str)),
        "ari": adjusted_rand_score(df[label_col].astype(str), df[pred_col].astype(str)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kmeans", default="outputs/kmeans_articles.csv")
    parser.add_argument("--lda", default="outputs/lda_articles.csv")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    rows = []
    if Path(args.kmeans).exists(): rows.append(eval_file(args.kmeans, "cluster_kmeans"))
    if Path(args.lda).exists(): rows.append(eval_file(args.lda, "topic_lda"))
    table = pd.DataFrame(rows)
    if Path(outdir / "lda_coherence.csv").exists():
        coh = pd.read_csv(outdir / "lda_coherence.csv").iloc[0]["coherence_cv"]
        table.loc[table["algorithme"] == "topic_lda", "coherence_cv"] = coh
    table.to_csv(outdir / "evaluation.csv", index=False)
    print(table.to_string(index=False))

if __name__ == "__main__":
    main()
