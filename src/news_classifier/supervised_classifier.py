"""Classifieur supervisé pour prédire le thème d'un nouvel article."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfidf", default="outputs/tfidf.joblib")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = joblib.load(args.tfidf)
    X = data["X"]
    df = data["df"].copy()

    valid = df["rubrique"].notna() & df["rubrique"].astype(str).str.strip().ne("")
    X = X[valid.to_numpy()]
    y = df.loc[valid, "rubrique"].astype(str).str.strip().to_numpy()
    if len(set(y)) < 2:
        raise SystemExit("Il faut au moins 2 thèmes différents pour entraîner le classifieur supervisé.")

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
    )

    counts = pd.Series(y).value_counts()
    min_class = int(counts.min())
    accuracy_mean = np.nan
    accuracy_std = np.nan
    f1_macro_mean = np.nan
    f1_macro_std = np.nan
    n_splits = 0

    if min_class >= 2:
        n_splits = min(5, min_class)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        f1 = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
        predictions_cv = cross_val_predict(model, X, y, cv=cv, method="predict")
        pd.DataFrame({"rubrique": y, "prediction_cv": predictions_cv}).to_csv(
            outdir / "supervised_predictions_cv.csv", index=False, encoding="utf-8"
        )
        accuracy_mean, accuracy_std = float(acc.mean()), float(acc.std())
        f1_macro_mean, f1_macro_std = float(f1.mean()), float(f1.std())

    model.fit(X, y)
    joblib.dump(
        {
            "model": model,
            "classes": list(model.classes_),
            "nb_documents": int(len(y)),
            "class_counts": counts.to_dict(),
        },
        outdir / "supervised_model.joblib",
    )

    pd.DataFrame([
        {
            "algorithme": "logistic_regression",
            "nb_articles": len(y),
            "nb_themes": len(set(y)),
            "cv_folds": n_splits,
            "accuracy_cv": accuracy_mean,
            "accuracy_cv_std": accuracy_std,
            "f1_macro_cv": f1_macro_mean,
            "f1_macro_cv_std": f1_macro_std,
        }
    ]).to_csv(outdir / "supervised_evaluation.csv", index=False)

    print(f"Classifieur supervisé entraîné sur {len(y)} articles et {len(set(y))} thèmes.")
    if n_splits:
        print(f"Validation croisée ({n_splits} folds) : accuracy={accuracy_mean:.3f}, F1-macro={f1_macro_mean:.3f}")
    else:
        print("Validation croisée non calculée : au moins une classe contient moins de 2 articles.")


if __name__ == "__main__":
    main()
