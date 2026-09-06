"""Clustering K-means des articles représentés en TF-IDF."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans

from .compare_models import majority_theme_mapping


def top_terms(model, vectorizer, n=12):
    """Retourne les termes les plus représentatifs de chaque cluster."""
    terms = vectorizer.get_feature_names_out()
    rows = []
    for i, center in enumerate(model.cluster_centers_):
        order = center.argsort()[::-1][:n]
        rows.append({"cluster": i, "termes": ", ".join(terms[j] for j in order)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfidf", default="outputs/tfidf.joblib")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = joblib.load(args.tfidf)
    X = data["X"]
    vectorizer = data["vectorizer"]
    df = data["df"].copy()

    model = KMeans(n_clusters=args.k, random_state=42, n_init="auto")
    df["cluster_kmeans"] = model.fit_predict(X)

    df.to_csv(outdir / "kmeans_articles.csv", index=False, encoding="utf-8")
    top_terms(model, vectorizer).to_csv(
        outdir / "kmeans_top_terms.csv", index=False, encoding="utf-8"
    )
    majority_theme_mapping(df, "cluster_kmeans").to_csv(
        outdir / "kmeans_cluster_theme.csv", index=False, encoding="utf-8"
    )
    joblib.dump(model, outdir / "kmeans_model.joblib")
    print(f"Résultats K-means enregistrés avec K={args.k}")


if __name__ == "__main__":
    main()
