"""Interface web locale du logiciel."""

import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for

from .classify_url import classify_article_url
from .pipeline import collect_corpus, corpus_stats, run_analysis


app = Flask(__name__)
app.secret_key = "classification-presse"
OUTPUTS = Path("outputs")


def read_metrics() -> dict:
    """Lit les résultats CSV produits par l'analyse."""
    metrics = {"unsupervised": [], "supervised": None, "comparison": []}

    evaluation = OUTPUTS / "evaluation.csv"
    if evaluation.exists():
        df = pd.read_csv(evaluation)
        for _, row in df.iterrows():
            coherence = row.get("coherence_cv")
            if pd.isna(coherence):
                coherence = None
            metrics["unsupervised"].append({
                "algorithm": row["algorithme"],
                "purity": row.get("purete"),
                "ari": row.get("ari"),
                "coherence": coherence,
            })

    supervised = OUTPUTS / "supervised_evaluation.csv"
    if supervised.exists():
        df = pd.read_csv(supervised)
        if not df.empty:
            row = df.iloc[0]
            metrics["supervised"] = {
                "accuracy": row["accuracy_cv"],
                "f1": row["f1_macro_cv"],
                "folds": int(row["cv_folds"]),
            }

    comparison = OUTPUTS / "comparison_reference_scores.csv"
    if comparison.exists():
        df = pd.read_csv(comparison)
        metrics["comparison"] = df.to_dict(orient="records")

    return metrics


def show_page(result=None, error=None, article_url=""):
    """Affiche la page principale avec les données actuelles."""
    try:
        stats = corpus_stats()
    except Exception as exc:
        stats = {"articles": 0, "themes": 0, "sites": 0, "by_theme": [], "error": str(exc)}

    visualizations = [
        {"file": "figures/kmeans_pca.svg", "title": "K-means — projection PCA", "note": "Chaque point représente un article ; la proximité montre les ressemblances dans l'espace TF-IDF."},
        {"file": "figures/kmeans_mots.svg", "title": "Mots représentatifs K-means", "note": "Les termes les plus pondérés aident à interpréter chaque cluster."},
        {"file": "figures/lda_mots.svg", "title": "Mots représentatifs LDA", "note": "Les termes les plus importants permettent d'interpréter les topics latents."},
        {"file": "figures/logistic_confusion_matrix.svg", "title": "Matrice de confusion — régression logistique", "note": "Elle montre les thèmes correctement classés et les confusions en validation croisée."},
        {"file": "figures/kmeans_elbow.svg", "title": "Courbe du coude K-means", "note": "Elle montre comment l'inertie évolue quand le nombre de clusters change."},
        {"file": "figures/lda_coherence_curve.svg", "title": "Courbe de cohérence LDA", "note": "Elle compare la cohérence obtenue pour plusieurs nombres de topics."},
    ]
    visualizations = [item for item in visualizations if (OUTPUTS / item["file"]).exists()]

    return render_template(
        "index.html",
        stats=stats,
        metrics=read_metrics(),
        result=result,
        error=error,
        url=article_url,
        visualizations=visualizations,
        corpus_chart=(OUTPUTS / "figures" / "comparaison_corpus.svg").exists(),
        corpus_reference_chart=(OUTPUTS / "figures" / "comparaison_corpus_reference.svg").exists(),
        url_chart=(OUTPUTS / "figures" / "comparaison_url.svg").exists(),
        url_theme_chart=(OUTPUTS / "figures" / "comparaison_url_par_theme.svg").exists(),
    )


@app.get("/")
def index():
    return show_page()


@app.get("/outputs/<path:filename>")
def output_file(filename):
    """Expose uniquement les résultats générés localement pour les graphiques."""
    response = send_from_directory(OUTPUTS.resolve(), filename)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/collect")
def collect():
    try:
        result = collect_corpus()
        flash(f"Corpus reconstruit : {result['articles']} articles, 10 par thème.", "success")
    except Exception as exc:
        flash(f"Erreur de collecte : {exc}", "error")
    return redirect(url_for("index"))


@app.post("/analyze")
def analyze():
    try:
        result = run_analysis()
        flash(f"Analyse terminée sur {result['articles']} articles. Comparaison des trois méthodes générée.", "success")
    except Exception as exc:
        flash(f"Erreur d'analyse : {exc}", "error")
    return redirect(url_for("index"))


@app.post("/classify")
def classify():
    article_url = request.form.get("url", "").strip()
    parsed = urlparse(article_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return show_page(error="URL invalide.", article_url=article_url)

    try:
        result = classify_article_url(article_url, top=6)
        return show_page(result=result, article_url=article_url)
    except Exception as exc:
        return show_page(error=str(exc), article_url=article_url)


def main() -> None:
    """Lance l'interface sur le port local 5000."""
    address = "http://127.0.0.1:5000"
    print(f"Interface : {address}")
    print("Arrêt : Ctrl+C")

    threading.Timer(1, lambda: webbrowser.open(address)).start()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
