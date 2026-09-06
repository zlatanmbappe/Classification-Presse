"""Pipeline principal : corpus, analyse et vérification."""

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from .db import get_conn
from .compare_models import build_corpus_comparison
from .visualizations import build_all_visualizations


def corpus_stats() -> dict:
    """Retourne les informations principales sur le corpus PostgreSQL."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*), COUNT(DISTINCT rubrique), COUNT(DISTINCT site) FROM articles"
            )
            articles, themes, sites = cur.fetchone()
            cur.execute(
                "SELECT rubrique, COUNT(*) FROM articles GROUP BY rubrique ORDER BY rubrique"
            )
            by_theme = [
                {"theme": theme, "count": count}
                for theme, count in cur.fetchall()
            ]

    return {
        "articles": articles,
        "themes": themes,
        "sites": sites,
        "by_theme": by_theme,
    }


def check_environment() -> dict:
    """Vérifie Python, les bibliothèques principales et PostgreSQL."""
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 ou supérieur est requis.")

    for module in ("pandas", "sklearn", "spacy", "gensim", "requests", "flask"):
        importlib.import_module(module)

    return {"python": sys.version.split()[0], **corpus_stats()}


def run_module(module: str, *arguments: str) -> None:
    """Lance un module Python du projet et arrête si une étape échoue."""
    command = [sys.executable, "-m", module, *arguments]
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError(f"L'étape {module} a échoué.")


def expected_distribution(config_path: str = "config/sources.yaml") -> dict[str, int]:
    """Retourne le nombre d'articles attendu pour chaque thème."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    expected: dict[str, int] = {}
    for source in config.get("sources", []):
        theme = source["theme"]
        expected[theme] = expected.get(theme, 0) + int(source.get("limit", 10))
    return expected


def collect_corpus() -> dict:
    """Recrée un corpus propre à partir de config/sources.yaml."""
    # Le bouton « Construire le corpus » repart toujours de zéro.
    # Cela évite d'accumuler des articles à chaque nouvelle collecte.
    reset_project()
    run_module("news_classifier.collect_sources")

    stats = corpus_stats()
    actual = {item["theme"]: item["count"] for item in stats["by_theme"]}
    expected = expected_distribution()

    if actual != expected:
        details = ", ".join(
            f"{theme}: {actual.get(theme, 0)}/{count}"
            for theme, count in expected.items()
        )
        raise RuntimeError(
            "Le corpus n'a pas pu atteindre la répartition prévue (" + details + ")."
        )

    return {
        "articles": stats["articles"],
        "themes": stats["themes"],
        "by_theme": stats["by_theme"],
    }


def run_analysis(outputs_dir: str | Path = "outputs") -> dict:
    """Exécute le prétraitement, les modèles et leur évaluation."""
    outputs = Path(outputs_dir)
    outputs.mkdir(exist_ok=True)

    stats = corpus_stats()
    if stats["articles"] < 4 or stats["themes"] < 2:
        raise RuntimeError("Construisez d'abord le corpus.")

    # Le nombre de clusters/topics est égal au nombre de thèmes du corpus.
    k = stats["themes"]

    run_module(
        "news_classifier.export_corpus",
        "--out", str(outputs / "corpus.csv"),
    )
    run_module(
        "news_classifier.preprocess",
        "--input", str(outputs / "corpus.csv"),
        "--outdir", str(outputs),
        "--min-df", "2",
        "--max-df", "0.90",
    )
    run_module(
        "news_classifier.cluster_kmeans",
        "--tfidf", str(outputs / "tfidf.joblib"),
        "--k", str(k),
        "--outdir", str(outputs),
    )
    run_module(
        "news_classifier.lda_topics",
        "--input", str(outputs / "corpus_pretraite.csv"),
        "--topics", str(k),
        "--no-below", "2",
        "--no-above", "0.90",
        "--outdir", str(outputs),
    )
    run_module(
        "news_classifier.evaluate",
        "--kmeans", str(outputs / "kmeans_articles.csv"),
        "--lda", str(outputs / "lda_articles.csv"),
        "--outdir", str(outputs),
    )
    run_module(
        "news_classifier.supervised_classifier",
        "--tfidf", str(outputs / "tfidf.joblib"),
        "--outdir", str(outputs),
    )

    # Comparaison des trois méthodes + visualisations utiles au rapport.
    build_corpus_comparison(outputs)
    build_all_visualizations(outputs)

    return {"articles": stats["articles"], "themes": stats["themes"], "k": k}


def reset_project() -> None:
    """Vide la base et les résultats générés."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE articles, experiments RESTART IDENTITY CASCADE")

    outputs = Path("outputs")
    if outputs.exists():
        shutil.rmtree(outputs)
    outputs.mkdir()


def main() -> None:
    """Petites commandes techniques utiles pour les vérifications."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage : python -m news_classifier.pipeline check|collect|analysis|reset")

    action = sys.argv[1]
    if action == "check":
        info = check_environment()
        print(f"Python {info['python']} | articles={info['articles']} | thèmes={info['themes']}")
    elif action == "collect":
        print(collect_corpus())
    elif action == "analysis":
        print(run_analysis())
    elif action == "reset":
        reset_project()
        print("Projet réinitialisé.")
    else:
        raise SystemExit("Commande inconnue.")


if __name__ == "__main__":
    main()
