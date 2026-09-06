"""Classification comparative d'un nouvel article à partir de son URL."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.metrics.pairwise import cosine_similarity

from .article_extraction import extract_article
from .compare_models import save_url_comparison
from .preprocess import lemmatize_texts


def _load_mapping(path: Path, id_col: str) -> dict[int, str]:
    if not path.exists():
        raise RuntimeError("Lancez d'abord l'analyse pour créer l'association avec les thèmes.")
    df = pd.read_csv(path)
    return {int(row[id_col]): str(row["theme"]) for _, row in df.iterrows()}


def classify_article_url(url: str, top: int = 6, outputs_dir: str | Path = "outputs") -> dict:
    """Extrait un article et confronte K-means, LDA et la régression logistique."""
    outputs = Path(outputs_dir)
    tfidf_file = outputs / "tfidf.joblib"
    logistic_file = outputs / "supervised_model.joblib"
    kmeans_file = outputs / "kmeans_model.joblib"
    lda_file = outputs / "lda_model.gensim"
    lda_dictionary_file = outputs / "lda_dictionary.gensim"

    required = [tfidf_file, logistic_file, kmeans_file, lda_file, lda_dictionary_file]
    if any(not path.exists() for path in required):
        raise RuntimeError("Lancez d'abord l'analyse.")

    article = extract_article(url)
    text = lemmatize_texts([article.contenu_nettoye])[0]
    tokens = text.split()

    tfidf_data = joblib.load(tfidf_file)
    vectorizer = tfidf_data["vectorizer"]
    vector = vectorizer.transform([text])

    if vector.nnz == 0:
        raise RuntimeError("Aucun mot de l'article n'est reconnu par le modèle TF-IDF.")

    # 1) Régression logistique : probabilité par thème, avec gestion de l'incertitude.
    logistic_model = joblib.load(logistic_file)["model"]
    probabilities = logistic_model.predict_proba(vector)[0]
    order = probabilities.argsort()[::-1]
    best_index = int(order[0])
    second_index = int(order[1])
    best_score = float(probabilities[best_index])
    second_score = float(probabilities[second_index])
    margin = best_score - second_score
    uncertain = best_score < 0.35 or margin < 0.08
    predicted = "INCERTAIN" if uncertain else str(logistic_model.classes_[best_index])
    rankings = [
        {"theme": str(logistic_model.classes_[int(index)]), "score": float(probabilities[int(index)])}
        for index in order[:top]
    ]

    # 2) K-means : cluster le plus proche, puis association de ce cluster à son thème majoritaire.
    kmeans_model = joblib.load(kmeans_file)
    cluster = int(kmeans_model.predict(vector)[0])
    similarities = cosine_similarity(vector, kmeans_model.cluster_centers_)[0]
    kmeans_similarity = float(max(0.0, min(1.0, similarities[cluster])))
    kmeans_mapping = _load_mapping(outputs / "kmeans_cluster_theme.csv", "cluster_kmeans")
    kmeans_theme = kmeans_mapping.get(cluster, "INCONNU")
    kmeans_by_theme: dict[str, float] = {}
    for cluster_id, similarity in enumerate(similarities):
        theme = kmeans_mapping.get(int(cluster_id))
        if theme:
            score = float(max(0.0, min(1.0, similarity)))
            kmeans_by_theme[theme] = max(kmeans_by_theme.get(theme, 0.0), score)

    # 3) LDA : topic dominant, puis association de ce topic à son thème majoritaire.
    lda_model = LdaModel.load(str(lda_file))
    dictionary = Dictionary.load(str(lda_dictionary_file))
    bow = dictionary.doc2bow(tokens)
    if bow:
        lda_distribution = lda_model.get_document_topics(bow, minimum_probability=0)
        lda_topic, lda_probability = max(lda_distribution, key=lambda item: item[1])
        lda_topic = int(lda_topic)
        lda_probability = float(lda_probability)
    else:
        lda_topic = -1
        lda_probability = 0.0
    lda_mapping = _load_mapping(outputs / "lda_topic_theme.csv", "topic_lda")
    lda_theme = lda_mapping.get(lda_topic, "INCONNU")
    lda_by_theme: dict[str, float] = {}
    for topic_id, probability in (lda_distribution if bow else []):
        theme = lda_mapping.get(int(topic_id))
        if theme:
            lda_by_theme[theme] = lda_by_theme.get(theme, 0.0) + float(probability)

    logistic_by_theme = {str(logistic_model.classes_[i]): float(probabilities[i]) for i in range(len(probabilities))}
    all_themes = list(logistic_model.classes_)
    theme_scores = [
        {
            "theme": str(theme),
            "kmeans": float(kmeans_by_theme.get(str(theme), 0.0)),
            "lda": float(lda_by_theme.get(str(theme), 0.0)),
            "logistic": float(logistic_by_theme.get(str(theme), 0.0)),
        }
        for theme in all_themes
    ]

    names = vectorizer.get_feature_names_out()
    row = vector.getrow(0)
    positions = row.data.argsort()[::-1][:10]
    terms = [str(names[row.indices[position]]) for position in positions]

    result = {
        "url": article.url,
        "site": article.site,
        "title": article.titre,
        "word_count": len(article.contenu_nettoye.split()),
        "tfidf_terms": int(vector.nnz),
        "terms": terms,
        "theme_scores": theme_scores,
        "kmeans": {
            "theme": kmeans_theme,
            "cluster": cluster,
            "score": kmeans_similarity,
            "score_label": "Similarité au centre",
        },
        "lda": {
            "theme": lda_theme,
            "topic": lda_topic,
            "score": lda_probability,
            "score_label": "Poids du topic",
        },
        "logistic": {
            "predicted": predicted,
            "best_theme": str(logistic_model.classes_[best_index]),
            "best_score": best_score,
            "margin": margin,
            "uncertain": uncertain,
            "rankings": rankings,
        },
        # Compatibilité avec l'ancienne interface et les anciens usages.
        "predicted": predicted,
        "best_theme": str(logistic_model.classes_[best_index]),
        "best_score": best_score,
        "margin": margin,
        "uncertain": uncertain,
        "rankings": rankings,
    }

    save_url_comparison(result, outputs)
    return result
