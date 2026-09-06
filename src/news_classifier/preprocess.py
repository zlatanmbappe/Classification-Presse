"""Prétraitement du français et création de la représentation TF-IDF."""

import argparse
from pathlib import Path

import joblib
import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

WEB_WORDS = {
    "cookie", "cookies", "publicité", "newsletter", "abonnement", "contenu",
    "site", "page", "article", "lire", "partager", "menu", "accueil",
}


def load_stopwords() -> set[str]:
    """Charge les mots vides français de NLTK."""
    try:
        words = stopwords.words("french")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("french")
    return set(words) | WEB_WORDS


def load_spacy():
    """Charge le petit modèle français de spaCy."""
    try:
        return spacy.load("fr_core_news_sm", disable=["parser", "ner"])
    except OSError as exc:
        raise RuntimeError("Le modèle spaCy fr_core_news_sm n'est pas installé.") from exc


def lemmatize_texts(texts: list[str]) -> list[str]:
    """Transforme chaque texte en une suite de lemmes utiles."""
    nlp = load_spacy()
    stops = load_stopwords()
    result = []

    for text in texts:
        doc = nlp(text)
        words = []
        for token in doc:
            lemma = token.lemma_.lower().strip()
            if token.pos_ not in {"NOUN", "PROPN", "ADJ", "VERB"}:
                continue
            if len(lemma) < 3 or not lemma.replace("-", "").isalpha():
                continue
            if lemma in stops or token.is_stop:
                continue
            words.append(lemma)
        result.append(" ".join(words))

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/corpus.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.90)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    texts = df["contenu_nettoye"].fillna("").tolist()
    df["texte_pretraite"] = lemmatize_texts(texts)
    df = df[df["texte_pretraite"].str.split().str.len() >= 20].copy()

    if df.empty:
        raise SystemExit("Aucun texte exploitable après prétraitement.")

    df.to_csv(outdir / "corpus_pretraite.csv", index=False, encoding="utf-8")

    vectorizer = TfidfVectorizer(
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=8000,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(df["texte_pretraite"])

    joblib.dump(
        {"X": X, "vectorizer": vectorizer, "df": df},
        outdir / "tfidf.joblib",
    )
    print(f"TF-IDF : {X.shape[0]} documents x {X.shape[1]} termes")


if __name__ == "__main__":
    main()
