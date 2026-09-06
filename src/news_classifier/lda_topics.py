"""LDA topic modeling with Gensim and c_v coherence."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

from .compare_models import majority_theme_mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/corpus_pretraite.csv")
    parser.add_argument("--topics", type=int, default=6)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--no-below", type=int, default=3)
    parser.add_argument("--no-above", type=float, default=0.8)
    args = parser.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    texts = [str(t).split() for t in df["texte_pretraite"].fillna("")]
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=args.no_below, no_above=args.no_above, keep_n=8000)
    if len(dictionary) == 0:
        raise SystemExit("Vocabulaire LDA vide : augmentez le corpus ou réduisez --no-below.")
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=args.topics, random_state=42, passes=10, alpha="auto", eta="auto")
    coherence = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence="c_v").get_coherence()

    topics = []
    for topic_id in range(args.topics):
        topics.append({"topic": topic_id, "termes": ", ".join(w for w, _ in lda.show_topic(topic_id, topn=12))})
    pd.DataFrame(topics).to_csv(outdir / "lda_topics.csv", index=False, encoding="utf-8")
    dominant = []
    for bow in corpus:
        dist = lda.get_document_topics(bow, minimum_probability=0)
        topic, prob = max(dist, key=lambda x: x[1])
        dominant.append((topic, prob))
    df["topic_lda"] = [x[0] for x in dominant]
    df["proba_topic_lda"] = [x[1] for x in dominant]
    df.to_csv(outdir / "lda_articles.csv", index=False, encoding="utf-8")
    majority_theme_mapping(df, "topic_lda").to_csv(
        outdir / "lda_topic_theme.csv", index=False, encoding="utf-8"
    )
    lda.save(str(outdir / "lda_model.gensim"))
    dictionary.save(str(outdir / "lda_dictionary.gensim"))
    pd.DataFrame([{"coherence_cv": coherence, "topics": args.topics}]).to_csv(outdir / "lda_coherence.csv", index=False)
    print(f"LDA c_v coherence = {coherence:.3f}")

if __name__ == "__main__":
    main()
