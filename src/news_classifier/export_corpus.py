"""Export PostgreSQL -> CSV pour les expériences NLP."""
from __future__ import annotations

import argparse

import pandas as pd

from .db import get_conn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/corpus.csv")
    args = parser.parse_args()

    sql = """
    SELECT id, url, site, titre, date, auteur, rubrique, contenu_nettoye
    FROM articles
    ORDER BY id
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

    df = pd.DataFrame(rows, columns=columns)
    df = df.dropna(subset=["contenu_nettoye", "rubrique"])
    df = df[df["rubrique"].astype(str).str.strip().ne("")]
    df = df.drop_duplicates(subset=["contenu_nettoye"])
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Exported {len(df)} articles to {args.out}")


if __name__ == "__main__":
    main()
