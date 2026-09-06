"""Stockage des articles dans PostgreSQL."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .db import get_conn


@dataclass
class Article:
    url: str
    titre: str
    date: Optional[str]
    auteur: Optional[str]
    rubrique: Optional[str]
    contenu_brut: str
    contenu_nettoye: str
    checksum: str
    site: Optional[str] = None
    encodage: str = "UTF-8"
    format_source: str = "HTML"


def save_article(article: Article) -> bool:
    """Insère un article. URL et checksum empêchent les doublons."""
    sql = """
    INSERT INTO articles
      (url, site, titre, date, auteur, rubrique, contenu_brut, contenu_nettoye,
       checksum, encodage, format_source)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT DO NOTHING;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    article.url,
                    article.site,
                    article.titre,
                    article.date,
                    article.auteur,
                    article.rubrique,
                    article.contenu_brut,
                    article.contenu_nettoye,
                    article.checksum,
                    article.encodage,
                    article.format_source,
                ),
            )
            return cur.rowcount == 1
