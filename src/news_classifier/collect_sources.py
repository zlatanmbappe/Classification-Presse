"""Collecte des articles définis dans config/sources.yaml."""

import argparse
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup

from .article_extraction import DEFAULT_USER_AGENT, extract_article, robots_allows
from .storage import Article, save_article
from .text_cleaning import checksum


def same_site(url1: str, url2: str) -> bool:
    """Retourne True si deux URLs appartiennent au même site."""
    site1 = urlparse(url1).netloc.lower().removeprefix("www.")
    site2 = urlparse(url2).netloc.lower().removeprefix("www.")
    return site1 == site2


def discover_links(page_url: str) -> list[str]:
    """Récupère des liens candidats depuis une page de rubrique."""
    if not robots_allows(page_url):
        raise RuntimeError("Accès refusé par robots.txt.")

    response = requests.get(
        page_url,
        headers={"User-Agent": DEFAULT_USER_AGENT, "Accept-Language": "fr-FR,fr;q=0.9"},
        timeout=20,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "lxml")

    links = []
    seen = set()

    # Les liens placés dans main sont en général les plus utiles sur une page de rubrique.
    container = soup.find("main") or soup
    for tag in container.find_all("a", href=True):
        url = urljoin(page_url, tag["href"])
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            continue
        if not same_site(url, page_url):
            continue
        if url in seen:
            continue
        if parsed.path in ("", "/"):
            continue
        if any(word in parsed.path.lower() for word in (
            "/tag/", "/auteur/", "/author/", "/contact", "/newsletter",
            "/login", "/connexion", "/search", "/recherche",
        )):
            continue

        seen.add(url)
        links.append(url)
        if len(links) >= 80:
            break

    return links


def save_url(url: str, theme: str) -> bool:
    """Extrait une URL et l'enregistre dans PostgreSQL."""
    extracted = extract_article(url)
    article = Article(
        url=extracted.url,
        site=extracted.site,
        titre=extracted.titre,
        date=extracted.date,
        auteur=extracted.auteur,
        rubrique=theme,
        contenu_brut=extracted.contenu_brut,
        contenu_nettoye=extracted.contenu_nettoye,
        checksum=checksum(extracted.contenu_nettoye),
        encodage=extracted.encodage,
    )
    return save_article(article)


def collect_source(source: dict) -> tuple[int, int]:
    """Collecte une source jusqu'à atteindre sa limite d'articles."""
    page_url = source["url"]
    theme = source["theme"]
    limit = int(source.get("limit", 10))

    if source.get("type", "page") == "article":
        links = [page_url]
    else:
        links = discover_links(page_url)

    added = 0
    tested = 0
    for url in links:
        if added >= limit:
            break
        tested += 1
        try:
            if save_url(url, theme):
                added += 1
                print(f"  [AJOUTÉ] {theme}: {url}")
        except Exception as exc:
            print(f"  [IGNORÉ] {url} -> {exc}")

    return added, tested


def collect_from_config(config_path: str = "config/sources.yaml") -> tuple[int, int]:
    """Collecte toutes les sources présentes dans le fichier YAML."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    sources = config.get("sources", [])

    total_added = 0
    total_tested = 0
    for source in sources:
        print(f"\nSource : {source['theme']} — {source['url']}")
        added, tested = collect_source(source)
        total_added += added
        total_tested += tested

    return total_added, total_tested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sources.yaml")
    args = parser.parse_args()

    added, tested = collect_from_config(args.config)
    print(f"\nTerminé : {added} article(s) ajouté(s), {tested} URL(s) testée(s).")


if __name__ == "__main__":
    main()
