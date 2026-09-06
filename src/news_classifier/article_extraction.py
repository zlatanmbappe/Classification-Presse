"""Extraction simple du contenu principal d'un article de presse."""

import os
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from .text_cleaning import clean_text

DEFAULT_USER_AGENT = "ProjetL3ClassificationPresse/1.0"


@dataclass
class ExtractedArticle:
    """Données utiles extraites d'une page web."""

    url: str
    site: str
    titre: str
    date: str | None
    auteur: str | None
    contenu_brut: str
    contenu_nettoye: str
    encodage: str


def robots_allows(url: str) -> bool:
    """Vérifie simplement si robots.txt autorise la lecture de l'URL."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    user_agent = os.getenv("USER_AGENT", DEFAULT_USER_AGENT)

    try:
        response = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=8)
        if response.status_code >= 400:
            return True
        robots = RobotFileParser()
        robots.parse(response.text.splitlines())
        return robots.can_fetch(user_agent, url)
    except requests.RequestException:
        # Si robots.txt est indisponible, on ne bloque pas toute l'application.
        return True


def _meta(soup: BeautifulSoup, name: str) -> str | None:
    """Lit une métadonnée HTML si elle existe."""
    tag = soup.find("meta", attrs={"property": name})
    if not tag:
        tag = soup.find("meta", attrs={"name": name})
    if tag and tag.get("content"):
        return str(tag.get("content")).strip()
    return None


def _paragraphs(container) -> list[str]:
    """Récupère les paragraphes suffisamment longs d'un bloc HTML."""
    if container is None:
        return []

    result = []
    for tag in container.find_all("p"):
        text = clean_text(tag.get_text(" ", strip=True))
        if len(text) >= 40:
            result.append(text)
    return result


def extract_article(url: str, min_words: int = 80) -> ExtractedArticle:
    """Télécharge une page et extrait son titre et son texte principal."""
    if not robots_allows(url):
        raise RuntimeError("Accès refusé par robots.txt pour cette URL.")

    user_agent = os.getenv("USER_AGENT", DEFAULT_USER_AGENT)
    response = requests.get(
        url,
        headers={"User-Agent": user_agent, "Accept-Language": "fr-FR,fr;q=0.9"},
        timeout=20,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "lxml")

    # Titre : métadonnée Open Graph, puis h1, puis balise title.
    title = _meta(soup, "og:title")
    if not title and soup.find("h1"):
        title = soup.find("h1").get_text(" ", strip=True)
    if not title and soup.title:
        title = soup.title.get_text(" ", strip=True)
    title = clean_text(title or "Titre non détecté")

    # Quelques métadonnées simples, utiles pour le stockage.
    date = _meta(soup, "article:published_time")
    if not date and soup.find("time"):
        date = soup.find("time").get("datetime")
    author = _meta(soup, "author")

    # On retire les parties de page qui ne font généralement pas partie de l'article.
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    # On essaie d'abord <article>, puis <main>, puis le corps entier.
    paragraphs = _paragraphs(soup.find("article"))
    if len(" ".join(paragraphs).split()) < min_words:
        paragraphs = _paragraphs(soup.find("main"))
    if len(" ".join(paragraphs).split()) < min_words:
        paragraphs = _paragraphs(soup.body)

    raw_text = " ".join(paragraphs)
    clean = clean_text(raw_text)
    word_count = len(clean.split())
    if word_count < min_words:
        raise RuntimeError(
            f"Impossible d'extraire assez de texte ({word_count} mots, minimum {min_words})."
        )

    parsed = urlparse(response.url)
    site = parsed.netloc.lower().removeprefix("www.")

    return ExtractedArticle(
        url=response.url,
        site=site,
        titre=title,
        date=date,
        auteur=author,
        contenu_brut=raw_text,
        contenu_nettoye=clean,
        encodage=response.encoding or "UTF-8",
    )
