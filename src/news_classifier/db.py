"""Connexion PostgreSQL."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_database_url() -> str:
    """Retourne DATABASE_URL depuis .env."""
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is missing. Copy .env.example to .env and edit it."
        )
    return url


@contextmanager
def get_conn() -> Iterator[psycopg2.extensions.connection]:
    """Ouvre une connexion, commit si succès, rollback si erreur, puis ferme."""
    conn = psycopg2.connect(get_database_url())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
