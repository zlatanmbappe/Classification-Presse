-- Schéma PostgreSQL du projet de classification thématique.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS articles (
    id              BIGSERIAL PRIMARY KEY,
    url             TEXT NOT NULL UNIQUE,
    site            TEXT,
    titre           TEXT NOT NULL,
    date            TIMESTAMPTZ,
    auteur          TEXT,
    rubrique        TEXT,
    contenu_brut    TEXT NOT NULL,
    contenu_nettoye TEXT NOT NULL,
    checksum        CHAR(64) NOT NULL UNIQUE,
    date_import     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    encodage        VARCHAR(40) DEFAULT 'UTF-8',
    format_source   VARCHAR(40) DEFAULT 'HTML'
);

-- Migration douce pour une base créée avec une ancienne version du projet.
ALTER TABLE articles ADD COLUMN IF NOT EXISTS site TEXT;

CREATE INDEX IF NOT EXISTS idx_articles_rubrique ON articles(rubrique);
CREATE INDEX IF NOT EXISTS idx_articles_site ON articles(site);
CREATE INDEX IF NOT EXISTS idx_articles_date ON articles(date);
CREATE INDEX IF NOT EXISTS idx_articles_checksum ON articles(checksum);
CREATE INDEX IF NOT EXISTS idx_articles_date_import ON articles(date_import);

CREATE TABLE IF NOT EXISTS experiments (
    id BIGSERIAL PRIMARY KEY,
    nom TEXT NOT NULL,
    algorithme TEXT NOT NULL,
    parametres JSONB,
    nb_documents INTEGER,
    nb_termes INTEGER,
    metriques JSONB,
    date_execution TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
