-- Requêtes de contrôle utiles pendant le projet.
SELECT COUNT(*) AS nb_articles FROM articles;
SELECT rubrique, COUNT(*) AS nb_articles FROM articles GROUP BY rubrique ORDER BY nb_articles DESC;
SELECT site, COUNT(*) AS nb_articles FROM articles GROUP BY site ORDER BY nb_articles DESC;
SELECT COUNT(*) AS doublons_checksum
FROM (SELECT checksum FROM articles GROUP BY checksum HAVING COUNT(*) > 1) d;
SELECT id, site, titre, rubrique, date, url
FROM articles
ORDER BY date_import DESC
LIMIT 20;
