"""Comparaison des trois méthodes sur le corpus et sur une nouvelle URL."""
from __future__ import annotations

import csv
import html
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def majority_theme_mapping(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Associe chaque cluster/topic au thème de référence majoritaire."""
    rows = []
    clean = df.dropna(subset=[group_col, "rubrique"]).copy()
    for group_id, group in clean.groupby(group_col):
        counts = group["rubrique"].astype(str).value_counts()
        theme = str(counts.index[0])
        support = int(len(group))
        majority_count = int(counts.iloc[0])
        rows.append({
            group_col: int(group_id),
            "theme": theme,
            "support": support,
            "majority_count": majority_count,
            "majority_share": majority_count / support if support else 0.0,
        })
    return pd.DataFrame(rows)


def mapped_scores(df: pd.DataFrame, pred_col: str, mapping: pd.DataFrame) -> tuple[float, float]:
    """Calcule un accord indicatif avec les thèmes après association groupe -> thème."""
    map_dict = dict(zip(mapping[pred_col].astype(int), mapping["theme"].astype(str)))
    valid = df.dropna(subset=[pred_col, "rubrique"]).copy()
    y_true = valid["rubrique"].astype(str)
    y_pred = valid[pred_col].astype(int).map(map_dict).fillna("INCONNU")
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def _svg_bar_chart(title: str, rows: list[dict], path: Path, note: str = "") -> None:
    """Écrit un petit graphique SVG sans dépendance graphique supplémentaire."""
    width = 900
    left = 220
    right = 70
    top = 80
    row_h = 54
    bottom = 70 if note else 42
    height = top + len(rows) * row_h + bottom
    bar_w = width - left - right

    def esc(value: object) -> str:
        return html.escape(str(value))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="28" y="38" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#172033">{esc(title)}</text>',
    ]

    for tick in range(0, 101, 20):
        x = left + bar_w * tick / 100
        parts.append(f'<line x1="{x:.1f}" y1="{top-10}" x2="{x:.1f}" y2="{top + len(rows)*row_h - 12}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{top-18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#667085">{tick}%</text>')

    for i, row in enumerate(rows):
        y = top + i * row_h
        score = max(0.0, min(1.0, float(row.get("score", 0.0))))
        label = esc(row.get("label", ""))
        detail = esc(row.get("detail", ""))
        parts.append(f'<text x="28" y="{y+19}" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#172033">{label}</text>')
        if detail:
            parts.append(f'<text x="28" y="{y+38}" font-family="Arial, sans-serif" font-size="11" fill="#667085">{detail}</text>')
        parts.append(f'<rect x="{left}" y="{y+6}" width="{bar_w}" height="24" rx="12" fill="#eef1f6"/>')
        parts.append(f'<rect x="{left}" y="{y+6}" width="{bar_w*score:.1f}" height="24" rx="12" fill="#3157d5"/>')
        parts.append(f'<text x="{left + bar_w + 12}" y="{y+23}" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#172033">{score*100:.1f}%</text>')

    if note:
        parts.append(f'<text x="28" y="{height-28}" font-family="Arial, sans-serif" font-size="11" fill="#667085">{esc(note)}</text>')

    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")



def _svg_grouped_theme_chart(rows: list[dict], path: Path) -> None:
    """Compare les scores des trois méthodes pour chacun des six thèmes."""
    if not rows:
        return
    width, height = 980, 540
    left, right, top, bottom = 80, 35, 85, 105
    plot_w, plot_h = width-left-right, height-top-bottom
    methods = [("kmeans", "K-means", "#3157d5"), ("lda", "LDA", "#17a673"), ("logistic", "Régression", "#d97706")]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="28" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#172033">Nouvelle URL — scores par thème et par méthode</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + plot_h*(1-tick/100)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#667085">{tick}%</text>')
    group_w = plot_w/len(rows)
    bar_w = min(30, group_w/4.4)
    for i,row in enumerate(rows):
        center = left + group_w*(i+0.5)
        for mi,(key,label,color) in enumerate(methods):
            value=max(0.0,min(1.0,float(row.get(key,0.0))))
            x=center+(mi-1)*bar_w*1.15-bar_w/2
            h=plot_h*value
            y=top+plot_h-h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="4" fill="{color}"/>')
        theme = html.escape(str(row.get("theme", "")))
        parts.append(f'<text x="{center:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-family="Arial" font-size="11" fill="#344054">{theme}</text>')
    lx=left+plot_w-310
    ly=55
    for mi,(key,label,color) in enumerate(methods):
        x=lx+mi*105
        parts.append(f'<rect x="{x}" y="{ly}" width="12" height="12" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{x+18}" y="{ly+11}" font-family="Arial" font-size="11" fill="#344054">{html.escape(label)}</text>')
    parts.append(f'<text x="{left+plot_w/2:.1f}" y="{height-28}" text-anchor="middle" font-family="Arial" font-size="10" fill="#667085">K-means = similarité au cluster · LDA = poids de topic agrégé · Régression = probabilité. Comparaison descriptive.</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")

def build_corpus_comparison(outputs_dir: str | Path = "outputs") -> dict:
    """Crée les tableaux et graphiques de comparaison sur le corpus."""
    out = Path(outputs_dir)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    kmeans_file = out / "kmeans_articles.csv"
    lda_file = out / "lda_articles.csv"
    eval_file = out / "evaluation.csv"
    sup_file = out / "supervised_evaluation.csv"

    rows_native: list[dict] = []
    rows_reference: list[dict] = []

    if kmeans_file.exists():
        kdf = pd.read_csv(kmeans_file)
        kmapping = majority_theme_mapping(kdf, "cluster_kmeans")
        kmapping.to_csv(out / "kmeans_cluster_theme.csv", index=False, encoding="utf-8")
        acc, f1 = mapped_scores(kdf, "cluster_kmeans", kmapping)
        rows_reference.append({"method": "K-means", "accuracy_like": acc, "f1_macro_like": f1, "evaluation": "association post-hoc des clusters"})

    if lda_file.exists():
        ldf = pd.read_csv(lda_file)
        lmapping = majority_theme_mapping(ldf, "topic_lda")
        lmapping.to_csv(out / "lda_topic_theme.csv", index=False, encoding="utf-8")
        acc, f1 = mapped_scores(ldf, "topic_lda", lmapping)
        rows_reference.append({"method": "LDA", "accuracy_like": acc, "f1_macro_like": f1, "evaluation": "topic dominant + association post-hoc"})

    if eval_file.exists():
        ev = pd.read_csv(eval_file)
        for _, row in ev.iterrows():
            name = "K-means" if row["algorithme"] == "cluster_kmeans" else "LDA"
            rows_native.append({"method": name, "metric": "Pureté", "score": float(row["purete"])})
            rows_native.append({"method": name, "metric": "ARI", "score": float(row["ari"])})
            coherence = row.get("coherence_cv")
            if pd.notna(coherence):
                rows_native.append({"method": name, "metric": "Cohérence", "score": float(coherence)})

    if sup_file.exists():
        sdf = pd.read_csv(sup_file)
        if not sdf.empty:
            row = sdf.iloc[0]
            acc = float(row["accuracy_cv"])
            f1 = float(row["f1_macro_cv"])
            rows_native.append({"method": "Régression logistique", "metric": "Accuracy CV", "score": acc})
            rows_native.append({"method": "Régression logistique", "metric": "F1-macro CV", "score": f1})
            rows_reference.append({"method": "Régression logistique", "accuracy_like": acc, "f1_macro_like": f1, "evaluation": "validation croisée"})

    pd.DataFrame(rows_native).to_csv(out / "comparison_native_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(rows_reference).to_csv(out / "comparison_reference_scores.csv", index=False, encoding="utf-8")

    native_chart_rows = [
        {"label": f"{row['method']} — {row['metric']}", "score": row["score"]}
        for row in rows_native
    ]
    _svg_bar_chart(
        "Comparaison des métriques sur le corpus",
        native_chart_rows,
        figures / "comparaison_corpus.svg",
        note="Les métriques n'ont pas toutes la même signification : elles sont affichées côte à côte pour comparaison descriptive.",
    )

    ref_chart_rows = []
    for row in rows_reference:
        ref_chart_rows.append({"label": f"{row['method']} — accord/accuracy", "score": row["accuracy_like"], "detail": row["evaluation"]})
        ref_chart_rows.append({"label": f"{row['method']} — F1-macro", "score": row["f1_macro_like"], "detail": row["evaluation"]})
    _svg_bar_chart(
        "Comparaison indicative avec les thèmes de référence",
        ref_chart_rows,
        figures / "comparaison_corpus_reference.svg",
        note="K-means et LDA sont associés aux thèmes après coup ; la régression logistique est évaluée en validation croisée. Les scores ne sont donc pas strictement équivalents.",
    )

    return {"native": rows_native, "reference": rows_reference}


def save_url_comparison(result: dict, outputs_dir: str | Path = "outputs") -> Path:
    """Conserve le résultat des trois méthodes pour une URL et produit un graphique SVG."""
    out = Path(outputs_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "url_comparisons.csv"

    row = {
        "url": result["url"],
        "title": result["title"],
        "site": result["site"],
        "kmeans_theme": result["kmeans"]["theme"],
        "kmeans_cluster": result["kmeans"]["cluster"],
        "kmeans_similarity": result["kmeans"]["score"],
        "lda_theme": result["lda"]["theme"],
        "lda_topic": result["lda"]["topic"],
        "lda_probability": result["lda"]["score"],
        "logistic_theme": result["logistic"]["predicted"],
        "logistic_best_theme": result["logistic"]["best_theme"],
        "logistic_probability": result["logistic"]["best_score"],
        "logistic_margin": result["logistic"]["margin"],
        "logistic_uncertain": result["logistic"]["uncertain"],
    }
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    chart_rows = [
        {"label": f"K-means → {result['kmeans']['theme']}", "score": result["kmeans"]["score"], "detail": "similarité cosinus au centre du cluster"},
        {"label": f"LDA → {result['lda']['theme']}", "score": result["lda"]["score"], "detail": "poids du topic dominant"},
        {"label": f"Régression → {result['logistic']['predicted']}", "score": result["logistic"]["best_score"], "detail": "probabilité de la meilleure classe"},
    ]
    _svg_bar_chart(
        "Comparaison des trois méthodes sur la nouvelle URL",
        chart_rows,
        out / "figures" / "comparaison_url.svg",
        note="Les trois barres représentent des scores de nature différente ; elles servent à confronter les décisions, pas à les interpréter comme une même probabilité.",
    )

    # Comparaison détaillée des six thèmes pour les trois méthodes.
    theme_scores = result.get("theme_scores", [])
    if theme_scores:
        pd.DataFrame(theme_scores).to_csv(out / "url_theme_scores.csv", index=False, encoding="utf-8")
        _svg_grouped_theme_chart(theme_scores, out / "figures" / "comparaison_url_par_theme.svg")
    return csv_path
