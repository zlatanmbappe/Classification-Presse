"""Visualisations complémentaires pour comparer les trois méthodes.

Les graphiques sont générés en SVG pour ne pas ajouter de dépendance graphique.
"""
from __future__ import annotations

import html
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def _esc(value: object) -> str:
    return html.escape(str(value))


def _write(path: Path, parts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _line_chart(title: str, x_values: list[int], y_values: list[float], x_label: str, y_label: str, path: Path, note: str = "") -> None:
    width, height = 900, 500
    left, right, top, bottom = 90, 45, 75, 80
    plot_w, plot_h = width-left-right, height-top-bottom
    if not x_values or not y_values:
        return
    y_min = min(y_values)
    y_max = max(y_values)
    if y_max == y_min:
        y_max = y_min + 1.0
    pad = (y_max-y_min)*0.08
    y_min -= pad
    y_max += pad

    def xp(x):
        if max(x_values) == min(x_values):
            return left + plot_w/2
        return left + (x-min(x_values))/(max(x_values)-min(x_values))*plot_w

    def yp(y):
        return top + (y_max-y)/(y_max-y_min)*plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="28" y="38" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#172033">{_esc(title)}</text>',
    ]
    for i in range(6):
        value = y_min + (y_max-y_min)*i/5
        y = yp(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#667085">{value:.3f}</text>')
    pts = " ".join(f'{xp(x):.1f},{yp(y):.1f}' for x,y in zip(x_values,y_values))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#3157d5" stroke-width="3"/>')
    for x,y in zip(x_values,y_values):
        parts.append(f'<circle cx="{xp(x):.1f}" cy="{yp(y):.1f}" r="5" fill="#3157d5"/>')
        parts.append(f'<text x="{xp(x):.1f}" y="{top+plot_h+24}" text-anchor="middle" font-family="Arial" font-size="11" fill="#667085">{x}</text>')
    parts.append(f'<text x="{left+plot_w/2:.1f}" y="{height-35}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700" fill="#344054">{_esc(x_label)}</text>')
    parts.append(f'<text x="22" y="{top+plot_h/2:.1f}" transform="rotate(-90 22 {top+plot_h/2:.1f})" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700" fill="#344054">{_esc(y_label)}</text>')
    if note:
        parts.append(f'<text x="{width-28}" y="{height-12}" text-anchor="end" font-family="Arial" font-size="10" fill="#667085">{_esc(note)}</text>')
    parts.append('</svg>')
    _write(path, parts)


def _scatter_pca(outputs: Path) -> None:
    data = joblib.load(outputs/'tfidf.joblib')
    X = data['X']
    df = pd.read_csv(outputs/'kmeans_articles.csv')
    coords = PCA(n_components=2, random_state=42).fit_transform(X.toarray())
    df['pca_x'] = coords[:,0]
    df['pca_y'] = coords[:,1]
    df[['rubrique','cluster_kmeans','pca_x','pca_y']].to_csv(outputs/'kmeans_pca.csv', index=False, encoding='utf-8')

    mapping_path = outputs/'kmeans_cluster_theme.csv'
    mapping = {}
    if mapping_path.exists():
        m = pd.read_csv(mapping_path)
        mapping = {int(r['cluster_kmeans']): str(r['theme']) for _,r in m.iterrows()}

    width, height = 900, 560
    left, right, top, bottom = 70, 210, 70, 65
    plot_w, plot_h = width-left-right, height-top-bottom
    xs, ys = coords[:,0], coords[:,1]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if xmax == xmin: xmax += 1
    if ymax == ymin: ymax += 1
    def xp(v): return left + (v-xmin)/(xmax-xmin)*plot_w
    def yp(v): return top + (ymax-v)/(ymax-ymin)*plot_h
    palette = ['#3157d5','#17a673','#d97706','#8b5cf6','#dc2626','#0891b2','#64748b','#db2777','#4f46e5','#65a30d']
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/>',
             '<text x="28" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#172033">K-means — projection PCA des articles</text>',
             f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfcfe" stroke="#e5e7eb"/>']
    for i,(x,y) in enumerate(coords):
        cluster = int(df.iloc[i]['cluster_kmeans'])
        parts.append(f'<circle cx="{xp(x):.1f}" cy="{yp(y):.1f}" r="6" fill="{palette[cluster%len(palette)]}" fill-opacity="0.78" stroke="white" stroke-width="1"/>')
    for idx, cluster in enumerate(sorted(df['cluster_kmeans'].unique())):
        y = top + 26 + idx*34
        color = palette[int(cluster)%len(palette)]
        label = f"Cluster {int(cluster)}"
        if int(cluster) in mapping: label += f" → {mapping[int(cluster)]}"
        parts.append(f'<circle cx="{left+plot_w+28}" cy="{y-5}" r="6" fill="{color}"/>')
        parts.append(f'<text x="{left+plot_w+42}" y="{y}" font-family="Arial" font-size="12" fill="#344054">{_esc(label)}</text>')
    parts += [f'<text x="{left+plot_w/2}" y="{height-28}" text-anchor="middle" font-family="Arial" font-size="12" fill="#667085">Composante principale 1</text>',
              f'<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20 {top+plot_h/2})" text-anchor="middle" font-family="Arial" font-size="12" fill="#667085">Composante principale 2</text>',
              '</svg>']
    _write(outputs/'figures'/'kmeans_pca.svg', parts)


def _terms_panels(title: str, groups: list[tuple[str,list[tuple[str,float]]]], path: Path) -> None:
    cols = 2
    panel_w, panel_h = 430, 190
    rows = (len(groups)+cols-1)//cols
    width, height = 900, 70 + rows*panel_h + 35
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/>',
             f'<text x="28" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#172033">{_esc(title)}</text>']
    for gi,(label,terms) in enumerate(groups):
        col, row = gi%cols, gi//cols
        x0, y0 = 25+col*panel_w, 68+row*panel_h
        parts.append(f'<rect x="{x0}" y="{y0}" width="{panel_w-18}" height="{panel_h-16}" rx="12" fill="#fbfcfe" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x0+14}" y="{y0+25}" font-family="Arial" font-size="14" font-weight="700" fill="#172033">{_esc(label)}</text>')
        maxv = max((v for _,v in terms), default=1.0) or 1.0
        for ti,(term,value) in enumerate(terms[:6]):
            y = y0+43+ti*20
            parts.append(f'<text x="{x0+14}" y="{y+10}" font-family="Arial" font-size="10" fill="#344054">{_esc(term[:28])}</text>')
            bx = x0+135; bw = 245
            parts.append(f'<rect x="{bx}" y="{y+1}" width="{bw}" height="11" rx="5" fill="#eef1f6"/>')
            parts.append(f'<rect x="{bx}" y="{y+1}" width="{bw*(value/maxv):.1f}" height="11" rx="5" fill="#3157d5"/>')
    parts.append('</svg>')
    _write(path, parts)


def _kmeans_terms(outputs: Path) -> None:
    data = joblib.load(outputs/'tfidf.joblib')
    vectorizer = data['vectorizer']
    model = joblib.load(outputs/'kmeans_model.joblib')
    names = vectorizer.get_feature_names_out()
    mapping = {}
    mp = outputs/'kmeans_cluster_theme.csv'
    if mp.exists():
        m = pd.read_csv(mp); mapping = {int(r['cluster_kmeans']):str(r['theme']) for _,r in m.iterrows()}
    groups=[]; csv_rows=[]
    for cluster,center in enumerate(model.cluster_centers_):
        order = center.argsort()[::-1][:10]
        terms=[(str(names[i]),float(center[i])) for i in order]
        label=f"Cluster {cluster}" + (f" → {mapping[cluster]}" if cluster in mapping else '')
        groups.append((label,terms))
        for rank,(term,weight) in enumerate(terms,1): csv_rows.append({'cluster':cluster,'theme_associe':mapping.get(cluster,''),'rang':rank,'terme':term,'poids':weight})
    pd.DataFrame(csv_rows).to_csv(outputs/'kmeans_top_terms_weights.csv',index=False,encoding='utf-8')
    _terms_panels('K-means — mots les plus représentatifs',groups,outputs/'figures'/'kmeans_mots.svg')


def _lda_terms(outputs: Path) -> None:
    model = LdaModel.load(str(outputs/'lda_model.gensim'))
    mapping={}
    mp=outputs/'lda_topic_theme.csv'
    if mp.exists():
        m=pd.read_csv(mp); mapping={int(r['topic_lda']):str(r['theme']) for _,r in m.iterrows()}
    groups=[]; csv_rows=[]
    for topic in range(model.num_topics):
        terms=[(str(w),float(v)) for w,v in model.show_topic(topic,topn=10)]
        label=f"Topic {topic}" + (f" → {mapping[topic]}" if topic in mapping else '')
        groups.append((label,terms))
        for rank,(term,weight) in enumerate(terms,1): csv_rows.append({'topic':topic,'theme_associe':mapping.get(topic,''),'rang':rank,'terme':term,'poids':weight})
    pd.DataFrame(csv_rows).to_csv(outputs/'lda_top_terms_weights.csv',index=False,encoding='utf-8')
    _terms_panels('LDA — mots les plus représentatifs des topics',groups,outputs/'figures'/'lda_mots.svg')


def _confusion(outputs: Path) -> None:
    pred_file=outputs/'supervised_predictions_cv.csv'
    if not pred_file.exists(): return
    df=pd.read_csv(pred_file)
    labels=sorted(set(df['rubrique'].astype(str)) | set(df['prediction_cv'].astype(str)))
    cm=confusion_matrix(df['rubrique'].astype(str),df['prediction_cv'].astype(str),labels=labels)
    pd.DataFrame(cm,index=labels,columns=labels).to_csv(outputs/'logistic_confusion_matrix.csv',encoding='utf-8')
    n=len(labels); cell=72; left=150; top=115; width=left+n*cell+45; height=top+n*cell+80; maxv=max(int(cm.max()),1)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/>','<text x="28" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#172033">Régression logistique — matrice de confusion CV</text>','<text x="28" y="62" font-family="Arial" font-size="11" fill="#667085">Lignes = thème réel · colonnes = thème prédit en validation croisée</text>']
    for j,label in enumerate(labels):
        x=left+j*cell+cell/2
        parts.append(f'<text x="{x:.1f}" y="{top-12}" text-anchor="middle" font-family="Arial" font-size="10" fill="#344054" transform="rotate(-35 {x:.1f} {top-12})">{_esc(label)}</text>')
    for i,label in enumerate(labels):
        y=top+i*cell+cell/2+4
        parts.append(f'<text x="{left-12}" y="{y:.1f}" text-anchor="end" font-family="Arial" font-size="10" fill="#344054">{_esc(label)}</text>')
        for j in range(n):
            v=int(cm[i,j]); opacity=0.10+0.80*v/maxv
            x=left+j*cell; yy=top+i*cell
            parts.append(f'<rect x="{x}" y="{yy}" width="{cell-2}" height="{cell-2}" rx="6" fill="#3157d5" fill-opacity="{opacity:.2f}"/>')
            color='white' if opacity>0.55 else '#172033'
            parts.append(f'<text x="{x+cell/2:.1f}" y="{yy+cell/2+5:.1f}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="{color}">{v}</text>')
    parts.append('</svg>'); _write(outputs/'figures'/'logistic_confusion_matrix.svg',parts)


def _elbow(outputs: Path) -> None:
    data=joblib.load(outputs/'tfidf.joblib'); X=data['X']; n=X.shape[0]
    ks=list(range(2,min(10,n-1)+1)); inertias=[]
    for k in ks:
        model=KMeans(n_clusters=k,random_state=42,n_init='auto').fit(X)
        inertias.append(float(model.inertia_))
    pd.DataFrame({'k':ks,'inertie':inertias}).to_csv(outputs/'kmeans_elbow.csv',index=False)
    _line_chart('K-means — courbe du coude',ks,inertias,'Nombre de clusters K','Inertie',outputs/'figures'/'kmeans_elbow.svg','Le coude aide à apprécier un nombre de clusters raisonnable.')


def _coherence_curve(outputs: Path) -> None:
    pre=outputs/'corpus_pretraite.csv'; dpath=outputs/'lda_dictionary.gensim'
    if not pre.exists() or not dpath.exists(): return
    df=pd.read_csv(pre); texts=[str(t).split() for t in df['texte_pretraite'].fillna('')]
    dictionary=Dictionary.load(str(dpath)); corpus=[dictionary.doc2bow(t) for t in texts]
    ks=list(range(2,min(10,max(2,len(texts)-1))+1)); scores=[]
    for k in ks:
        model=LdaModel(corpus=corpus,id2word=dictionary,num_topics=k,random_state=42,passes=5,alpha='auto',eta='auto')
        score=float(CoherenceModel(model=model,texts=texts,dictionary=dictionary,coherence='c_v').get_coherence())
        scores.append(score)
    pd.DataFrame({'topics':ks,'coherence_cv':scores}).to_csv(outputs/'lda_coherence_curve.csv',index=False)
    _line_chart('LDA — cohérence selon le nombre de topics',ks,scores,'Nombre de topics','Cohérence c_v',outputs/'figures'/'lda_coherence_curve.svg','Plus la cohérence est élevée, plus les mots d’un topic sont cohérents entre eux.')


def build_all_visualizations(outputs_dir: str | Path='outputs') -> list[str]:
    """Génère les visualisations complémentaires utilisées par l'interface et le rapport."""
    outputs=Path(outputs_dir); (outputs/'figures').mkdir(parents=True,exist_ok=True)
    generated=[]
    jobs=[
        ('kmeans_pca.svg',lambda:_scatter_pca(outputs)),
        ('kmeans_mots.svg',lambda:_kmeans_terms(outputs)),
        ('lda_mots.svg',lambda:_lda_terms(outputs)),
        ('logistic_confusion_matrix.svg',lambda:_confusion(outputs)),
        ('kmeans_elbow.svg',lambda:_elbow(outputs)),
        ('lda_coherence_curve.svg',lambda:_coherence_curve(outputs)),
    ]
    for filename,func in jobs:
        func()
        if (outputs/'figures'/filename).exists(): generated.append(filename)
    return generated
