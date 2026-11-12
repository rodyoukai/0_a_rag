#!/usr/bin/env python3
"""Genera un mapa HTML interactivo a partir de clusters.json."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
# Los colores se asignan por grupo de forma cíclica si hay más clusters que
# tonos; las palabras frecuentes de función se excluyen al describirlos.
PALETTE = ["#d45c3f", "#167d72", "#c78c16", "#4567a8", "#8b5a9f", "#567d3e", "#bd4f82", "#57737c"]
STOPWORDS = set("""
a al algo ante bajo cada casi como con contra cual cuando de del desde donde durante e el ella ellas ellos en entre era es esa ese eso esta estaba estaban estar este esto fue ha había han hasta hay la las le les lo los más mas me mi mis muy no nos o para pero por porque que se ser si sin sobre su sus también te tiene todo un una uno unos y ya
además ahora así aunque bien cada cuales cualquier cómo cuál cuáles cuyo cuya cuyos cuyas dicho dicha dicen donde entonces esta estas estos fue fueron hacer hace hacen hecho hechos mucha muchas mucho muchos nada nadie ni otra otras otro otros pues quien quienes según será son sido tiene tienen todas todos tras vez veces
excelsior contenido pag página principal redacción redacción enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre
""".split())


def document_title(filename: str) -> str:
  # Recupera el encabezado del Markdown para que el mapa muestre títulos
  # comprensibles; usa el nombre del archivo como alternativa.
    source = ROOT / "notas_de_prensa/seleccion_taller/corpus_practica" / filename
    if source.is_file():
        for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
            heading = line.strip()
            if heading.startswith("# "):
                return heading[2:].strip()
    return Path(filename).stem


def build_html(points: list[dict[str, object]], colors: dict[str, str], cluster_terms: dict[str, list[dict[str, object]]]) -> str:
  # Serializa los datos que consumirán los controles del navegador. Escapar
  # el signo < evita que un texto fuente pueda iniciar una etiqueta HTML.
    safe_points = json.dumps(points, ensure_ascii=False).replace("<", "\\u003c")
    safe_colors = json.dumps(colors)
    safe_terms = json.dumps(cluster_terms, ensure_ascii=False).replace("<", "\\u003c")
    # La página se genera como un único archivo: CSS para el diseño, HTML para
    # los paneles y JavaScript para dibujar y filtrar los documentos.
    template = r'''<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mapa de similitud de notas</title>
  <style>
    /* Variables y reglas globales mantienen colores y dimensiones coherentes. */
    :root { color-scheme: light; --ink: #14251f; --muted: #60716a; --line: #d7dfd9; --paper: #f3f6f2; --white: #fff; --accent: #167d72; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--paper); color: var(--ink); font-family: Georgia, "Times New Roman", serif; }
    header { padding: 26px clamp(18px, 4vw, 56px) 20px; border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; align-items: end; gap: 20px; }
    .eyebrow { margin: 0 0 8px; color: var(--accent); font: 700 11px/1.2 Arial, sans-serif; letter-spacing: 1.2px; text-transform: uppercase; }
    h1 { margin: 0; font-size: clamp(28px, 4vw, 44px); font-weight: 500; line-height: 1.05; }
    .summary { color: var(--muted); font: 14px/1.5 Arial, sans-serif; text-align: right; white-space: nowrap; }
    main { padding: 22px clamp(18px, 4vw, 56px) 36px; }
    .layout { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 22px; align-items: start; }
    .chart-panel { min-width: 0; background: var(--white); border: 1px solid var(--line); }
    .chart-heading { padding: 15px 18px 10px; display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
    h2 { margin: 0; font-size: 18px; font-weight: 500; }
    .chart-heading span, .hint, .side-label { color: var(--muted); font: 12px/1.5 Arial, sans-serif; }
    .plot-wrap { position: relative; padding: 0 10px 10px; }
    canvas { display: block; width: 100%; height: 520px; touch-action: manipulation; }
    .tooltip { position: absolute; z-index: 2; display: none; max-width: 290px; padding: 10px 12px; background: var(--ink); color: white; pointer-events: none; font: 12px/1.45 Arial, sans-serif; box-shadow: 0 4px 16px #14251f33; }
    .tooltip strong { display: block; margin-bottom: 4px; font-size: 13px; }
    .hint { padding: 0 18px 14px; }
    .terms-panel { padding: 15px 18px 18px; border-top: 1px solid var(--line); }
    .terms-heading { display: flex; justify-content: space-between; align-items: center; gap: 16px; }
    .terms-heading h2 { font-size: 18px; }
    .terms-select { width: auto; min-width: 135px; height: 34px; }
    .terms-explanation { margin: 6px 0 13px; color: var(--muted); font: 12px/1.5 Arial, sans-serif; }
    .term-list { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 22px; }
    .term-row { min-width: 0; }
    .term-meta { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; font: 12px/1.4 Arial, sans-serif; }
    .term-meta strong { overflow-wrap: anywhere; font-size: 13px; }
    .term-meta span { flex: 0 0 auto; color: var(--muted); font-size: 11px; text-align: right; }
    .term-track { height: 5px; margin-top: 4px; background: #e7ece8; }
    .term-track i { display: block; height: 100%; min-width: 2px; }
    .term-empty { grid-column: 1 / -1; color: var(--muted); font: 13px/1.5 Arial, sans-serif; }
    .glossary { margin-top: 14px; border-top: 1px solid var(--line); padding-top: 12px; }
    .glossary summary { color: var(--ink); cursor: pointer; font: 700 12px Arial, sans-serif; }
    .glossary dl { display: grid; grid-template-columns: minmax(100px, 0.22fr) 1fr; gap: 8px 14px; margin: 12px 0 2px; font: 12px/1.5 Arial, sans-serif; }
    .glossary dt { color: var(--accent); font-weight: 700; }
    .glossary dd { margin: 0; color: var(--muted); }
    aside { border-top: 3px solid var(--ink); }
    .side-head { padding: 13px 0 9px; display: flex; justify-content: space-between; align-items: baseline; }
    .side-head h2 { font-size: 18px; }
    .legend { display: grid; gap: 4px; margin-bottom: 17px; }
    .legend button { width: 100%; border: 0; background: transparent; padding: 8px 7px; color: var(--ink); display: flex; align-items: center; gap: 9px; text-align: left; cursor: pointer; font: 13px Arial, sans-serif; }
    .legend button:hover, .legend button[aria-pressed="false"] { background: #e6ece7; }
    .swatch { width: 11px; height: 11px; flex: 0 0 11px; border-radius: 50%; }
    .count { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }
    label { display: block; margin: 0 0 6px; font: 700 11px Arial, sans-serif; text-transform: uppercase; color: var(--muted); }
    input { width: 100%; height: 38px; border: 1px solid #b9c5bd; border-radius: 0; padding: 0 10px; background: white; color: var(--ink); font: 14px Arial, sans-serif; }
    input:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
    .doc-list { margin-top: 9px; max-height: 310px; overflow: auto; border-top: 1px solid var(--line); }
    .doc-list a { display: block; padding: 9px 4px; border-bottom: 1px solid var(--line); color: var(--ink); text-decoration: none; font: 13px/1.35 Arial, sans-serif; }
    .doc-list a:hover { color: var(--accent); }
    .doc-list small { display: block; color: var(--muted); margin-top: 3px; font-size: 11px; }
    .note { margin: 19px 0 0; padding-top: 12px; border-top: 1px solid var(--line); color: var(--muted); font: 12px/1.5 Arial, sans-serif; }
    /* Los puntos de quiebre apilan los paneles y reducen el gráfico en móvil. */
    @media (max-width: 800px) { header { align-items: start; flex-direction: column; } .summary { text-align: left; white-space: normal; } .layout { grid-template-columns: 1fr; } canvas { height: 430px; } aside { border-top-width: 2px; } .doc-list { max-height: 240px; } }
    @media (max-width: 560px) { .term-list { grid-template-columns: 1fr; } .term-meta span { font-size: 10px; } }
    @media (max-width: 460px) { .chart-heading { align-items: start; flex-direction: column; gap: 3px; } .terms-heading { align-items: start; flex-direction: column; gap: 9px; } canvas { height: 350px; } }
  </style>
</head>
<body>
  <!-- El encabezado, el mapa y la columna lateral conforman la vista principal. -->
  <header>
    <div><p class="eyebrow">Exploración del corpus</p><h1>Mapa de similitud</h1></div>
    <div class="summary" id="summary"></div>
  </header>
  <main>
    <div class="layout">
      <section class="chart-panel" aria-labelledby="map-title">
        <div class="chart-heading"><h2 id="map-title">Proyección UMAP de los embeddings</h2><span>Un punto representa una nota</span></div>
        <div class="plot-wrap" id="plot-wrap"><canvas id="plot" aria-label="Mapa de notas agrupadas por cluster"></canvas><div class="tooltip" id="tooltip"></div></div>
        <div class="hint">Pasa sobre un punto para ver la nota. Los ejes muestran la proyección 2D; sus distancias son exploratorias.</div>
        <section class="terms-panel" aria-labelledby="terms-title">
          <div class="terms-heading"><h2 id="terms-title">Términos que distinguen al cluster</h2><select class="terms-select" id="term-cluster" aria-label="Cluster para analizar términos"></select></div>
          <p class="terms-explanation">Contraste entre el TF-IDF medio normalizado del grupo y el resto. Solo se incluyen términos presentes en varias notas; la cuenta indica cuántas.</p>
          <div class="term-list" id="term-list" aria-live="polite"></div>
        </section>
        <details class="glossary" open>
          <summary>Glosario</summary>
          <dl>
            <dt>UMAP</dt><dd>Método que comprime los vectores de cada nota a dos coordenadas para dibujarlos. Ayuda a explorar similitudes; los ejes y sus distancias no son categorías ni una prueba de que los grupos sean correctos.</dd>
            <dt>Embedding</dt><dd>Vector numérico que representa una nota y recoge relaciones de significado aprendidas por un modelo.</dd>
            <dt>Cluster</dt><dd>Grupo de notas que el algoritmo reunió por la semejanza de sus embeddings. El número solo identifica el grupo; no le asigna un tema.</dd>
            <dt>TF-IDF</dt><dd>Peso de una palabra que combina cuánto aparece en una nota con qué tan poco común es en el conjunto. Aquí se usa para describir diferencias entre grupos, no para formar estos clusters.</dd>
            <dt>dif. pp</dt><dd>Diferencia en puntos porcentuales entre la proporción TF-IDF media del término dentro del cluster y la del resto. Un valor positivo indica que el término pesa más en ese cluster.</dd>
            <dt>Prevalencia</dt><dd>En cuántas notas del cluster aparece el término, respecto al total de notas del grupo.</dd>
          </dl>
        </details>
      </section>
      <aside>
        <div class="side-head"><h2>Grupos</h2><span class="side-label">Pulsa para filtrar</span></div>
        <div class="legend" id="legend"></div>
        <label for="search">Buscar nota</label>
        <input id="search" type="search" placeholder="Título o identificador">
        <div class="doc-list" id="doc-list"></div>
        <p class="note">Los grupos se calcularon sobre embeddings en alta dimensión. Esta vista 2D ayuda a explorarlos, pero no asigna por sí sola un tema a cada cluster.</p>
      </aside>
    </div>
  </main>
  <script>
    // Los marcadores se sustituyen con datos calculados en Python antes de
    // guardar el HTML; enabled contiene los clusters visibles actualmente.
    const points = __DATA__;
    const colors = __COLORS__;
    const clusterTerms = __TERMS__;
    const enabled = new Set(Object.keys(colors));
    const canvas = document.getElementById("plot");
    const ctx = canvas.getContext("2d");
    const tooltip = document.getElementById("tooltip");
    const search = document.getElementById("search");
    const termCluster = document.getElementById("term-cluster");
    let drawn = [];

    // Construye el resumen, la leyenda de filtros y el selector de términos.
    const counts = Object.fromEntries(Object.keys(colors).map((cluster) => [cluster, points.filter((p) => String(p.cluster) === cluster).length]));
    document.getElementById("summary").textContent = `${points.length} notas · ${Object.keys(colors).length} clusters · BAAI/bge-m3`;
    document.getElementById("legend").innerHTML = Object.keys(colors).map((cluster) =>
      `<button type="button" data-cluster="${cluster}" aria-pressed="true"><i class="swatch" style="background:${colors[cluster]}"></i>Cluster ${cluster}<span class="count">${counts[cluster]} notas</span></button>`
    ).join("");
    termCluster.innerHTML = Object.keys(colors).map((cluster) => `<option value="${cluster}">Cluster ${cluster}</option>`).join("");

    function renderTerms() {
      // Las barras comparan los términos informativos del cluster seleccionado.
      const cluster = termCluster.value;
      const terms = clusterTerms[cluster] || [];
      const maxScore = Math.max(...terms.map((item) => item.score), 0);
      document.getElementById("term-list").innerHTML = terms.length ? terms.map((item) => {
        const width = maxScore ? Math.max(2, item.score / maxScore * 100) : 0;
        return `<div class="term-row"><div class="term-meta"><strong>${escapeHtml(item.term)}</strong><span>${item.prevalence}/${item.total} notas · dif. ${(item.score * 100).toFixed(2)} pp</span></div><div class="term-track"><i style="width:${width}%;background:${colors[cluster]}"></i></div></div>`;
      }).join("") : '<div class="term-empty">No hay términos distintivos suficientes con el filtro de frecuencia actual.</div>';
    }

    function queryMatches(point) {
      // La búsqueda no distingue mayúsculas y consulta tanto título como ID.
      const query = search.value.trim().toLocaleLowerCase("es");
      return !query || `${point.title} ${point.file}`.toLocaleLowerCase("es").includes(query);
    }

    function renderList() {
      // La lista refleja simultáneamente los clusters habilitados y el texto buscado.
      const shown = points.filter((point) => enabled.has(String(point.cluster)) && queryMatches(point));
      document.getElementById("doc-list").innerHTML = shown.map((point) => {
        const href = `../notas_de_prensa/seleccion_taller/corpus_practica/${encodeURIComponent(point.file)}`;
        return `<a href="${href}" target="_blank" rel="noreferrer">${escapeHtml(point.title)}<small>Cluster ${point.cluster} · ${escapeHtml(point.file)}</small></a>`;
      }).join("") || '<div class="side-label" style="padding:12px 4px">No hay notas con ese filtro.</div>';
    }

    function escapeHtml(value) {
      // Los títulos y nombres proceden de documentos; se escapan antes de
      // insertarlos con innerHTML para evitar que se interpreten como marcado.
      return String(value).replace(/[&<>"']/g, (char) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", "\"":"&quot;", "'":"&#39;"}[char]));
    }

    function draw() {
      // Ajusta la resolución interna del canvas a la pantalla para evitar que
      // el gráfico se vea borroso en pantallas de alta densidad.
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);

      // Convierte las coordenadas UMAP al espacio del canvas y deja margen
      // para las etiquetas de los ejes.
      const pad = { left: 54, right: 20, top: 20, bottom: 42 };
      const xs = points.map((p) => p.x);
      const ys = points.map((p) => p.y);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const spanX = maxX - minX || 1, spanY = maxY - minY || 1;
      const px = (value) => pad.left + ((value - minX) / spanX) * (width - pad.left - pad.right);
      const py = (value) => height - pad.bottom - ((value - minY) / spanY) * (height - pad.top - pad.bottom);

      // Dibuja una cuadrícula con marcas numéricas para facilitar la lectura;
      // los ejes son coordenadas de la proyección, no categorías del corpus.
      ctx.font = "11px Arial, sans-serif";
      ctx.strokeStyle = "#e4eae5";
      ctx.fillStyle = "#68776f";
      ctx.lineWidth = 1;
      for (let tick = 0; tick <= 4; tick += 1) {
        const x = pad.left + tick * (width - pad.left - pad.right) / 4;
        const y = pad.top + tick * (height - pad.top - pad.bottom) / 4;
        ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, height - pad.bottom); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
        ctx.textAlign = "center"; ctx.fillText((minX + tick * spanX / 4).toFixed(1), x, height - 19);
        ctx.textAlign = "right"; ctx.fillText((maxY - tick * spanY / 4).toFixed(1), pad.left - 9, y + 4);
      }
      ctx.fillStyle = "#53645b"; ctx.textAlign = "center"; ctx.fillText("UMAP 1", (pad.left + width - pad.right) / 2, height - 2);
      ctx.save(); ctx.translate(13, height / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("UMAP 2", 0, 0); ctx.restore();

      // Guarda las posiciones dibujadas para calcular luego cuál punto está
      // bajo el puntero. Opacidad y tamaño indican filtro y coincidencia.
      drawn = [];
      points.forEach((point) => {
        const active = enabled.has(String(point.cluster));
        const match = queryMatches(point);
        const x = px(point.x), y = py(point.y);
        ctx.beginPath(); ctx.arc(x, y, match && active ? 7 : 6, 0, Math.PI * 2);
        ctx.fillStyle = colors[String(point.cluster)];
        ctx.globalAlpha = !active ? 0.08 : (match ? 0.9 : 0.16);
        ctx.fill();
        if (match && active) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke(); }
        drawn.push({ point, x, y, active });
      });
      ctx.globalAlpha = 1;
    }

    // Cada interacción actualiza los paneles relacionados para que filtros,
    // términos y listado representen siempre la misma selección.
    document.getElementById("legend").addEventListener("click", (event) => {
      const button = event.target.closest("button[data-cluster]");
      if (!button) return;
      const cluster = button.dataset.cluster;
      if (enabled.has(cluster)) enabled.delete(cluster); else enabled.add(cluster);
      button.setAttribute("aria-pressed", String(enabled.has(cluster)));
      termCluster.value = cluster;
      renderTerms();
      draw(); renderList();
    });
    termCluster.addEventListener("change", renderTerms);
    search.addEventListener("input", () => { draw(); renderList(); });
    // Busca el punto activo más cercano al puntero para mostrar su ficha breve.
    canvas.addEventListener("pointermove", (event) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left, y = event.clientY - rect.top;
      const nearest = drawn.filter((item) => item.active).map((item) => ({...item, distance: Math.hypot(item.x - x, item.y - y)})).sort((a, b) => a.distance - b.distance)[0];
      if (!nearest || nearest.distance > 14) { tooltip.style.display = "none"; canvas.style.cursor = "default"; return; }
      tooltip.innerHTML = `<strong>${escapeHtml(nearest.point.title)}</strong>Cluster ${nearest.point.cluster}<br>${escapeHtml(nearest.point.file)}`;
      tooltip.style.display = "block";
      tooltip.style.left = `${Math.min(x + 13, rect.width - 300)}px`;
      tooltip.style.top = `${Math.max(4, y - 15)}px`;
      canvas.style.cursor = "pointer";
    });
    canvas.addEventListener("pointerleave", () => { tooltip.style.display = "none"; });
    // Redibuja al cambiar el tamaño de la ventana y realiza el primer render.
    new ResizeObserver(draw).observe(canvas);
    renderList();
    renderTerms();
    draw();
  </script>
</body>
</html>
'''
    return template.replace("__DATA__", safe_points).replace("__COLORS__", safe_colors).replace("__TERMS__", safe_terms)


def load_cluster_terms(path: Path, filenames: list[str], labels: list[int]) -> dict[str, list[dict[str, object]]]:
  # Comprueba que la matriz TF-IDF corresponda exactamente a los documentos
  # agrupados; así no se comparan por error filas desalineadas.
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames or "archivo" not in reader.fieldnames:
            raise SystemExit(f"El CSV TF-IDF no contiene una columna 'archivo': {path}")
        terms = [field for field in reader.fieldnames if field != "archivo"]
        rows: dict[str, dict[str, float]] = {}
        for row in reader:
            filename = Path(row["archivo"]).name
            if filename in rows:
                raise SystemExit(f"Documento duplicado en el CSV TF-IDF: {filename}")
            values = {}
            for term in terms:
                try:
                    value = float(row[term] or 0)
                except (TypeError, ValueError) as exc:
                    raise SystemExit(f"Valor TF-IDF inválido para {filename}, término {term}.") from exc
                if not math.isfinite(value) or value < 0:
                    raise SystemExit(f"Valor TF-IDF inválido para {filename}, término {term}.")
                values[term] = value
            rows[filename] = values

    expected = {Path(filename).name for filename in filenames}
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        extra = sorted(set(rows) - expected)
        raise SystemExit(f"El CSV TF-IDF no coincide con los documentos del clustering. Faltan: {missing}; sobran: {extra}.")

    # Normaliza cada fila para comparar proporciones aunque los documentos
    # tengan longitudes distintas; una fila vacía permanece en ceros.
    normalized = {}
    for filename, values in rows.items():
        total = sum(values.values())
        normalized[filename] = {term: value / total for term, value in values.items()} if total else values

    # Para cada cluster, resta el peso medio del resto al peso medio interno.
    # Solo se presentan términos frecuentes y con diferencia positiva.
    cluster_terms = {}
    for cluster in sorted(set(labels)):
        member_names = [Path(filename).name for filename, label in zip(filenames, labels, strict=True) if label == cluster]
        other_names = [Path(filename).name for filename, label in zip(filenames, labels, strict=True) if label != cluster]
        min_prevalence = min(len(member_names), max(2, math.ceil(len(member_names) * 0.2)))
        ranked = []
        for term in terms:
            cleaned = term.casefold()
            if len(cleaned) < 3 or cleaned in STOPWORDS or not re.search(r"[a-záéíóúüñ]", cleaned, re.IGNORECASE):
                continue
            prevalence = sum(rows[filename][term] > 0 for filename in member_names)
            if prevalence < min_prevalence:
                continue
            inside = sum(normalized[filename][term] for filename in member_names) / len(member_names)
            outside = sum(normalized[filename][term] for filename in other_names) / len(other_names) if other_names else 0
            score = inside - outside
            if score > 0:
                ranked.append({"term": term, "score": score, "prevalence": prevalence, "total": len(member_names)})
        ranked.sort(key=lambda item: (-float(item["score"]), str(item["term"])))
        cluster_terms[str(cluster)] = ranked[:10]
    return cluster_terms


def main() -> None:
  # Define los dos insumos: coordenadas/etiquetas del clustering y pesos
  # TF-IDF para explicar qué palabras distinguen a cada grupo.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=ROOT / "resultados/clusters.json")
    parser.add_argument("--tfidf", type=Path, default=ROOT / "resultados/tfidf_tfidf.csv", help="Matriz TF-IDF generada por 03_representaciones.py")
    parser.add_argument("--salida", type=Path, default=ROOT / "resultados/clusters_visualizacion.html")
    args = parser.parse_args()

    # Verifica que las tres listas del JSON estén alineadas antes de combinar
    # cada archivo con su etiqueta y sus coordenadas.
    data = json.loads(args.entrada.read_text(encoding="utf-8"))
    filenames = data.get("archivos", [])
    labels = data.get("cluster", [])
    projection = data.get("proyeccion_2d", [])
    if not filenames or len(filenames) != len(labels) or len(filenames) != len(projection):
        raise SystemExit("El JSON debe tener archivos, cluster y proyeccion_2d con la misma cantidad de elementos.")

    # Convierte los datos a un formato pequeño y validado que consume la página.
    points = []
    for filename, label, coordinates in zip(filenames, labels, projection, strict=True):
        if len(coordinates) != 2 or not all(math.isfinite(float(value)) for value in coordinates):
            raise SystemExit(f"Coordenadas 2D inválidas para {filename}.")
        points.append({
            "file": Path(filename).name,
            "title": document_title(Path(filename).name),
            "cluster": int(label),
            "x": float(coordinates[0]),
            "y": float(coordinates[1]),
        })

    # Asigna un color estable a cada identificador y genera la página final.
    cluster_ids = sorted({str(point["cluster"]) for point in points}, key=int)
    colors = {cluster: PALETTE[index % len(PALETTE)] for index, cluster in enumerate(cluster_ids)}
    cluster_terms = load_cluster_terms(args.tfidf, filenames, labels)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(build_html(points, colors, cluster_terms), encoding="utf-8")
    print(f"Visualización escrita: {args.salida} ({len(points)} notas, {len(cluster_ids)} clusters)")


if __name__ == "__main__":
    main()