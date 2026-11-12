#!/usr/bin/env python3
"""Ejercicio de las diapositivas 10 y 11: clusters y proyección exploratoria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Resuelve las rutas desde la ubicación del proyecto, incluso si se ejecuta
# el script desde otra carpeta de la terminal.
ROOT = Path(__file__).resolve().parent


def main() -> None:
    # Recibe el JSON de embeddings, la ruta opcional de salida y la cantidad
    # de grupos que KMeans intentará formar.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--entrada", type=Path, default=ROOT / "resultados/tfidf.json", help="JSON producido por 03_representaciones.py con embeddings")
    parser.add_argument("--salida", type=Path, default=ROOT / "resultados/clusters.json", help="Ruta del JSON de resultados")
    parser.add_argument("--clusters", type=int, default=4, help="Cantidad de clusters")
    args = parser.parse_args()

    # Importa las dependencias aquí para poder comunicar cuál falta y cómo
    # instalarla sin ocultar el error original.
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise SystemExit("Falta scikit-learn. Instala: python3 -m pip install scikit-learn") from exc
    try:
        from umap.umap_ import UMAP
    except ImportError as exc:
        raise SystemExit("Falta UMAP. Instala: python3 -m pip install umap-learn") from exc

    # Lee los vectores generados por 03_representaciones.py y comprueba que la
    # entrada contenga embeddings antes de continuar.
    data = json.loads(args.entrada.read_text(encoding="utf-8"))
    if "embeddings" not in data:
        raise SystemExit("La entrada no contiene embeddings. Vuelve a ejecutar 03_representaciones.py con --modelo.")
    # Cada vector representa un documento. KMeans requiere al menos dos
    # clusters y menos clusters que documentos disponibles.
    vectors = data["embeddings"]
    if args.clusters < 2 or args.clusters >= len(vectors):
        raise SystemExit("--clusters debe ser mayor que 1 y menor que el número de documentos")

    # KMeans asigna una etiqueta de cluster a cada documento. UMAP proyecta
    # los vectores a dos dimensiones para visualizarlos; esa proyección ayuda
    # a explorar, pero no determina por sí sola el significado de los grupos.
    labels = KMeans(n_clusters=args.clusters, random_state=42, n_init=10).fit_predict(vectors).tolist()
    # NumPy convierte el resultado de UMAP a una lista serializable por JSON.
    import numpy as np

    projection = np.asarray(UMAP(n_components=2, random_state=42).fit_transform(vectors)).tolist()
    result = {"archivos": data["archivos"], "cluster": labels, "proyeccion_2d": projection, "nota": "La proyección es exploratoria; revisar el texto antes de asignar significado al cluster."}

    # Guarda el JSON si se indicó una ruta, creando primero sus directorios;
    # si no, presenta los resultados directamente en la terminal.
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(payload + "\n", encoding="utf-8")
        print(f"Escrito: {args.salida}")
    else:
        print(payload)


# Ejecuta el análisis solo al iniciar este archivo directamente.
if __name__ == "__main__":
    main()
