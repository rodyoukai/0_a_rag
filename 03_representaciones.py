#!/usr/bin/env python3
"""Ejercicio de la diapositiva 9: bolsa de palabras, TF-IDF y embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


# Define la carpeta de práctica y una expresión regular que reconoce palabras
# con letras, dígitos y caracteres frecuentes del español.
ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "notas_de_prensa/seleccion_taller/corpus_practica"
DEFAULT_MODEL = "BAAI/bge-m3"
# Este patrón mantiene juntas palabras con tildes y la letra ñ; también admite
# números y guiones bajos por el uso de \w en expresiones regulares de Python.
TOKEN = re.compile(r"[\wáéíóúüñ]+", re.IGNORECASE)


def tokens(text: str) -> list[str]:
    # Extrae las palabras y las pasa a minúsculas para unificar variantes que
    # solo difieren en mayúsculas y minúsculas.
    return [item.lower() for item in TOKEN.findall(text)]


def write_matrix_csv(path: Path, filenames: list[str], columns: list[str], rows: list[dict[str, int | float]]) -> None:
    # Escribe una matriz en formato tabular: una fila por documento y una
    # columna por término. Los términos ausentes reciben valor cero.
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["archivo", *columns])
        for filename, row in zip(filenames, rows):
            writer.writerow([filename, *(row.get(term, 0) for term in columns)])


def main() -> None:
    # Permite elegir el corpus, el archivo JSON de intercambio y el modelo
    # para calcular embeddings semánticos.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--salida", type=Path, default=ROOT / "resultados/tfidf.json", help="JSON de intercambio; los CSV se guardan junto a este archivo")
    parser.add_argument("--modelo", default=DEFAULT_MODEL, help="Modelo sentence-transformers para calcular embeddings")
    args = parser.parse_args()

    # Ordena los documentos para mantener alineados y reproducibles los
    # resultados de cada archivo.
    files = sorted(args.corpus.glob("*.md"))
    if not files:
        raise SystemExit(f"No hay archivos .md en {args.corpus}")
    # Convierte cada documento en una lista de términos. La frecuencia
    # documental (df) cuenta en cuántos documentos aparece cada término.
    docs = [tokens(path.read_text(encoding="utf-8", errors="replace")) for path in files]
    df = Counter(term for doc in docs for term in set(doc))
    vocabulary = sorted(df)

    # Calcula por documento las frecuencias simples y los pesos TF-IDF.
    # TF mide la frecuencia local; IDF reduce el peso de términos comunes en
    # muchos documentos. Solo se guardan términos presentes en cada texto.
    matrix = []
    frequencies = []
    for doc in docs:
        counts = Counter(doc)
        frequencies.append({term: counts[term] for term in vocabulary if counts[term]})
        # Se aplica logaritmo a la frecuencia para atenuar repeticiones; el
        # factor IDF aumenta el peso de términos raros en el conjunto.
        matrix.append({term: round((1 + math.log(counts[term])) * (math.log((1 + len(docs)) / (1 + df[term])) + 1), 6) for term in vocabulary if counts[term]})

    # Asocia cada fila de frecuencias y TF-IDF con el nombre de su documento.
    result = {"archivos": [path.name for path in files], "frecuencias": frequencies, "tfidf": matrix}
    if args.modelo:
        # Los embeddings requieren sentence-transformers y representan cada
        # texto como un vector semántico de longitud fija.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit("Falta sentence-transformers. Instala: python3 -m pip install sentence-transformers") from exc
        model = SentenceTransformer(args.modelo)
        texts = [path.read_text(encoding="utf-8", errors="replace") for path in files]
        result["modelo_embeddings"] = args.modelo
        result["embeddings"] = model.encode(texts, normalize_embeddings=True).tolist()

    # Usa el nombre del JSON como base para las tablas CSV relacionadas.
    csv_stem = args.salida.stem
    frequencies_output = args.salida.with_name(f"{csv_stem}_frecuencias.csv")
    tfidf_output = args.salida.with_name(f"{csv_stem}_tfidf.csv")
    args.salida.parent.mkdir(parents=True, exist_ok=True)

    # Exporta las matrices en formato ancho: los encabezados son términos y
    # cada fila corresponde a un documento del corpus.
    filenames = [path.name for path in files]
    write_matrix_csv(frequencies_output, filenames, vocabulary, frequencies)
    write_matrix_csv(tfidf_output, filenames, vocabulary, matrix)

    # Si se calcularon embeddings, guárdalos también como una tabla numérica
    # que puede abrirse en una hoja de cálculo o herramienta de análisis.
    if args.modelo:
        embeddings_output = args.salida.with_name(f"{csv_stem}_embeddings.csv")
        embeddings = result["embeddings"]
        with embeddings_output.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["archivo", *(f"vector_{index}" for index in range(len(embeddings[0])))])
            for filename, vector in zip(filenames, embeddings):
                writer.writerow([filename, *vector])
        print(f"CSV de embeddings: {embeddings_output}")

    # Conserva el JSON para que 04_clustering.py pueda consumir embeddings
    # sin perder la estructura y el orden de documentos de este resultado.
    # El JSON conserva listas y metadatos estructurados; ensure_ascii=False
    # mantiene legibles los caracteres del español en el archivo resultante.
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    args.salida.write_text(payload + "\n", encoding="utf-8")
    print(f"CSV de frecuencias: {frequencies_output}")
    print(f"CSV de TF-IDF: {tfidf_output}")
    print(f"JSON para clustering: {args.salida}")


# Ejecuta el procesamiento solo al iniciar este archivo directamente.
if __name__ == "__main__":
    main()
