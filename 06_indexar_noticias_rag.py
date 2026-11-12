#!/usr/bin/env python3
"""Crea un índice semántico local con los 300 documentos de prensa IBNP."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from rag_ollama_common import (
    DEFAULT_API,
    DEFAULT_CORPUS,
    DEFAULT_EMBEDDING_MODEL,
    document_title,
    embed_texts,
    news_files,
    ollama_request,
    split_text,
)


INDEX_VERSION = 1


def load_index(path: Path) -> dict[str, Any] | None:
    # Un índice inexistente o dañado no debe impedir reconstruirlo desde las
    # fuentes; por eso esta función devuelve None ante fallos de lectura.
    if not path.is_file():
        return None
    try:
        opener = gzip.open if path.suffix == ".gz" else Path.open
        with opener(path, "rt", encoding="utf-8") as input_file:
            data = json.load(input_file)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def corpus_fingerprint(files: list[Path], model: str, chunk_size: int, overlap: int) -> str:
    # La huella incluye contenido y parámetros de procesamiento. Si cambia
    # cualquiera de ellos, el índice anterior ya no representa la configuración.
    digest = hashlib.sha256()
    digest.update(f"{INDEX_VERSION}\0{model}\0{chunk_size}\0{overlap}\n".encode())
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    # Los argumentos permiten experimentar con tamaño/solapamiento y probar
    # subconjuntos antes de indexar todas las noticias.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="Carpeta con notas IBNP*.md; no busca en subcarpetas")
    parser.add_argument("--salida", type=Path, default=Path(__file__).resolve().parent / "resultados/rag_noticias.json")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--modelo-embeddings", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--tamano-chunk", type=int, default=1200, help="Tamaño aproximado de cada fragmento en caracteres")
    parser.add_argument("--solapamiento", type=int, default=180)
    parser.add_argument("--lote", type=int, default=16, help="Cantidad de fragmentos por llamada a Ollama")
    parser.add_argument("--limite", type=int, help="Procesa solo los primeros N documentos, útil para una prueba")
    parser.add_argument("--forzar", action="store_true", help="Recalcula aunque ya exista un índice vigente")
    args = parser.parse_args()

    if args.limite is not None and args.limite < 1:
        raise SystemExit("--limite debe ser mayor que cero.")
    if args.lote < 1:
        raise SystemExit("--lote debe ser mayor que cero.")

    # Comprueba primero que el modelo de embeddings exista, para fallar antes
    # de invertir tiempo en leer y fragmentar el corpus.
    installed = ollama_request(args.api, "/api/tags", timeout=10).get("models", [])
    installed_names = {str(model.get("name", "")) for model in installed}
    if not any(name == args.modelo_embeddings or name == f"{args.modelo_embeddings}:latest" for name in installed_names):
        raise SystemExit(f"No está descargado {args.modelo_embeddings}. Ejecuta scripts/preparar_ollama_rag.py.")

    # Selecciona las notas de entrada y crea una huella para reutilizar un
    # índice previo cuando no hubo cambios en textos ni parámetros.
    files = news_files(args.corpus)
    if args.limite:
        files = files[:args.limite]
    fingerprint = corpus_fingerprint(files, args.modelo_embeddings, args.tamano_chunk, args.solapamiento)
    existing = load_index(args.salida)
    if not args.forzar and existing and existing.get("fingerprint") == fingerprint:
        print(f"Índice vigente: {args.salida} ({existing.get('document_count')} documentos, {existing.get('chunk_count')} fragmentos).")
        print("No cambió el corpus ni la configuración; usa --forzar para reconstruirlo.")
        return

    # Cada fragmento conserva su archivo, título y offsets para poder regresar
    # a la fuente; el prefijo search_document sigue el formato del modelo.
    records = []
    embed_inputs = []
    for document_number, path in enumerate(files, start=1):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            print(f"Aviso: se omite documento vacío: {path.name}")
            continue
        title = document_title(text, path)
        for chunk_number, (start, end, chunk) in enumerate(split_text(text, args.tamano_chunk, args.solapamiento), start=1):
            records.append({
                "id": f"{path.stem}#p{chunk_number:03d}",
                "archivo": path.name,
                "titulo": title,
                "fragmento": chunk_number,
                "inicio": start,
                "fin": end,
                "texto": chunk,
            })
            embed_inputs.append(f"search_document: {chunk}")
        if document_number % 50 == 0 or document_number == len(files):
            print(f"Notas preparadas: {document_number}/{len(files)} · fragmentos: {len(records)}")

    if not records:
        raise SystemExit("No se encontraron fragmentos para indexar.")

    # Convierte el texto de cada fragmento a un vector numérico y verifica que
    # todos compartan la dimensión esperada antes de guardarlos.
    print(f"Solicitando embeddings para {len(records)} fragmentos con {args.modelo_embeddings}...")
    vectors = embed_texts(args.api, args.modelo_embeddings, embed_inputs, args.lote)
    dimensions = len(vectors[0])
    if any(len(vector) != dimensions for vector in vectors):
        raise SystemExit("Ollama devolvió embeddings de dimensiones distintas.")
    for record, vector in zip(records, vectors, strict=True):
        record["embedding"] = vector

    # El índice incluye configuración y estadísticas, además de los registros,
    # para explicar cómo se produjo y facilitar futuras comprobaciones.
    index = {
        "version": INDEX_VERSION,
        "corpus": str(args.corpus.resolve()),
        "embedding_model": args.modelo_embeddings,
        "chunk_size": args.tamano_chunk,
        "overlap": args.solapamiento,
        "dimensions": dimensions,
        "document_count": len(files),
        "chunk_count": len(records),
        "fingerprint": fingerprint,
        "records": records,
    }
    # Escribe primero en un temporal y luego lo renombra: así se evita dejar
    # un archivo final parcialmente escrito si el proceso se interrumpe.
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.salida.with_name(args.salida.name + ".tmp")
    try:
        opener = gzip.open if args.salida.suffix == ".gz" else Path.open
        with opener(temporary, "wt", encoding="utf-8") as output_file:
            json.dump(index, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        temporary.replace(args.salida)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"No se pudo escribir el índice en {args.salida}: {exc}") from exc
    print(f"Índice escrito: {args.salida}")
    print(f"{len(files)} documentos · {len(records)} fragmentos · dimensión {dimensions}")


if __name__ == "__main__":
    main()