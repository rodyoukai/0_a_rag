#!/usr/bin/env python3
"""Utilidades compartidas para los ejercicios RAG locales con Ollama."""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
# Los valores predeterminados centralizados evitan que los scripts del flujo
# usen direcciones, corpus o nombres de modelo distintos entre sí.
DEFAULT_CORPUS = ROOT / "notas_de_prensa"
DEFAULT_API = "http://localhost:11434"

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_CHAT_MODEL = "gemma4:e2b"

def ollama_request(api: str, endpoint: str, payload: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    # Encapsula la comunicación HTTP para compartir serialización, tiempos de
    # espera y mensajes de error en todos los ejercicios RAG.
    url = f"{api.rstrip('/')}/{endpoint.lstrip('/') }"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Ollama respondió HTTP {exc.code} en {endpoint}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(
            f"No se pudo conectar con Ollama en {api}. Comprueba que Ollama esté instalado y activo. Detalle: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Ollama devolvió una respuesta JSON inválida en {endpoint}.") from exc


def available_models(api: str) -> set[str]:
    # Consulta los nombres instalados para que el preparador pueda decidir qué
    # descargas son necesarias.
    payload = ollama_request(api, "/api/tags", timeout=10)
    return {str(model.get("name", "")) for model in payload.get("models", [])}


def model_is_available(model: str, installed: set[str]) -> bool:
    # Ollama suele mostrar una etiqueta explícita; acepta también :latest.
    return model in installed or f"{model}:latest" in installed


def news_files(corpus: Path = DEFAULT_CORPUS) -> list[Path]:
    # El patrón solo recorre la carpeta indicada, no incorpora subcarpetas ni
    # mezcla por accidente conjuntos reservados de evaluación.
    files = sorted(path for path in corpus.glob("IBNP*.md") if path.is_file())
    if not files:
        raise SystemExit(f"No se encontraron archivos IBNP*.md en {corpus}")
    return files


def document_title(text: str, path: Path) -> str:
    # Usa el primer encabezado Markdown como título; el nombre del archivo es
    # una alternativa estable cuando el documento no tiene encabezado.
    for line in text.splitlines():
        heading = line.strip()
        if heading.startswith("# "):
            return heading[2:].strip()
    return path.stem


def split_text(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
    # Divide documentos largos para que cada pasaje quepa en el modelo, pero
    # conserva un solapamiento para no perder contexto entre límites.
    if chunk_size < 100:
        raise SystemExit("--tamano-chunk debe ser de al menos 100 caracteres.")
    if overlap < 0 or overlap >= chunk_size:
        raise SystemExit("--solapamiento debe ser mayor o igual a 0 y menor que --tamano-chunk.")

    chunks = []
    start = 0
    while start < len(text):
        # Intenta cortar en un párrafo, oración o espacio cercano al límite,
        # antes de recurrir al corte exacto por cantidad de caracteres.
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            boundary = max(window.rfind("\n\n"), window.rfind(". "), window.rfind(" "))
            if boundary >= int(chunk_size * 0.6):
                end = start + boundary + (2 if window[boundary:boundary + 2] == "\n\n" else 1)

        # Los offsets corresponden al texto fuente sin espacios periféricos;
        # son útiles para localizar la evidencia original.
        chunk = text[start:end].strip()
        if chunk:
            left = start + len(text[start:end]) - len(text[start:end].lstrip())
            right = start + len(text[start:end].rstrip())
            chunks.append((left, right, chunk))
        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap)
        while next_start < end and not text[next_start].isspace():
            next_start += 1
        start = next_start if next_start < end else end
    return chunks


def embed_texts(api: str, model: str, texts: list[str], batch_size: int = 16) -> list[list[float]]:
    # Procesa en lotes para no enviar demasiados textos en una única petición
    # y valida la respuesta antes de usarla como dato numérico.
    if batch_size < 1:
        raise SystemExit("--lote debe ser mayor que cero.")
    vectors = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset:offset + batch_size]
        batch_number = offset // batch_size + 1
        batch_count = (len(texts) + batch_size - 1) // batch_size
        print(f"Embeddings: lote {batch_number}/{batch_count} ({min(offset + len(batch), len(texts))}/{len(texts)} fragmentos)", flush=True)
        payload = ollama_request(
            api,
            "/api/embed",
            {"model": model, "input": batch, "truncate": False},
            timeout=300,
        )
        batch_vectors = payload.get("embeddings")
        if not isinstance(batch_vectors, list) or len(batch_vectors) != len(batch):
            raise SystemExit("La respuesta de Ollama no contiene un embedding por cada fragmento.")
        # Convierte valores a float y descarta vectores vacíos o no finitos,
        # que después invalidarían la similitud matemática.
        for vector in batch_vectors:
            if not isinstance(vector, list) or not vector:
                raise SystemExit("Ollama devolvió un vector vacío.")
            values = [float(value) for value in vector]
            if not all(math.isfinite(value) for value in values):
                raise SystemExit("Ollama devolvió valores no finitos en un embedding.")
            vectors.append(values)
    return vectors


def cosine_similarity(left: list[float], right: list[float]) -> float:
    # El coseno compara la orientación de dos vectores: valores altos indican
    # mayor similitud, sin depender directamente de la magnitud de cada uno.
    if len(left) != len(right):
        raise ValueError("Los vectores deben tener la misma dimensión.")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def safe_excerpt(text: str, query: str, width: int = 240) -> str:
    # Compacta el fragmento y trata de mostrar una ventana alrededor del primer
    # término de búsqueda encontrado; si no coincide ninguno, muestra el inicio.
    normalized = re.sub(r"\s+", " ", text).strip()
    query_terms = [term.casefold() for term in re.findall(r"[\wáéíóúüñ]+", query, re.IGNORECASE) if len(term) > 2]
    folded = normalized.casefold()
    positions = [folded.find(term) for term in query_terms]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - width // 3)
    end = min(len(normalized), start + width)
    excerpt = normalized[start:end]
    if start:
        excerpt = "..." + excerpt
    if end < len(normalized):
        excerpt += "..."
    return excerpt