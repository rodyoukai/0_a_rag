#!/usr/bin/env python3
"""Comprueba Ollama y descarga los modelos para el mini RAG."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

from rag_ollama_common import DEFAULT_API, DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL, available_models


def pull_model(api: str, model: str) -> None:
    # Ollama envía eventos JSON línea por línea; stream=True permite informar
    # del avance mientras descarga un modelo que puede ocupar varios GB.
    url = f"{api.rstrip('/')}/api/pull"
    payload = json.dumps({"model": model, "stream": True}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    print(f"Descargando {model}...")
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            for line in response:
                if not line.strip():
                    continue
                event = json.loads(line.decode("utf-8"))
                status = event.get("status", "descargando")
                total = event.get("total")
                completed = event.get("completed")
                if total and completed is not None:
                    # Cuando Ollama conoce el tamaño total, calcula el avance;
                    # si no, muestra el estado textual del evento.
                    percent = completed / total * 100
                    print(f"\r  {status}: {percent:5.1f}%", end="", flush=True)
                else:
                    print(f"  {status}")
                if event.get("error"):
                    raise SystemExit(f"Ollama no pudo descargar {model}: {event['error']}")
        print(f"\nModelo listo: {model}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Error HTTP {exc.code} al descargar {model}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"No se pudo completar la descarga de {model}: {exc}") from exc


def main() -> None:
    # Permite cambiar la dirección del servicio y los modelos sin editar código.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=DEFAULT_API, help="Dirección del servicio Ollama")
    parser.add_argument("--modelo-llm", default=DEFAULT_CHAT_MODEL, help="Modelo de chat para generar respuestas")
    parser.add_argument("--modelo-embeddings", default=DEFAULT_EMBEDDING_MODEL, help="Modelo para representar los textos como vectores")
    parser.add_argument("--solo-comprobar", action="store_true", help="No descargar; solo informar qué modelos faltan")
    args = parser.parse_args()

    # Contrasta los modelos pedidos con el inventario de Ollama. El sufijo
    # :latest se considera equivalente al nombre sin etiqueta.
    installed = available_models(args.api)
    requested = [args.modelo_llm, args.modelo_embeddings]
    missing = [model for model in requested if not any(name == model or name == f"{model}:latest" for name in installed)]

    # Este modo es útil en clase para diagnosticar requisitos sin descargar.
    if args.solo_comprobar:
        for model in requested:
            status = "disponible" if model not in missing else "falta descargar"
            print(f"{model}: {status}")
        if missing:
            raise SystemExit(1)
        return

    # Descarga solo lo que hace falta y vuelve a consultar el servidor para
    # comprobar que los modelos efectivamente quedaron disponibles.
    for model in missing:
        pull_model(args.api, model)

    installed = available_models(args.api)
    still_missing = [model for model in requested if not any(name == model or name == f"{model}:latest" for name in installed)]
    if still_missing:
        raise SystemExit(f"No aparecen en Ollama los modelos solicitados: {', '.join(still_missing)}")
    print("Preparación completa. Ollama tiene disponibles el modelo de chat y el de embeddings.")


if __name__ == "__main__":
    main()