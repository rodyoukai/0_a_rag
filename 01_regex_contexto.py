#!/usr/bin/env python3
"""Ejercicio de la diapositiva 5: formas textuales con contexto revisable."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


# La raíz del proyecto se obtiene desde la ubicación de este script, así las
# rutas predeterminadas funcionan aunque se ejecute desde otro directorio.
ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "notas_de_prensa/seleccion_taller/corpus_practica"
DEFAULT_PATTERN = r"\b(?:fosa|fosas|narcofosa|narcofosas)\b"
DEFAULT_OUTPUT = ROOT / "resultados/contexto_regex.csv"


def context(text: str, start: int, end: int, window: int) -> str:
    # Busca el inicio de línea más cercano que quede dentro de la ventana
    # izquierda; nunca permite que el índice sea menor que cero.
    left = max(0, text.rfind("\n", 0, max(0, start - window)) + 1)

    # Busca el siguiente salto de línea después de la ventana derecha. Si no
    # existe, el texto termina en el final del documento.
    right_break = text.find("\n", min(len(text), end + window))
    right = len(text) if right_break == -1 else right_break

    # Une los espacios y saltos de línea para mostrar el contexto en una sola
    # fila de salida, sin alterar el texto original usado para buscar.
    return " ".join(text[left:right].split())


def main() -> None:
    # Define las opciones que puede recibir el script desde la terminal.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--patron", default=DEFAULT_PATTERN)
    parser.add_argument("--ventana", type=int, default=120)
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT, help="Ruta del CSV de resultados")
    args = parser.parse_args()

    # Compila la expresión regular una sola vez. IGNORECASE permite encontrar
    # las formas aunque aparezcan con mayúsculas o minúsculas distintas.
    pattern = re.compile(args.patron, re.IGNORECASE)

    # Reúne los archivos Markdown del corpus en orden alfabético para que la
    # salida sea reproducible entre ejecuciones.
    files = sorted(args.corpus.glob("*.md"))
    if not files:
        raise SystemExit(f"No hay archivos .md en {args.corpus}")

    # Crea la carpeta de destino si todavía no existe.
    args.salida.parent.mkdir(parents=True, exist_ok=True)

    # Abre el CSV con newline vacío para que csv.writer controle correctamente
    # los finales de línea y las comas o comillas dentro del contexto.
    with args.salida.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["archivo", "forma", "contexto"])

        for path in files:
            # Lee cada archivo como UTF-8; reemplaza caracteres inválidos para
            # que un problema de codificación no detenga todo el análisis.
            text = path.read_text(encoding="utf-8", errors="replace")

            # Busca las coincidencias y guarda una fila por cada una con el
            # nombre del archivo, la forma encontrada y su contexto.
            for match in pattern.finditer(text):
                writer.writerow([
                    path.name,
                    match.group(0),
                    context(text, match.start(), match.end(), args.ventana),
                ])

    # Informa dónde quedó el archivo para facilitar su localización.
    print(f"Escrito: {args.salida}")


# Ejecuta main solo cuando este archivo se inicia directamente, no al
# importarlo desde otro módulo.
if __name__ == "__main__":
    main()
