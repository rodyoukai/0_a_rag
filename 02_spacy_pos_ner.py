#!/usr/bin/env python3
"""Ejercicio de las diapositivas 7 y 8: POS y entidades nombradas."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


# Calcula rutas predeterminadas desde la raíz del proyecto, no desde el
# directorio actual desde el que se ejecute el script.
ROOT = Path(__file__).resolve().parent
DEFAULT_FILE = ROOT / "notas_de_prensa/seleccion_taller/corpus_practica/IBNP2009010501.md"


def main() -> None:
    # Define el archivo que se analizará, el modelo lingüístico y si se permite
    # descargar automáticamente el modelo cuando no esté instalado.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archivo", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--modelo", default="es_core_news_sm")
    parser.add_argument("--no-descargar-modelo", action="store_true", help="Falla si el modelo no está instalado")
    parser.add_argument("--salida-tokens", type=Path, help="CSV de tokens; por defecto se guarda en resultados/")
    parser.add_argument("--salida-entidades", type=Path, help="CSV de entidades; por defecto se guarda en resultados/")
    args = parser.parse_args()

    # Importa spaCy dentro de main para mostrar una instrucción útil si falta
    # la dependencia, sin impedir importar este módulo desde otro programa.
    try:
        import spacy
    except ImportError as exc:
        raise SystemExit("Falta spaCy. Instala: python3 -m pip install spacy") from exc

    # Carga el modelo de español instalado. Si no está disponible, intenta
    # descargarlo salvo que el usuario haya pedido explícitamente no hacerlo.
    try:
        nlp = spacy.load(args.modelo)
    except OSError as exc:
        if args.no_descargar_modelo:
            raise SystemExit(f"No se encuentra el modelo {args.modelo}. Ejecuta: python3 -m spacy download {args.modelo}") from exc
        print(f"No se encuentra {args.modelo}; se intentará descargarlo...", file=sys.stderr)
        try:
            from spacy.cli.download import download
            download(args.modelo)
            nlp = spacy.load(args.modelo)
        except Exception as download_error:
            raise SystemExit(f"No fue posible descargar {args.modelo}: {download_error}") from download_error

    # Lee el texto y lo procesa con el pipeline de spaCy, que separa tokens y
    # calcula sus atributos lingüísticos y las entidades nombradas.
    text = args.archivo.read_text(encoding="utf-8", errors="replace")
    doc = nlp(text)

    # Usa rutas personalizadas si se proporcionaron; de lo contrario, crea dos
    # nombres distintos en resultados/ a partir del nombre del documento.
    token_output = args.salida_tokens or ROOT / "resultados" / f"{args.archivo.stem}_tokens.csv"
    entity_output = args.salida_entidades or ROOT / "resultados" / f"{args.archivo.stem}_entidades.csv"

    # Crea las carpetas necesarias para ambos archivos de salida.
    token_output.parent.mkdir(parents=True, exist_ok=True)
    entity_output.parent.mkdir(parents=True, exist_ok=True)

    # Guarda cada token en una fila. POS es la categoría gramatical y
    # dependencia contiene relaciones sintácticas como nsubj.
    with token_output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["archivo", "texto", "pos", "dependencia"])
        for token in doc:
            if not token.is_space:
                writer.writerow([args.archivo.name, token.text, token.pos_, token.dep_])

    # Guarda las entidades nombradas por separado, con sus offsets de inicio
    # y fin para poder localizarlas dentro del texto original.
    with entity_output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["archivo", "texto", "tipo", "inicio", "fin"])
        for entity in doc.ents:
            writer.writerow([args.archivo.name, entity.text, entity.label_, entity.start_char, entity.end_char])

    # Informa la ruta de cada tabla generada.
    print(f"CSV de tokens: {token_output}")
    print(f"CSV de entidades: {entity_output}")


# Ejecuta el análisis solo al iniciar este archivo directamente.
if __name__ == "__main__":
    main()
