# Scripts del taller: análisis de prensa y mini RAG

Los scripts siguen la secuencia de la presentación elegida:

1. `01_regex_contexto.py`: búsqueda de formas textuales y contexto verificable.
2. `02_spacy_pos_ner.py`: tokens, POS y entidades nombradas sobre una nota.
3. `03_representaciones.py`: bolsa de palabras y TF-IDF para comparar documentos.
4. `04_clustering.py`: agrupamiento y proyección 2D. La proyección sirve para explorar, no para validar categorías.
5. `05_visualizar_clusters.py`: mapa interactivo y términos TF-IDF distintivos por grupo.
Preparación de Ollama (`preparar_ollama_rag.py`): verifica Ollama y descarga los modelos para el mini RAG.
6. `06_indexar_noticias_rag.py`: divide las noticias en fragmentos y crea el índice de embeddings.
7. `07_consultar_rag_ollama.py`: recupera fragmentos y responde con gemma4:e2b mostrando las fuentes.
8. `08_app_streamlit.py`: interfaz para consultar el mini RAG y ver la respuesta, los fragmentos recuperados y los mensajes enviados a Ollama.
9. `09_evaluar_rag_ollama.py`: evalúa una respuesta generada según fidelidad al contexto y relevancia; puede añadir solapamiento léxico con una referencia.

Las actividades individuales y en equipo asociadas a cada script están en
[ACTIVIDADES_TALLER.md](./ACTIVIDADES_TALLER.md).

Todos usan por defecto `notas_de_prensa/seleccion_taller/corpus_practica`. El corpus de `corpus_evaluacion` queda reservado para una comprobación posterior y no debe mezclarse con la práctica.

## Uso rápido

Desde la raíz del repositorio:

```bash
python3 scripts/01_regex_contexto.py --salida resultados/contexto_regex.csv
python3 scripts/02_spacy_pos_ner.py --archivo notas_de_prensa/seleccion_taller/corpus_practica/IBNP2009010501.md
python3 scripts/03_representaciones.py --salida resultados/tfidf.json
python3 scripts/04_clustering.py --entrada resultados/tfidf.json --salida resultados/clusters.json
```

El primer y el tercer script no requieren paquetes externos. Para POS y NER instala spaCy:

El script `01_regex_contexto.py` guarda por defecto sus coincidencias en `resultados/contexto_regex.csv`; usa `--salida` para elegir otra ruta.

```bash
python3 -m pip install spacy
```

`02_spacy_pos_ner.py` comprueba si existe `es_core_news_sm` y lo descarga automáticamente cuando falta. Para impedir cualquier descarga automática usa `--no-descargar-modelo`.

El script guarda por defecto dos CSV en `resultados/`: uno con tokens, POS y dependencias, y otro con entidades y sus posiciones. Las rutas pueden cambiarse con `--salida-tokens` y `--salida-entidades`.

Para embeddings y clustering/proyección:

```bash
python3 -m pip install -r requirements.txt
```

`03_representaciones.py` guarda por defecto matrices de frecuencias y TF-IDF en `resultados/tfidf_frecuencias.csv` y `resultados/tfidf_tfidf.csv`. También conserva `resultados/tfidf.json` para que el script de clustering reciba los embeddings.

El script de representaciones puede calcular embeddings si se le pasa `--modelo`; `sentence-transformers` descarga ese modelo automáticamente la primera vez y puede requerir varios GB de disco. En ese caso también se guarda una tabla `*_embeddings.csv`:

```bash
python3 scripts/03_representaciones.py --modelo BAAI/bge-m3 --salida resultados/bge-m3.json
```

La salida conserva siempre el nombre del archivo fuente. Antes de interpretar una coincidencia, entidad o cluster hay que volver a la nota original.

## Mini RAG local con Ollama

Estos ejercicios adaptan la secuencia de embeddings y recuperación aumentada de las carpetas `01_embeddings` y `02_retrieval_augmented_generation` del [taller genai4lib-workshop](https://github.com/nulib/genai4lib-workshop). Usan la API HTTP de Ollama y la biblioteca estándar de Python; no requieren LangChain, una base de datos vectorial ni paquetes Python adicionales.

El corpus predeterminado son los 300 archivos `IBNP*.md` que están directamente en `notas_de_prensa/`. No se recorren subcarpetas, por lo que `IDEA.md`, las notas seleccionadas para práctica y el conjunto reservado para evaluación no se mezclan en el índice.

### Requisitos y preparación

1. Instala o actualiza Ollama desde [ollama.com/download](https://ollama.com/download) e inicia el servicio local.
2. Desde la raíz del proyecto, prepara los modelos:

```bash
python3 scripts/preparar_ollama_rag.py
```

El script descarga `gemma4:e2b` para generar respuestas y `nomic-embed-text` para embeddings. Para comprobarlos sin descargar:

```bash
python3 scripts/preparar_ollama_rag.py --solo-comprobar
```

Si Ollama responde que el manifiesto requiere una versión más reciente, actualiza Ollama y vuelve a ejecutar el preparador. La descarga del programa Ollama se hace desde su instalador oficial; el script automatiza las descargas de modelos mediante la API local.

### Crear el índice

```bash
python3 scripts/06_indexar_noticias_rag.py
```

El proceso genera embeddings en lotes, divide el texto en fragmentos de 1200 caracteres con 180 de solapamiento y guarda el índice como JSON indentado y legible en `resultados/rag_noticias.json`. Al incluir los vectores numéricos de los embeddings, el archivo puede ser grande. Si corpus y parámetros no cambiaron, reutiliza el índice; `--forzar` lo reconstruye. Para probar primero con cinco documentos:

```bash
python3 scripts/06_indexar_noticias_rag.py --limite 5 --salida resultados/rag_prueba.json
```

### Recuperar y generar

Una pregunta de una sola ejecución:

```bash
python3 scripts/07_consultar_rag_ollama.py "¿Qué ocurrió en las notas sobre Durango?"
```

Modo interactivo:

```bash
python3 scripts/07_consultar_rag_ollama.py
```

Para observar únicamente la recuperación semántica, sin llamar al LLM:

```bash
python3 scripts/07_consultar_rag_ollama.py --solo-recuperar "¿Qué ocurrió en las notas sobre Durango?"
```

El buscador ordena fragmentos por similitud coseno. La generación recibe solo los fragmentos recuperados, debe abstenerse si no hay evidencia y cita los archivos fuente. El modo `--solo-recuperar` permite separar y enseñar la etapa de búsqueda de la etapa de generación.

### Interfaz Streamlit

Instala las dependencias de la interfaz y ejecútala desde la raíz del proyecto:

```bash
python3 -m pip install -r requirements.txt
streamlit run scripts/08_app_streamlit.py
```

La interfaz reutiliza la recuperación y el prompt del script 07. Muestra la respuesta generada, cada fragmento recuperado con archivo y similitud, y los mensajes exactos (instrucciones del sistema, contexto y pregunta) enviados al modelo de chat. El servicio local de Ollama y el índice deben estar disponibles.

### Evaluar una respuesta

El evaluador ejecuta una consulta completa sobre el mismo índice, genera una respuesta y pide a un modelo local que califique su fidelidad al contexto y su relevancia. La salida JSON incluye respuesta, puntuaciones, justificación, fragmentos, fuentes, similitudes y tiempos.

Por defecto, el evaluador **guarda automáticamente la evaluación en un archivo JSON con timestamp** en el directorio `resultados/`:

```bash
python3 scripts/09_evaluar_rag_ollama.py \
  "¿Qué ocurrió en las notas sobre Durango?" \
  --esperada "Las notas describen hallazgos de cuerpos en Durango."
```

El archivo se guarda como `evaluacion_rag_YYYYMMDD_HHMMSS.json` e incluye un campo `timestamp` con la fecha y hora de la evaluación.

**Opciones:**

- Si omites la pregunta, usa por defecto `¿Qué ocurrió en las notas sobre Durango?`
- `--esperada` es opcional. Si se proporciona, el script calcula el solapamiento de palabras entre la respuesta y la referencia. Esa cifra es una señal léxica sencilla: no mide equivalencia semántica ni reemplaza la revisión de los fragmentos y del juicio del modelo.
- El juez predeterminado es `gemma4:e2b`; se cambia con `--modelo-juez`
- Para especificar un directorio diferente para guardar el JSON: `--output-dir ./mis_resultados`
- Para **no guardar** el archivo JSON (solo stdout): `--no-guardar-json`
