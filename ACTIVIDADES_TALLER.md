**Actividades del taller:**

Esta guía propone un reto colaborativo por cada uno de los diez scripts del taller. Cada equipo comparte y revisa el trabajo y entrega un producto. Ejecuten los comandos desde la carpeta del proyecto.

**Antes de empezar**

- Usa `notas_de_prensa/seleccion_taller/corpus_practica` para practicar. No mezcles el corpus reservado para evaluación con los datos de trabajo.

- Cuando una salida muestre una coincidencia, entidad, cluster o respuesta, vuelve al documento original para comprobarla.

- En las actividades de Ollama, comprueba primero que el servicio local esté activo. Para experimentar con el indexado, limita el procesamiento a unos pocos documentos antes de usar el corpus completo.

- En equipo, distribuyan las tareas de ejecución, revisión de fuentes y registro de resultados; roten esos roles entre actividades.

**01. Búsqueda con expresiones regulares**

**Reto en equipo — Diseñen y evalúen una búsqueda.** Elijan una categoría del corpus y acuerden sus variantes, límites de palabra y posibles ambigüedades. Ejecuten `01_regex_contexto.py` con el patrón predeterminado y creen otro para esa categoría; prueben dos valores de `--ventana` y revisen al menos cinco coincidencias en sus notas originales. Clasifiquen una muestra como relevante o no relevante. **Entrega:** patrón acordado, ejemplos encontrados y una breve tabla con aciertos, falsos positivos y casos que el patrón podría omitir.

**02. Tokens, categorías gramaticales y entidades**

**Reto en equipo — Auditen las anotaciones.** Ejecuten `02_spacy_pos_ner.py` sobre una nota. Revisen diez tokens y cinco entidades del CSV, comprueben en el texto sus posiciones y etiquetas, y comparen las entidades que propone spaCy con las que identificaría el equipo. **Entrega:** una guía breve con ejemplos correctos, errores o desacuerdos y recomendaciones para revisar las anotaciones antes de reutilizarlas.

**03. Frecuencias, TF-IDF y embeddings**

**Reto en equipo — Comparen representaciones y decisiones de limpieza.** Ejecuten `03_representaciones.py`, elijan tres términos y comparen sus frecuencias y pesos TF-IDF en dos notas. Después, modifiquen en una copia una decisión de preprocesamiento —por ejemplo, excluir números o palabras muy comunes—, regeneren las matrices y comparen los términos más representativos. **Entrega:** una tabla de términos y valores, la decisión de limpieza, los cambios observados y qué información se pierde o se vuelve más visible.

**04. Agrupamiento y proyección**

**Reto en equipo — Pongan a prueba los grupos.** Ejecuten `04_clustering.py` con dos valores distintos de `--clusters` y comparen qué notas permanecen juntas o cambian de grupo. Elijan un grupo, propongan una descripción provisional y lean varias notas de ese grupo y de otros para buscar evidencia a favor y en contra. **Entrega:** dos observaciones respaldadas por las notas, una descripción provisional y sus límites. No interpreten los números de cluster como categorías; la proyección 2D es exploratoria.

**05. Visualización de clusters**

**Reto en equipo — Exploren y mejoren el mapa.** Abran el HTML producido por `05_visualizar_clusters.py`, elijan un cluster y revisen sus términos distintivos y varias notas fuente. Implementen una mejora, como un filtro por cluster o una búsqueda por título, y comprueben que ayuda a encontrar una nota sin ocultar su fuente. **Entrega:** descripción del grupo respaldada por términos y notas, demostración de la mejora y una limitación del mapa.

**Preparación de Ollama**

**Reto en equipo — Preparen Ollama para otra computadora.** Ejecuten `preparar_ollama_rag.py --solo-comprobar` e identifiquen qué modelos están disponibles y cuáles faltan sin iniciar descargas. Revisen las opciones de API y selección de modelos y diseñen una lista de comprobación que distinga entre verificar requisitos y descargar modelos. **Entrega:** resultados, pasos para dejar listo el entorno y problemas que deben resolverse antes de indexar.

**06. Indexación y fragmentación**

**Reto en equipo — Comparen la fragmentación.** Con Ollama disponible, indexen tres documentos usando dos combinaciones de tamaño de fragmento y solapamiento. Por ejemplo:

```bash
python 06_indexar_noticias_rag.py --limite 3 --tamano-chunk 500 --solapamiento 80 --salida resultados/indice_aula.json
```

Revisen los textos y metadatos de algunos fragmentos en ambas configuraciones. Comparen la cantidad de fragmentos, la continuidad del contexto y el tiempo de ejecución. **Entrega:** ejemplos de cortes útiles y de pérdida de contexto, configuración recomendada y el equilibrio que priorizaron.

**07. Recuperación y generación**

**Reto en equipo — Prueben recuperación y generación.** Diseñen una pregunta respondible, una ambigua y otra que probablemente no tenga respuesta en el corpus. Ejecuten cada una con `--solo-recuperar` y en modo normal; comparen los fragmentos y las respuestas con distintos valores de `-k`. **Entrega:** preguntas, evidencia útil, revisión de las citas y cualquier afirmación no verificable; evalúen si el sistema reconoce la falta de evidencia.

**08. Interfaz Streamlit**

**Reto en equipo — Sigan y mejoren la revisión del contexto.** Consulten el índice desde `08_app_streamlit.py`, expandan los fragmentos y los mensajes enviados al modelo, e identifiquen qué texto recibió para construir la respuesta. Implementen una mejora que facilite verificarla, como un botón para limpiar la consulta o destacar las fuentes. Prueben la interfaz con una pregunta respondible y otra sin evidencia suficiente. **Entrega:** recorrido de la pregunta a la respuesta, mejora demostrada, pruebas realizadas y observación sobre la similitud y las fuentes.

**09. Evaluación del mini RAG**

**Reto en equipo — Contrasten la evaluación automática y humana.** Ejecuten `09_evaluar_rag_ollama.py` con una pregunta y una respuesta de referencia mediante `--esperada`. Antes de consultar la puntuación del juez, revisen a ciegas la respuesta y sus fuentes; después comparen el juicio humano, la fidelidad al contexto, la relevancia y el solapamiento léxico. **Entrega:** tabla de resultados, desacuerdos y una recomendación sobre qué permite concluir cada señal y qué no.

**Reto integrador**

En equipos, formulen una pregunta de investigación acotada y recorran la secuencia: búsqueda o análisis lingüístico, representación, agrupamiento exploratorio, indexación, consulta y evaluación. Cada afirmación debe poder rastrearse hasta una nota fuente. **Presenten:** pregunta, decisiones tomadas, evidencia, limitaciones y una demostración reproducible del flujo. No mezclen el conjunto reservado para evaluación con el corpus usado para ajustar sus decisiones.

**Criterios de revisión sugeridos**

- **Reproducibilidad:** indican corpus, opciones y archivos generados.

- **Evidencia:** respaldan interpretaciones con documentos y fragmentos fuente.

- **Criterio crítico:** distinguen una salida del modelo de un hecho verificado.

- **Comunicación:** explican decisiones, desacuerdos y límites de manera clara.
