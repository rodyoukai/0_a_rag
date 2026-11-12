#!/usr/bin/env python3
"""Interfaz local para consultar el mini RAG y revisar la evidencia enviada."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parent
SCRIPTS = Path(__file__).resolve().parent
DEFAULT_INDEX = ROOT / "resultados/rag_noticias.json"
DEFAULT_API = "http://localhost:11434"
DEFAULT_CHAT_MODEL = "gemma4:e2b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# Los nombres de archivo empiezan con números, por eso se carga el script 07
# por ruta para reutilizar exactamente su recuperación y construcción de prompt.
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from rag_ollama_common import embed_texts, ollama_request  # noqa: E402

spec = importlib.util.spec_from_file_location("rag_consulta", SCRIPTS / "07_consultar_rag_ollama.py")
if spec is None or spec.loader is None:
    raise RuntimeError("No se pudo cargar scripts/07_consultar_rag_ollama.py")
rag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag)


@st.cache_resource(show_spinner=False)
def load_index(index_path: str) -> dict[str, Any]:
    return rag.load_index(Path(index_path))


def run_query(api: str, chat_model: str, embedding_model: str, question: str,
              index: dict[str, Any], result_count: int, max_per_document: int) -> dict[str, Any]:
    query_vector = embed_texts(api, embedding_model, [f"search_query: {question}"], batch_size=1)[0]
    results = rag.retrieve(index, query_vector, result_count, max_per_document)
    if not results:
        return {"results": [], "messages": [], "answer": "No se encontraron fragmentos relevantes."}

    messages = rag.build_messages(question, results)
    answer, word_count, elapsed = rag.answer_messages(api, chat_model, messages)
    return {
        "results": results,
        "messages": messages,
        "answer": answer,
        "input_word_count": word_count,
        "elapsed_seconds": elapsed,
    }


st.set_page_config(page_title="Mini RAG de notas de prensa", page_icon="📚", layout="wide")
st.title("Mini RAG de notas de prensa")
st.caption("Consulta el corpus y revisa qué fragmentos y qué instrucciones recibió el modelo.")

with st.sidebar:
    st.header("Configuración")
    api = st.text_input("API de Ollama", value=DEFAULT_API)
    chat_model = st.text_input("Modelo de chat", value=DEFAULT_CHAT_MODEL)
    embedding_model = st.text_input("Modelo de embeddings", value=DEFAULT_EMBEDDING_MODEL)
    index_path = st.text_input("Índice", value=str(DEFAULT_INDEX))
    result_count = st.slider("Fragmentos recuperados", min_value=1, max_value=10, value=5)
    max_per_document = st.slider("Máximo por documento", min_value=1, max_value=5, value=2)

try:
    index = load_index(index_path)
except (SystemExit, OSError, ValueError) as exc:
    st.error(str(exc))
    st.info("Crea el índice con `python3 scripts/06_indexar_noticias_rag.py` y comprueba que Ollama esté activo.")
    st.stop()

st.caption(
    f"Índice: {index.get('document_count')} documentos · "
    f"{index.get('chunk_count')} fragmentos · embeddings {index.get('embedding_model')}"
)

with st.form("consulta_rag"):
    question = st.text_area("Pregunta", placeholder="¿Qué ocurrió en las notas sobre Durango?", height=90)
    submitted = st.form_submit_button("Consultar", type="primary")

if submitted:
    if not question.strip():
        st.warning("Escribe una pregunta para iniciar la consulta.")
    else:
        try:
            with st.spinner("Recuperando evidencia y consultando el modelo..."):
                st.session_state["last_query"] = run_query(
                    api, chat_model, embedding_model, question.strip(), index,
                    result_count, max_per_document,
                )
                st.session_state["last_question"] = question.strip()
                st.session_state["last_models"] = (chat_model, embedding_model)
        except (SystemExit, OSError, ValueError) as exc:
            st.error(str(exc))

query = st.session_state.get("last_query")
if query:
    st.subheader("Respuesta")
    st.markdown(query["answer"])
    if "elapsed_seconds" in query:
        left, middle, right = st.columns(3)
        left.metric("Fragmentos usados", len(query["results"]))
        middle.metric("Palabras enviadas", query["input_word_count"])
        right.metric("Tiempo de generación", f"{query['elapsed_seconds']:.2f} s")

    st.subheader("Contexto recuperado")
    st.caption("Estos pasajes se insertaron en el mensaje del usuario enviado a Ollama.")
    for number, item in enumerate(query["results"], start=1):
        label = (
            f"{number}. {item['archivo']} · {item['titulo']} · "
            f"fragmento {item['fragmento']} · similitud {item['score']:.3f}"
        )
        with st.expander(label, expanded=number == 1):
            st.text(item["texto"])

    st.subheader("Mensajes exactos enviados al modelo")
    for message in query["messages"]:
        role_label = "Instrucciones del sistema" if message["role"] == "system" else "Pregunta y contexto"
        with st.expander(role_label, expanded=message["role"] == "user"):
            st.code(message["content"], language="text")

    with st.expander("Configuración de esta consulta"):
        st.write({
            "pregunta": st.session_state.get("last_question"),
            "modelo_chat": st.session_state.get("last_models", ("", ""))[0],
            "modelo_embeddings": st.session_state.get("last_models", ("", ""))[1],
            "índice": index_path,
        })
