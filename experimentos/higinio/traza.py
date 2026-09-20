"""Utilidades compartidas para inspeccionar trazas de herramientas."""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from taller_nlp.retrieval import extraer_chunk_ids_formateados


def extraer_chunks_recuperados_busqueda(
    mensajes: object,
) -> tuple[str, ...]:
    """Extrae los chunks devueltos por llamadas exitosas a ``search_filings``."""
    if not isinstance(mensajes, (list, tuple)):
        return ()

    ids_llamadas = {
        llamada.get("id")
        for mensaje in mensajes
        if isinstance(mensaje, AIMessage)
        for llamada in mensaje.tool_calls
        if llamada.get("name") == "search_filings" and llamada.get("id")
    }
    encontrados: list[str] = []
    for mensaje in mensajes:
        if not isinstance(mensaje, ToolMessage) or mensaje.status == "error":
            continue
        es_busqueda = (
            mensaje.name == "search_filings"
            or mensaje.tool_call_id in ids_llamadas
        )
        if not es_busqueda:
            continue
        encontrados.extend(extraer_chunk_ids_formateados(mensaje.text))
    return tuple(dict.fromkeys(encontrados))
