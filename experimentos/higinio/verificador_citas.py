"""Middleware determinista para reparar citas con evidencia recuperada."""

from __future__ import annotations

import re
from collections.abc import Mapping
from difflib import SequenceMatcher
from types import MappingProxyType
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from pydantic import JsonValue

from experimentos.higinio.traza import extraer_chunks_recuperados_busqueda
from taller_nlp import CorpusVariant, RespuestaFinanciera
from taller_nlp.citas import (
    cargar_fragmentos,
    cita_esta_respaldada,
    normalizar_texto,
)


_PALABRA = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_NUMERO = re.compile(r"\d[\d.,]*%?")


class VerificadorCitas(AgentMiddleware):
    """Repara citas solo a partir de chunks devueltos por ``search_filings``."""

    def __init__(self, corpus: CorpusVariant) -> None:
        if not isinstance(corpus, CorpusVariant):
            raise TypeError("corpus debe ser una instancia de CorpusVariant.")
        self._corpus = corpus
        self._chunks = cargar_fragmentos(corpus.ruta_chunks)

    @property
    def parametros(self) -> Mapping[str, JsonValue]:
        return MappingProxyType(
            {
                "solo_chunks_recuperados": True,
                "asocia_chunk_id_por_coincidencia_literal": True,
                "literaliza_linea_del_chunk": True,
            }
        )

    def after_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Repara de forma conservadora el par ``cita``/``chunk_id``."""
        del runtime
        respuesta = state.get("structured_response")
        if not isinstance(respuesta, RespuestaFinanciera):
            return None
        cita = respuesta.cita
        if cita is None or not cita.strip():
            return None

        recuperados = extraer_chunks_recuperados_busqueda(
            state.get("messages", ())
        )
        if not recuperados:
            return None

        chunk_actual = respuesta.chunk_id
        if self._par_valido(cita, chunk_actual, recuperados):
            return None

        coincidencias = tuple(
            chunk_id
            for chunk_id in recuperados
            if (fragmento := self._chunks.get(chunk_id)) is not None
            and cita_esta_respaldada(cita, fragmento.texto)
        )
        if len(coincidencias) == 1:
            return {
                "structured_response": respuesta.model_copy(
                    update={"chunk_id": coincidencias[0]}
                )
            }

        if chunk_actual not in recuperados:
            return None
        fragmento = self._chunks.get(chunk_actual)
        if fragmento is None:
            return None
        linea = self.seleccionar_linea_literal(cita, fragmento.texto)
        if linea is None:
            return None
        return {
            "structured_response": respuesta.model_copy(update={"cita": linea})
        }

    def _par_valido(
        self,
        cita: str,
        chunk_id: str | None,
        recuperados: tuple[str, ...],
    ) -> bool:
        if chunk_id is None or chunk_id not in recuperados:
            return False
        fragmento = self._chunks.get(chunk_id)
        return fragmento is not None and cita_esta_respaldada(
            cita, fragmento.texto
        )

    @classmethod
    def seleccionar_linea_literal(
        cls,
        cita: str,
        texto: str,
    ) -> str | None:
        """Elige una línea inequívoca usando solapamiento léxico y numérico."""
        lineas = tuple(linea.strip() for linea in texto.splitlines() if linea.strip())
        candidatas = []
        for posicion, linea in enumerate(lineas):
            puntuacion = cls.puntuar_linea(cita, linea)
            if puntuacion is not None:
                candidatas.append((puntuacion, -posicion, linea))
        if not candidatas:
            return None
        candidatas.sort(reverse=True)
        mejor = candidatas[0]
        if len(candidatas) > 1 and mejor[0] == candidatas[1][0]:
            return None
        return mejor[2]

    @staticmethod
    def puntuar_linea(
        cita: str,
        linea: str,
    ) -> tuple[int, int, float, float] | None:
        palabras_cita = set(_PALABRA.findall(normalizar_texto(cita)))
        palabras_linea = set(_PALABRA.findall(normalizar_texto(linea)))
        numeros_cita = VerificadorCitas._extraer_numeros(cita)
        numeros_linea = VerificadorCitas._extraer_numeros(linea)
        palabras_comunes = len(palabras_cita & palabras_linea)
        numeros_comunes = len(numeros_cita & numeros_linea)
        cobertura = palabras_comunes / max(
            1,
            min(len(palabras_cita), len(palabras_linea)),
        )
        similitud = SequenceMatcher(
            None,
            normalizar_texto(cita),
            normalizar_texto(linea),
        ).ratio()
        tiene_evidencia_numerica = numeros_comunes >= 1 and palabras_comunes >= 1
        tiene_evidencia_lexica = (
            not numeros_cita
            and palabras_comunes >= 4
            and cobertura >= 0.5
            and similitud >= 0.4
        )
        if not (tiene_evidencia_numerica or tiene_evidencia_lexica):
            return None
        return numeros_comunes, palabras_comunes, cobertura, similitud

    @classmethod
    def _seleccionar_linea_literal(
        cls,
        cita: str,
        texto: str,
    ) -> str | None:
        """Alias compatible para las pruebas y variantes anteriores."""
        return cls.seleccionar_linea_literal(cita, texto)

    @staticmethod
    def _extraer_numeros(texto: str) -> set[str]:
        numeros = set()
        for numero in _NUMERO.findall(texto):
            digitos = re.sub(r"\D", "", numero)
            if not digitos:
                continue
            if len(digitos) == 4 and 1900 <= int(digitos) <= 2100:
                continue
            numeros.add(digitos)
        return numeros
