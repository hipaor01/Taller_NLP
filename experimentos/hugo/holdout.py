"""Evaluación tolerante de un JSONL de preguntas (plan B para el día 24).

El día 24 se evalúan 10 preguntas ciegas y el profesor ha avisado de que al
menos dos no tienen respuesta en el corpus (un ejercicio que no está, un
concepto que la compañía no reporta). El cargador estricto del código común
(``CasoGolden.cargar_jsonl``) rechaza el fichero ENTERO en cuanto una pregunta
no cuadra con el corpus, así que ``python -m agente --evaluar`` no respondería
ninguna.

Este módulo vive en la carpeta de Hugo y NO modifica el código común: solo
usa sus clases. Lo invoca ``experimentos/hugo/evaluar_holdout.py``. Clasifica
cada línea y la evalúa por separado:

- ``golden``: pasa el validador y cuadra con el corpus. Se puntúa con el
  ``EvaluadorFinanciero`` de siempre, exactamente igual que en modo estricto.
- ``sin_respuesta``: la pregunta pide algo que no está en el corpus. Acierta
  si el agente responde ``fuente="ninguna"`` sin afirmar ninguna cifra, que es
  lo que el enunciado pide en vez de inventarse un número.
- ``no_evaluable``: no se puede saber qué se esperaba. Se ejecuta igualmente
  para que la respuesta quede registrada, pero no puntúa.

Ninguna regla mira preguntas concretas: se basan en el contrato del golden set
y en los mismos datos del corpus que usa el validador.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import ValidationError

from taller_nlp import CorpusVariant, EvaluadorFinanciero, RespuestaAgente
from taller_nlp.golden import CasoGolden
from taller_nlp.model_resilience import formatear_excepcion_modelo


TipoCaso = Literal["golden", "sin_respuesta", "no_evaluable"]

_MARCAS_SIN_RESPUESTA = re.compile(
    r"no\s+(est[aá]|figura|aparece|consta|existe|se\s+encuentra|hay|reporta|"
    r"report[oó]|est[aá]\s+disponible)|"
    r"sin\s+(dato|respuesta|informaci[oó]n)|"
    r"not\s+(in|available|reported|present)|n/?a\b|fuera\s+del\s+corpus",
    re.IGNORECASE,
)
"""Frases con las que un golden suele decir que la respuesta no existe."""

_PROBLEMAS_SIN_RESPUESTA = ("no está en el corpus", "no reporta")
"""Textos de ``CasoGolden.problemas_contra_corpus`` que indican ausencia."""


@dataclass(frozen=True)
class CasoHoldout:
    """Una línea del JSONL ya clasificada."""

    id: str
    pregunta: str | None
    tipo: TipoCaso
    caso: CasoGolden | None = None
    motivos: tuple[str, ...] = field(default_factory=tuple)
    familia: str | None = None
    ticker: str | None = None


def _texto_indica_sin_respuesta(texto: object) -> bool:
    return isinstance(texto, str) and bool(_MARCAS_SIN_RESPUESTA.search(texto))


def clasificar(
    datos: dict[str, Any],
    *,
    secciones: pd.DataFrame,
    xbrl: pd.DataFrame,
    posicion: int,
) -> CasoHoldout:
    """Decide cómo evaluar una línea del JSONL."""
    identificador = str(datos.get("id") or f"linea-{posicion}")
    pregunta = datos.get("pregunta")
    pregunta = pregunta.strip() if isinstance(pregunta, str) and pregunta.strip() else None
    familia = datos.get("familia") if isinstance(datos.get("familia"), str) else None
    ticker = datos.get("ticker") if isinstance(datos.get("ticker"), str) else None
    comunes = {"id": identificador, "pregunta": pregunta, "familia": familia, "ticker": ticker}

    if pregunta is None:
        return CasoHoldout(tipo="no_evaluable", motivos=("La línea no trae pregunta.",), **comunes)

    try:
        caso = CasoGolden.model_validate(datos)
    except ValidationError as exc:
        motivos = tuple(error["msg"] for error in exc.errors())
        sin_cifra = datos.get("familia") in {"numerica", "comparativa"} and (
            datos.get("cifra_esperada") is None
        )
        if sin_cifra or _texto_indica_sin_respuesta(datos.get("respuesta_esperada")):
            return CasoHoldout(tipo="sin_respuesta", motivos=motivos, **comunes)
        return CasoHoldout(tipo="no_evaluable", motivos=motivos, **comunes)

    problemas = tuple(caso.problemas_contra_corpus(secciones, xbrl))
    if not problemas:
        return CasoHoldout(tipo="golden", caso=caso, **comunes)
    if all(
        any(marca in problema for marca in _PROBLEMAS_SIN_RESPUESTA)
        for problema in problemas
    ):
        return CasoHoldout(tipo="sin_respuesta", caso=caso, motivos=problemas, **comunes)
    return CasoHoldout(tipo="no_evaluable", caso=caso, motivos=problemas, **comunes)


def cargar_holdout(ruta: str | Path, corpus: CorpusVariant) -> tuple[CasoHoldout, ...]:
    """Lee el JSONL línea a línea sin rechazar el conjunto por un caso malo."""
    ruta = Path(ruta)
    secciones = pd.read_json(corpus.ruta_secciones, lines=True)
    xbrl = pd.read_parquet(corpus.ruta_xbrl)
    casos = []
    with ruta.open(encoding="utf-8") as fichero:
        for posicion, linea in enumerate(fichero, start=1):
            if not linea.strip():
                continue
            try:
                datos = json.loads(linea)
            except json.JSONDecodeError as exc:
                casos.append(
                    CasoHoldout(
                        id=f"linea-{posicion}",
                        pregunta=None,
                        tipo="no_evaluable",
                        motivos=(f"JSON inválido: {exc.msg}",),
                    )
                )
                continue
            if not isinstance(datos, dict):
                casos.append(
                    CasoHoldout(
                        id=f"linea-{posicion}",
                        pregunta=None,
                        tipo="no_evaluable",
                        motivos=("La línea no es un objeto JSON.",),
                    )
                )
                continue
            casos.append(
                clasificar(datos, secciones=secciones, xbrl=xbrl, posicion=posicion)
            )
    return tuple(casos)


def respuesta_sin_dato_correcta(respuesta: RespuestaAgente) -> bool:
    """El agente reconoce que el dato no está y no afirma ninguna cifra."""
    return (
        respuesta.error is None
        and respuesta.fuente == "ninguna"
        and respuesta.cifra is None
    )


class _ProgresoHoldout:
    """Respuestas ya obtenidas, para no repetirlas ni pagarlas otra vez."""

    def __init__(self, ruta: Path | None) -> None:
        self._ruta = ruta
        self._respuestas: dict[str, RespuestaAgente] = {}
        if ruta is not None and ruta.is_file():
            for linea in ruta.read_text(encoding="utf-8").splitlines():
                if linea.strip():
                    fila = json.loads(linea)
                    self._respuestas[fila["id"]] = RespuestaAgente.model_validate(
                        fila["respuesta"]
                    )

    def obtener(self, identificador: str) -> RespuestaAgente | None:
        return self._respuestas.get(identificador)

    def guardar(self, identificador: str, respuesta: RespuestaAgente) -> None:
        self._respuestas[identificador] = respuesta
        if self._ruta is None or respuesta.error is not None:
            return
        self._ruta.parent.mkdir(parents=True, exist_ok=True)
        with self._ruta.open("a", encoding="utf-8") as fichero:
            fichero.write(
                json.dumps(
                    {"id": identificador, "respuesta": respuesta.model_dump(mode="json")},
                    ensure_ascii=False,
                )
                + "\n"
            )


def _responder(
    responder: Callable[[str], RespuestaAgente], pregunta: str
) -> RespuestaAgente:
    """Como el evaluador estricto, pero un fallo no detiene el resto."""
    try:
        respuesta = responder(pregunta)
    except Exception as exc:  # noqa: BLE001 - un fallo no debe parar el resto
        return RespuestaAgente(
            respuesta="",
            fuente="ninguna",
            latencia_ms=0,
            error=formatear_excepcion_modelo(exc),
        )
    if not isinstance(respuesta, RespuestaAgente):
        return RespuestaAgente(
            respuesta="",
            fuente="ninguna",
            latencia_ms=0,
            error="responder() no devolvió una RespuestaAgente.",
        )
    return respuesta


COLUMNAS_HOLDOUT = (
    "id",
    "familia",
    "ticker",
    "latencia_s",
    "coste_usd",
    "llamadas",
    "cita",
    "cifra",
    "trayectoria",
    "recall",
    "error",
    "tipo",
    "acierto",
    "fuente",
    "cifra_agente",
    "respuesta",
    "motivo",
)
"""Las columnas de la tabla estricta, en el mismo orden, y cinco más."""


def evaluar_holdout(
    casos: Iterable[CasoHoldout],
    *,
    evaluador: EvaluadorFinanciero,
    responder: Callable[[str], RespuestaAgente],
    ruta_progreso: str | Path | None = None,
) -> pd.DataFrame:
    """Responde y puntúa cada caso según su tipo; nunca aborta el conjunto."""
    progreso = _ProgresoHoldout(Path(ruta_progreso) if ruta_progreso else None)
    filas = []
    for caso in casos:
        respuesta = None
        if caso.pregunta is not None:
            respuesta = progreso.obtener(caso.id)
            if respuesta is None:
                respuesta = _responder(responder, caso.pregunta)
                progreso.guardar(caso.id, respuesta)

        fila: dict[str, Any] = {
            "id": caso.id,
            "familia": caso.familia,
            "ticker": caso.ticker,
            "latencia_s": respuesta.latencia_ms / 1_000 if respuesta else 0.0,
            "coste_usd": (respuesta.coste_usd or 0.0) if respuesta else 0.0,
            "llamadas": len(respuesta.llamadas) if respuesta else 0,
            "cita": None,
            "cifra": None,
            "trayectoria": None,
            "recall": None,
            "error": respuesta.error if respuesta else None,
            "tipo": caso.tipo,
            "acierto": None,
            "fuente": respuesta.fuente if respuesta else None,
            "cifra_agente": respuesta.cifra if respuesta else None,
            "respuesta": respuesta.respuesta if respuesta else None,
            "motivo": " | ".join(caso.motivos) or None,
        }
        if caso.tipo == "golden" and caso.caso is not None and respuesta is not None:
            resultado = evaluador._evaluar_caso(caso.caso, respuesta)
            fila.update(
                cita=(
                    None
                    if resultado.cita_existe is None
                    else bool(resultado.cita_existe and resultado.cita_respalda)
                ),
                cifra=resultado.cifra_correcta,
                trayectoria=resultado.trayectoria_correcta,
                recall=resultado.recall_at_k,
                acierto=resultado.acierto,
            )
        elif caso.tipo == "sin_respuesta" and respuesta is not None:
            fila["acierto"] = respuesta_sin_dato_correcta(respuesta)
        filas.append(fila)
    return pd.DataFrame(filas, columns=COLUMNAS_HOLDOUT)


def resumen_texto(tabla: pd.DataFrame) -> str:
    """Una línea legible para imprimir al terminar."""
    partes = []
    for tipo, nombre in (
        ("golden", "con respuesta"),
        ("sin_respuesta", "sin respuesta en el corpus"),
    ):
        subtabla = tabla[tabla["tipo"] == tipo]
        if len(subtabla):
            aciertos = int(subtabla["acierto"].fillna(False).astype(bool).sum())
            partes.append(f"{nombre}: {aciertos}/{len(subtabla)}")
    no_evaluables = int((tabla["tipo"] == "no_evaluable").sum())
    if no_evaluables:
        partes.append(f"no evaluables: {no_evaluables}")
    return " · ".join(partes)


__all__ = [
    "COLUMNAS_HOLDOUT",
    "CasoHoldout",
    "cargar_holdout",
    "clasificar",
    "evaluar_holdout",
    "respuesta_sin_dato_correcta",
    "resumen_texto",
]
