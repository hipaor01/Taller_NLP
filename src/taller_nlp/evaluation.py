"""Evaluación reproducible de respuestas y trayectorias del agente."""

from __future__ import annotations

import json
import hashlib
import math
import re
from collections.abc import Callable
from pathlib import Path

from .contracts import (
    InformeEvaluacion,
    LlamadaHerramienta,
    RespuestaAgente,
    ResultadoPregunta,
)
from .corpus import CorpusVariant
from .golden import CasoGolden
from .evaluation_progress import (
    AlmacenProgresoEvaluacion,
    ContextoProgreso,
)
from .hashing import calcular_sha256
from .model_resilience import (
    ErrorProveedorAgotadoError,
    RateLimitAgotadoError,
    es_error_rate_limit,
    es_error_transitorio_modelo,
    formatear_excepcion_modelo,
    texto_indica_error_transitorio,
    texto_indica_rate_limit,
)


JuezCitas = Callable[[str, tuple[str, ...]], bool]


class EvaluadorFinanciero:
    """Evalúa respuestas, evidencia, cifras y trayectorias contra un JSONL."""

    def __init__(
        self,
        corpus: CorpusVariant,
        juez_citas: JuezCitas,
        *,
        k_retrieval: int = 5,
        tolerancia_absoluta: float = 1.0,
        tolerancia_relativa: float = 1e-6,
        numero_esperado: int | None = None,
        minimo_comparativas: int = 0,
        ruta_progreso: str | Path | None = None,
    ) -> None:
        if k_retrieval <= 0:
            raise ValueError("k_retrieval debe ser mayor que cero.")
        if tolerancia_absoluta < 0 or tolerancia_relativa < 0:
            raise ValueError("Las tolerancias no pueden ser negativas.")
        if numero_esperado is not None and numero_esperado <= 0:
            raise ValueError("numero_esperado debe ser mayor que cero.")
        if minimo_comparativas < 0:
            raise ValueError("minimo_comparativas no puede ser negativo.")

        self._corpus = corpus
        self._juez_citas = juez_citas
        self._k = k_retrieval
        self._tolerancia_absoluta = tolerancia_absoluta
        self._tolerancia_relativa = tolerancia_relativa
        self._numero_esperado = numero_esperado
        self._minimo_comparativas = minimo_comparativas
        self._ruta_progreso = (
            Path(ruta_progreso).resolve() if ruta_progreso is not None else None
        )
        self._chunks = self._cargar_chunks(corpus.ruta_chunks)

    @property
    def corpus(self) -> CorpusVariant:
        """Variante del corpus contra la que se realiza la evaluación."""
        return self._corpus

    @property
    def k_retrieval(self) -> int:
        """Profundidad usada para calcular recall@k."""
        return self._k

    @property
    def juez_citas(self) -> JuezCitas:
        return self._juez_citas

    @property
    def tolerancia_absoluta(self) -> float:
        return self._tolerancia_absoluta

    @property
    def tolerancia_relativa(self) -> float:
        return self._tolerancia_relativa

    @property
    def numero_esperado(self) -> int | None:
        return self._numero_esperado

    @property
    def minimo_comparativas(self) -> int:
        return self._minimo_comparativas

    @property
    def ruta_progreso(self) -> Path | None:
        return self._ruta_progreso

    def evaluar(
        self,
        *,
        nombre_agente: str,
        responder: Callable[[str], RespuestaAgente],
        ruta_jsonl: Path,
    ) -> InformeEvaluacion:
        """Ejecuta y puntúa todos los casos del fichero indicado."""
        casos = CasoGolden.cargar_jsonl(
            ruta_jsonl,
            self._corpus,
            numero_esperado=self._numero_esperado,
            minimo_comparativas=self._minimo_comparativas,
        )
        almacen = (
            AlmacenProgresoEvaluacion(self._ruta_progreso)
            if self._ruta_progreso is not None
            else None
        )
        anteriores: dict[str, ResultadoPregunta] = {}
        if almacen is not None:
            anteriores = almacen.preparar(
                self._crear_contexto_progreso(nombre_agente, ruta_jsonl, casos)
            )
            no_definitivos = {
                id_pregunta
                for id_pregunta, resultado in anteriores.items()
                if texto_indica_error_transitorio(
                    resultado.respuesta_agente.error
                )
            }
            if no_definitivos:
                almacen.descartar(no_definitivos)
                anteriores = {
                    id_pregunta: resultado
                    for id_pregunta, resultado in anteriores.items()
                    if id_pregunta not in no_definitivos
                }

        resultados = []
        for caso in casos:
            anterior = anteriores.get(caso.id)
            if anterior is not None:
                resultados.append(anterior)
                continue

            respuesta = self._responder_seguro(responder, caso)
            if texto_indica_error_transitorio(respuesta.error):
                tipo_error = (
                    RateLimitAgotadoError
                    if texto_indica_rate_limit(respuesta.error)
                    else ErrorProveedorAgotadoError
                )
                raise tipo_error(
                    f"La evaluación se detuvo en {caso.id} por un error "
                    "transitorio del modelo. "
                    "Los casos anteriores permanecen guardados."
                )
            resultado = self._evaluar_caso(caso, respuesta)
            resultados.append(resultado)
            if almacen is not None:
                almacen.registrar(resultado)

        return InformeEvaluacion(
            nombre_agente=nombre_agente,
            ruta_jsonl=str(ruta_jsonl),
            resultados=tuple(resultados),
            k_retrieval=self._k,
            tolerancia_absoluta=self._tolerancia_absoluta,
            tolerancia_relativa=self._tolerancia_relativa,
            metodo_soporte_citas=self._nombre_juez(),
        )

    @staticmethod
    def _responder_seguro(
        responder: Callable[[str], RespuestaAgente], caso: CasoGolden
    ) -> RespuestaAgente:
        try:
            respuesta = responder(caso.pregunta)
        except Exception as exc:
            if es_error_transitorio_modelo(exc):
                tipo_error = (
                    RateLimitAgotadoError
                    if es_error_rate_limit(exc)
                    else ErrorProveedorAgotadoError
                )
                raise tipo_error(
                    f"Error transitorio al responder {caso.id}."
                ) from exc
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

    def _evaluar_caso(
        self, caso: CasoGolden, respuesta: RespuestaAgente
    ) -> ResultadoPregunta:
        observaciones = []
        trayectoria = self._trayectoria_correcta(caso, respuesta)
        if not trayectoria:
            observaciones.append("La trayectoria no contiene las llamadas esperadas.")
        if respuesta.error:
            observaciones.append(f"Error del agente: {respuesta.error}")

        cifra_correcta = None
        if caso.familia in {"numerica", "comparativa"}:
            cifra_correcta = self._cifra_correcta(caso, respuesta)
            if not cifra_correcta:
                observaciones.append("La cifra o su unidad no coinciden.")

        cita_existe = None
        cita_respalda = None
        recall_at_k = None
        if caso.familia in {"extractiva", "comparativa"}:
            cita_existe, cita_respalda, error_juez = self._evaluar_citas(
                caso, respuesta
            )
            recall_at_k = self._calcular_recall(caso, respuesta)
            if not cita_existe:
                observaciones.append("Falta una cita o algún chunk_id no existe.")
            elif not cita_respalda:
                observaciones.append("La cita no respalda completamente la respuesta.")
            if error_juez:
                observaciones.append(f"Error del juez de citas: {error_juez}")

        return ResultadoPregunta(
            id_pregunta=caso.id,
            familia=caso.familia,
            respuesta_agente=respuesta,
            cita_existe=cita_existe,
            cita_respalda=cita_respalda,
            cifra_correcta=cifra_correcta,
            trayectoria_correcta=trayectoria,
            recall_at_k=recall_at_k,
            observaciones=tuple(observaciones),
        )

    def _evaluar_citas(
        self, caso: CasoGolden, respuesta: RespuestaAgente
    ) -> tuple[bool, bool, str | None]:
        ids = respuesta.citas
        cita_existe = bool(ids) and all(chunk_id in self._chunks for chunk_id in ids)
        if not cita_existe:
            return False, False, None

        evidencias = tuple(self._chunks[chunk_id] for chunk_id in ids)
        contiene_ancla = any(
            caso.ancla_texto in evidencia for evidencia in evidencias
        )
        if not contiene_ancla:
            return True, False, None
        try:
            respaldo_semantico = bool(
                self._juez_citas(respuesta.respuesta, evidencias)
            )
        except Exception as exc:
            if es_error_transitorio_modelo(exc):
                tipo_error = (
                    RateLimitAgotadoError
                    if es_error_rate_limit(exc)
                    else ErrorProveedorAgotadoError
                )
                raise tipo_error(
                    f"Error transitorio al juzgar las citas de {caso.id}."
                ) from exc
            return True, False, formatear_excepcion_modelo(exc)
        return True, respaldo_semantico, None

    def _cifra_correcta(
        self, caso: CasoGolden, respuesta: RespuestaAgente
    ) -> bool:
        if respuesta.cifra is None or caso.cifra_esperada is None:
            return False
        if not math.isfinite(respuesta.cifra):
            return False
        if respuesta.unidad is None or caso.unidad is None:
            return False
        unidad_correcta = (
            respuesta.unidad.strip().casefold() == caso.unidad.strip().casefold()
        )
        cifra_correcta = math.isclose(
            respuesta.cifra,
            caso.cifra_esperada,
            rel_tol=self._tolerancia_relativa,
            abs_tol=self._tolerancia_absoluta,
        )
        return unidad_correcta and cifra_correcta

    def _trayectoria_correcta(
        self, caso: CasoGolden, respuesta: RespuestaAgente
    ) -> bool:
        llamadas = tuple(llamada for llamada in respuesta.llamadas if llamada.exitosa)
        ejercicios = self._ejercicios_requeridos(caso)
        for nombre in caso.herramienta_esperada:
            if nombre == "list_available":
                if not any(l.nombre == nombre for l in llamadas):
                    return False
            elif nombre == "get_xbrl_fact":
                if not all(
                    any(
                        self._coincide_xbrl(llamada, caso, ejercicio)
                        for llamada in llamadas
                    )
                    for ejercicio in ejercicios
                ):
                    return False
            elif nombre in {"search_filings", "read_section"}:
                if not all(
                    any(
                        self._coincide_texto(llamada, caso, ejercicio, nombre)
                        for llamada in llamadas
                    )
                    for ejercicio in ejercicios
                ):
                    return False
        return True

    @staticmethod
    def _coincide_xbrl(
        llamada: LlamadaHerramienta, caso: CasoGolden, ejercicio: int
    ) -> bool:
        args = llamada.argumentos
        return (
            llamada.nombre == "get_xbrl_fact"
            and str(args.get("ticker", "")).upper() == caso.ticker
            and EvaluadorFinanciero._entero(args.get("fiscal_year")) == ejercicio
            and args.get("concept") == caso.concept_xbrl
        )

    @staticmethod
    def _coincide_texto(
        llamada: LlamadaHerramienta,
        caso: CasoGolden,
        ejercicio: int,
        nombre: str,
    ) -> bool:
        args = llamada.argumentos
        return (
            llamada.nombre == nombre
            and str(args.get("ticker", "")).upper() == caso.ticker
            and EvaluadorFinanciero._entero(args.get("fiscal_year")) == ejercicio
            and args.get("item") == caso.item_esperado
        )

    @staticmethod
    def _entero(valor: object) -> int | None:
        try:
            return int(valor)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _ejercicios_requeridos(caso: CasoGolden) -> tuple[int, ...]:
        if caso.familia != "comparativa":
            return (caso.fiscal_year,)
        mencionados = sorted(
            {int(anio) for anio in re.findall(r"\b20\d{2}\b", caso.pregunta)}
        )
        if len(mencionados) >= 2:
            return tuple(mencionados)
        return (caso.fiscal_year - 1, caso.fiscal_year)

    def _calcular_recall(
        self, caso: CasoGolden, respuesta: RespuestaAgente
    ) -> float:
        recuperados = {
            chunk_id
            for llamada in respuesta.llamadas
            if llamada.nombre == "search_filings" and llamada.exitosa
            for chunk_id in llamada.chunk_ids[: self._k]
        }
        encontro_ancla = any(
            caso.ancla_texto in self._chunks.get(chunk_id, "")
            for chunk_id in recuperados
        )
        return float(encontro_ancla)

    @staticmethod
    def _cargar_chunks(ruta: Path) -> dict[str, str]:
        chunks = {}
        with ruta.open(encoding="utf-8") as fichero:
            for numero_linea, linea in enumerate(fichero, start=1):
                if not linea.strip():
                    continue
                try:
                    fila = json.loads(linea)
                    chunk_id = fila["chunk_id"]
                    texto = fila["texto"]
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(
                        f"Chunk inválido en {ruta}:{numero_linea}."
                    ) from exc
                if chunk_id in chunks:
                    raise ValueError(f"chunk_id duplicado: {chunk_id}")
                chunks[chunk_id] = texto
        return chunks

    def _nombre_juez(self) -> str:
        return getattr(
            self._juez_citas,
            "__name__",
            type(self._juez_citas).__name__,
        )

    def _crear_contexto_progreso(
        self,
        nombre_agente: str,
        ruta_jsonl: Path,
        casos: tuple[CasoGolden, ...],
    ) -> ContextoProgreso:
        parametros_juez = getattr(self._juez_citas, "parametros", None)
        firma = {
            "corpus": self._corpus.model_dump(mode="json"),
            "k_retrieval": self._k,
            "tolerancia_absoluta": self._tolerancia_absoluta,
            "tolerancia_relativa": self._tolerancia_relativa,
            "numero_esperado": self._numero_esperado,
            "minimo_comparativas": self._minimo_comparativas,
            "juez": self._nombre_juez(),
            "parametros_juez": (
                dict(parametros_juez) if parametros_juez is not None else None
            ),
        }
        serializado = json.dumps(
            firma, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return ContextoProgreso(
            nombre_agente=nombre_agente,
            sha256_golden=calcular_sha256(ruta_jsonl),
            firma_evaluacion=hashlib.sha256(serializado).hexdigest(),
            ids_casos=tuple(caso.id for caso in casos),
        )
