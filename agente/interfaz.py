"""Adaptador entre el agente del proyecto y el contrato del notebook S2."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from taller_nlp import (
        CasoGolden,
        ConstructorAgente,
        InformeEvaluacion,
        MotorLangChain,
    )

# El notebook añade la raíz del repositorio a sys.path, pero el paquete del
# proyecto usa layout src/. Este ajuste permite utilizar la interfaz desde un
# clon limpio sin exigir que el propio notebook instale el paquete editable.
_RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
_DIRECTORIO_SRC = _RAIZ_PROYECTO / "src"
if str(_DIRECTORIO_SRC) not in sys.path:
    sys.path.insert(0, str(_DIRECTORIO_SRC))

VARIABLE_VARIANTE = "TALLER_VARIANTE_AGENTE"
MODULO_BASELINE = "experimentos.baseline"
NOMBRE_FACTORIA = "crear_constructor"
_COLUMNAS_EVALUACION = (
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
)


def _informe_a_dataframe(
    informe: InformeEvaluacion,
    casos: tuple[CasoGolden, ...],
) -> pd.DataFrame:
    """Adapta el informe auditable a la tabla exigida por el notebook S2."""
    casos_por_id = {caso.id: caso for caso in casos}
    filas = []
    for resultado in informe.resultados:
        caso = casos_por_id.get(resultado.id_pregunta)
        if caso is None:
            raise ValueError(
                f"El informe contiene un caso ausente del JSONL: "
                f"{resultado.id_pregunta}."
            )
        respuesta = resultado.respuesta_agente
        cita = (
            None
            if resultado.familia == "numerica"
            else bool(resultado.cita_existe and resultado.cita_respalda)
        )
        filas.append(
            {
                "id": resultado.id_pregunta,
                "familia": resultado.familia,
                "ticker": caso.ticker,
                "latencia_s": respuesta.latencia_ms / 1_000,
                "coste_usd": respuesta.coste_usd or 0.0,
                "llamadas": len(respuesta.llamadas),
                "cita": cita,
                "cifra": resultado.cifra_correcta,
                "trayectoria": resultado.trayectoria_correcta,
                "recall": resultado.recall_at_k,
                "error": respuesta.error,
            }
        )
    return pd.DataFrame(filas, columns=_COLUMNAS_EVALUACION)


def resumir(tabla: pd.DataFrame, etiqueta: str) -> dict[str, float | str]:
    """Convierte la evaluación detallada en una fila del informe comparativo."""
    if not isinstance(tabla, pd.DataFrame):
        raise TypeError("tabla debe ser un pandas.DataFrame.")
    if not isinstance(etiqueta, str):
        raise TypeError("etiqueta debe ser texto.")
    etiqueta_normalizada = etiqueta.strip()
    if not etiqueta_normalizada:
        raise ValueError("etiqueta no puede estar vacía.")

    def tasa(columna: str) -> float:
        valores = tabla[columna].dropna() if columna in tabla else ()
        return float(valores.mean()) if len(valores) else float("nan")

    return {
        "versión": etiqueta_normalizada,
        "cita": tasa("cita"),
        "cifra": tasa("cifra"),
        "trayectoria": tasa("trayectoria"),
        "recall@5": tasa("recall"),
        "coste medio (¢)": (
            float(tabla["coste_usd"].mean() * 100)
            if "coste_usd" in tabla
            else float("nan")
        ),
        "latencia media (s)": (
            float(tabla["latencia_s"].mean())
            if "latencia_s" in tabla
            else float("nan")
        ),
        "llamadas/pregunta": (
            float(tabla["llamadas"].mean())
            if "llamadas" in tabla
            else float("nan")
        ),
    }


def tabla_desde_manifiesto(
    ruta_manifiesto: str | Path,
) -> pd.DataFrame:
    """Recupera la tabla ya evaluada sin volver a ejecutar el agente."""
    from taller_nlp import CasoGolden, ManifiestoExperimento

    manifiesto = ManifiestoExperimento.cargar(ruta_manifiesto)
    informe = manifiesto.informe
    if informe is None:
        raise ValueError("El manifiesto no contiene un informe de evaluación.")

    ruta_golden = Path(informe.ruta_jsonl)
    casos = CasoGolden.cargar_jsonl(
        ruta_golden,
        manifiesto.corpus,
        numero_esperado=manifiesto.numero_esperado,
        minimo_comparativas=manifiesto.minimo_comparativas,
    )
    return _informe_a_dataframe(informe, casos)


def _crear_constructor_configurado(
    *,
    ruta_progreso: str | Path | None = None,
) -> ConstructorAgente:
    """Carga la factoría común de la variante seleccionada por el entorno."""
    nombre_modulo = os.getenv(VARIABLE_VARIANTE, MODULO_BASELINE).strip()
    if not nombre_modulo:
        raise ValueError(f"{VARIABLE_VARIANTE} no puede estar vacía.")

    modulo = import_module(nombre_modulo)
    fabrica = getattr(modulo, NOMBRE_FACTORIA, None)
    if not callable(fabrica):
        raise AttributeError(
            f"{nombre_modulo} debe exponer una función {NOMBRE_FACTORIA}()."
        )
    constructor = fabrica()
    if ruta_progreso is not None:
        constructor = constructor.con_ruta_progreso(ruta_progreso)
    return constructor


@dataclass(frozen=True, slots=True)
class _RuntimeNotebook:
    """Componentes compartidos durante toda la sesión del notebook."""

    constructor: ConstructorAgente
    motor: MotorLangChain

    @classmethod
    def crear(
        cls,
        *,
        ruta_progreso: str | Path | None = None,
    ) -> "_RuntimeNotebook":
        from langgraph.checkpoint.memory import InMemorySaver

        constructor = _crear_constructor_configurado(
            ruta_progreso=ruta_progreso,
        )
        motor = constructor.construir_motor(checkpointer=InMemorySaver())
        return cls(constructor=constructor, motor=motor)

    def responder(self, pregunta: str, thread_id: str | None = None) -> dict[str, Any]:
        if not isinstance(pregunta, str):
            raise TypeError("pregunta debe ser texto.")
        pregunta_normalizada = pregunta.strip()
        if not pregunta_normalizada:
            raise ValueError("La pregunta no puede estar vacía.")

        ejecucion = self.motor.ejecutar(
            pregunta_normalizada,
            thread_id=thread_id if thread_id is not None else "s2",
        )
        return {
            **ejecucion.estado,
            # El notebook multiplica directamente este valor y no admite None.
            "coste_usd": ejecucion.coste_usd or 0.0,
            "latencia_s": ejecucion.latencia_s,
            "tokens_entrada": ejecucion.tokens_entrada,
            "tokens_salida": ejecucion.tokens_salida,
            "llamadas_modelo_auxiliares": (
                ejecucion.llamadas_modelo_auxiliares
            ),
            "coste_modelo_principal_usd": (
                ejecucion.coste_modelo_principal_usd
            ),
            "tokens_entrada_modelo_principal": (
                ejecucion.tokens_entrada_modelo_principal
            ),
            "tokens_salida_modelo_principal": (
                ejecucion.tokens_salida_modelo_principal
            ),
        }

    def evaluar_informe(self, ruta_jsonl: str | Path) -> InformeEvaluacion:
        """Ejecuta la evaluación y conserva su informe estructurado completo."""
        ruta = Path(ruta_jsonl)
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el golden set: {ruta}")
        return self.constructor.evaluador.evaluar(
            nombre_agente=self.constructor.nombre,
            responder=self.motor.responder,
            ruta_jsonl=ruta,
        )

    def evaluar(
        self,
        ruta_jsonl: str | Path,
        *,
        salida: str | Path | None = None,
    ) -> pd.DataFrame:
        """Evalúa un JSONL y devuelve la tabla compatible con el notebook."""
        from taller_nlp import CasoGolden

        ruta = Path(ruta_jsonl)
        informe = self.evaluar_informe(ruta)
        evaluador = self.constructor.evaluador
        casos = CasoGolden.cargar_jsonl(
            ruta,
            self.constructor.corpus,
            numero_esperado=evaluador.numero_esperado,
            minimo_comparativas=evaluador.minimo_comparativas,
        )
        tabla = _informe_a_dataframe(informe, casos)
        if salida is not None:
            ruta_salida = Path(salida)
            ruta_salida.parent.mkdir(parents=True, exist_ok=True)
            tabla.to_csv(ruta_salida, index=False)
        return tabla


_RUNTIME: _RuntimeNotebook | None = None
_RUNTIME_LOCK = Lock()


def _obtener_runtime() -> _RuntimeNotebook:
    """Construye una vez el motor y conserva su memoria entre llamadas."""
    global _RUNTIME
    if _RUNTIME is None:
        with _RUNTIME_LOCK:
            if _RUNTIME is None:
                _RUNTIME = _RuntimeNotebook.crear()
    return _RUNTIME


def responder(pregunta: str, thread_id: str | None = None) -> dict[str, Any]:
    """Responde con el estado y la telemetría que consume el notebook S2."""
    return _obtener_runtime().responder(pregunta, thread_id=thread_id)


def evaluar(
    ruta_jsonl: str | Path,
    *,
    salida: str | Path | None = None,
    ruta_progreso: str | Path | None = None,
) -> pd.DataFrame:
    """Evalúa un JSONL y devuelve la tabla exigida para el informe S2."""
    runtime = (
        _RuntimeNotebook.crear(ruta_progreso=ruta_progreso)
        if ruta_progreso is not None
        else _obtener_runtime()
    )
    return runtime.evaluar(ruta_jsonl, salida=salida)


def evaluar_informe(
    ruta_jsonl: str | Path,
    *,
    ruta_progreso: str | Path | None = None,
) -> InformeEvaluacion:
    """Devuelve el informe estructurado completo para análisis y auditoría."""
    runtime = (
        _RuntimeNotebook.crear(ruta_progreso=ruta_progreso)
        if ruta_progreso is not None
        else _obtener_runtime()
    )
    return runtime.evaluar_informe(ruta_jsonl)
