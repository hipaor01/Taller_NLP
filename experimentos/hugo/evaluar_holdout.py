"""Plan B para el día 24: evalúa un JSONL sin abortar por preguntas raras.

Sirve para cualquier variante del repositorio (la de Hugo, la de Higinio o el
baseline) y NO modifica el código común: construye el agente con la
``crear_constructor()`` de la variante y evalúa pregunta a pregunta con
``experimentos/hugo/holdout.py``.

Qué resuelve respecto a ``python -m agente --evaluar``:
1. Preguntas sin respuesta en el corpus (empresa o ejercicio que no está,
   concepto que no se reporta, numérica sin cifra esperada). El evaluador
   común rechaza el fichero entero; aquí cada pregunta se evalúa por su lado
   y acierta si el agente dice que el dato no está, sin inventar una cifra.
2. ``segmentation fault`` cuando el agente lanza dos ``search_filings`` a la
   vez. Mientras se ejecuta este script, y solo en este proceso, la
   codificación de consultas de ``RetrieverFaiss`` se serializa con un
   cerrojo. No se edita ningún fichero común; el ranking no cambia.
3. Si se corta, repetir el mismo comando sigue por donde iba sin volver a
   preguntar ni pagar lo ya respondido.

Uso, desde la raíz del repositorio:
    python -m experimentos.hugo.evaluar_holdout holdout.jsonl
    python -m experimentos.hugo.evaluar_holdout holdout.jsonl \
        --variante experimentos.higinio.agente_v009 \
        --salida resultados/holdout_v009.csv

Sin ``--variante`` se usa ``TALLER_VARIANTE_AGENTE`` o, si no está, el
baseline (la misma regla que ``python -m agente``).
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import sys
import threading
from functools import wraps
from importlib import import_module
from pathlib import Path

import pandas as pd

from experimentos.baseline import RAIZ_PROYECTO

VARIABLE_VARIANTE = "TALLER_VARIANTE_AGENTE"
MODULO_BASELINE = "experimentos.baseline"
DIRECTORIO_RESULTADOS = Path(__file__).resolve().parent / "resultados"

_CERROJO = threading.Lock()


def activar_cerrojo_codificacion() -> None:
    """Serializa ``RetrieverFaiss._codificar`` solo en este proceso.

    Cubre también los buscadores que envuelven a ``RetrieverFaiss`` (híbrido,
    reescritura). Es idempotente.
    """
    from taller_nlp import RetrieverFaiss

    original = RetrieverFaiss._codificar
    if getattr(original, "_hugo_serializado", False):
        return

    @wraps(original)
    def serializado(self, query):  # type: ignore[no-untyped-def]
        with _CERROJO:
            return original(self, query)

    serializado._hugo_serializado = True  # type: ignore[attr-defined]
    RetrieverFaiss._codificar = serializado  # type: ignore[method-assign]


def _huella(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()[:12]


def main(argumentos: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("jsonl", type=Path, help="Preguntas a evaluar (p. ej. holdout.jsonl)")
    parser.add_argument("--variante", help="Módulo con crear_constructor(); por defecto el baseline")
    parser.add_argument("--salida", type=Path, help="CSV de salida")
    parser.add_argument("--progreso", type=Path, help="Fichero de progreso reanudable")
    parser.add_argument("--reiniciar", action="store_true", help="Descarta el progreso previo")
    opciones = parser.parse_args(argumentos)
    faulthandler.enable()

    ruta = opciones.jsonl if opciones.jsonl.is_absolute() else Path.cwd() / opciones.jsonl
    if not ruta.is_file():
        parser.error(f"no existe {ruta}")
    variante = (opciones.variante or os.getenv(VARIABLE_VARIANTE) or MODULO_BASELINE).strip()
    etiqueta = f"{variante.rsplit('.', 1)[-1]}_{ruta.stem}_{_huella(ruta)}"
    salida = opciones.salida or DIRECTORIO_RESULTADOS / f"holdout_{etiqueta}.csv"
    progreso = opciones.progreso or DIRECTORIO_RESULTADOS / "progreso" / f"holdout_{etiqueta}.jsonl"
    if opciones.reiniciar and progreso.exists():
        progreso.unlink()

    activar_cerrojo_codificacion()
    from experimentos.hugo.holdout import cargar_holdout, evaluar_holdout, resumen_texto

    fabrica = getattr(import_module(variante), "crear_constructor", None)
    if not callable(fabrica):
        parser.error(f"{variante} no expone crear_constructor()")
    constructor = fabrica()
    motor = constructor.construir_motor()

    casos = cargar_holdout(ruta, constructor.corpus)
    tipos = pd.Series([c.tipo for c in casos]).value_counts().to_dict()
    print(
        f"[holdout] {constructor.nombre} · {len(casos)} preguntas · {tipos}\n"
        f"[holdout] Progreso: {progreso}",
        file=sys.stderr,
        flush=True,
    )
    tabla = evaluar_holdout(
        casos,
        evaluador=constructor.evaluador,
        responder=motor.responder,
        ruta_progreso=progreso,
    )

    salida.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida, index=False)
    salida.with_suffix(".json").write_text(
        json.dumps(tabla.to_dict("records"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(tabla[["id", "tipo", "familia", "acierto", "cita", "cifra", "trayectoria",
                     "fuente", "cifra_agente", "coste_usd", "latencia_s", "error"]].to_string(index=False))
    print(f"\n[holdout] {resumen_texto(tabla)}")
    print(f"[holdout] Coste total {tabla['coste_usd'].sum():.4f} $ · "
          f"latencia media {tabla['latencia_s'].mean():.1f} s")
    from experimentos.hugo.guardrails import volcar_intervenciones

    ruta_intervenciones = salida.with_name(f"{salida.stem}_intervenciones.json")
    intervenciones = volcar_intervenciones(constructor, ruta_intervenciones)
    if intervenciones:
        print(f"[holdout] Intervenciones de los middlewares: {intervenciones} "
              f"({ruta_intervenciones.name})")
    print(f"[holdout] Guardado: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
