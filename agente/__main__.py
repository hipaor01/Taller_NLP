"""CLI de la interfaz entregable, independiente de la variante elegida."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .interfaz import (
    MODULO_BASELINE,
    VARIABLE_VARIANTE,
    evaluar,
    resumir,
    tabla_desde_manifiesto,
)

from taller_nlp import VERSION_PROTOCOLO_CITAS
from taller_nlp.hashing import calcular_sha256


_RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
_DIRECTORIO_PROGRESO = (
    _RAIZ_PROYECTO / "experimentos" / "resultados" / "progreso"
)


def _texto_no_vacio(valor: str) -> str:
    normalizado = valor.strip()
    if not normalizado:
        raise argparse.ArgumentTypeError("el valor no puede estar vacío")
    return normalizado


def _crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agente",
        description=(
            "Evalúa una variante, resume un CSV o exporta un manifiesto "
            "existente sin volver a ejecutar el agente."
        ),
    )
    operacion = parser.add_mutually_exclusive_group(required=True)
    operacion.add_argument(
        "--evaluar",
        type=Path,
        metavar="RUTA_JSONL",
        help="Golden set u hold-out que se evaluará completo.",
    )
    operacion.add_argument(
        "--resumir",
        type=Path,
        metavar="RUTA_CSV",
        help="Tabla CSV de evaluación que se convertirá en una fila resumen.",
    )
    operacion.add_argument(
        "--desde-manifiesto",
        type=Path,
        metavar="RUTA_JSON",
        help=(
            "Manifiesto evaluado del que se exportarán el CSV detallado y "
            "su resumen sin realizar llamadas al modelo."
        ),
    )
    parser.add_argument(
        "--salida",
        type=Path,
        metavar="RUTA_CSV",
        help="Guarda en esta ruta la tabla compatible con el notebook.",
    )
    parser.add_argument(
        "--resumen",
        type=Path,
        metavar="RUTA_CSV",
        help=(
            "Con --desde-manifiesto, guarda aquí la fila resumen del informe."
        ),
    )
    parser.add_argument(
        "--variante",
        type=_texto_no_vacio,
        metavar="MODULO",
        help=(
            "Módulo que expone crear_constructor(); si se omite se usa "
            f"{VARIABLE_VARIANTE} o el baseline."
        ),
    )
    parser.add_argument(
        "--etiqueta",
        type=_texto_no_vacio,
        metavar="TEXTO",
        help=(
            "Nombre de versión para --resumir o --desde-manifiesto; por "
            "defecto se usa el nombre del CSV detallado."
        ),
    )
    parser.add_argument(
        "--progreso",
        type=Path,
        metavar="RUTA_JSON",
        help=(
            "Fichero de progreso reanudable para --evaluar. Si se omite, "
            "se crea automáticamente en experimentos/resultados/progreso/."
        ),
    )
    parser.add_argument(
        "--reiniciar-progreso",
        action="store_true",
        help="Descarta el progreso previo antes de evaluar desde cero.",
    )
    return parser


def _ruta_progreso_automatica(
    ruta_golden: Path,
    *,
    modulo_variante: str,
) -> Path:
    """Deriva una ruta estable sin mezclar variantes ni golden sets."""
    etiqueta = modulo_variante.rsplit(".", 1)[-1]
    huella = calcular_sha256(ruta_golden)[:12]
    return _DIRECTORIO_PROGRESO / (
        f"{etiqueta}_{ruta_golden.stem}_{huella}_"
        f"citas-v{VERSION_PROTOCOLO_CITAS}.json"
    )


def main(argumentos: Sequence[str] | None = None) -> int:
    parser = _crear_parser()
    opciones = parser.parse_args(argumentos)

    if opciones.evaluar is None and (
        opciones.progreso is not None or opciones.reiniciar_progreso
    ):
        parser.error(
            "--progreso y --reiniciar-progreso solo se pueden usar con "
            "--evaluar"
        )

    if opciones.desde_manifiesto is not None:
        if opciones.variante is not None:
            parser.error("--variante solo se puede usar con --evaluar")
        if opciones.salida is None or opciones.resumen is None:
            parser.error(
                "--desde-manifiesto requiere --salida y --resumen"
            )
        if opciones.salida.resolve() == opciones.resumen.resolve():
            parser.error("--salida y --resumen deben ser rutas distintas")
        ruta = opciones.desde_manifiesto
        if not ruta.is_file():
            parser.error(f"no existe el manifiesto: {ruta}")
        try:
            tabla = tabla_desde_manifiesto(ruta)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        etiqueta = opciones.etiqueta or opciones.salida.stem
        tabla_resumen = pd.DataFrame([resumir(tabla, etiqueta)])

        opciones.salida.parent.mkdir(parents=True, exist_ok=True)
        opciones.resumen.parent.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(opciones.salida, index=False)
        tabla_resumen.to_csv(opciones.resumen, index=False)

        print(tabla.to_string(index=False))
        print("\nResumen:")
        print(tabla_resumen.to_string(index=False))
        print(
            f"Tabla guardada en {opciones.salida.resolve()}",
            file=sys.stderr,
        )
        print(
            f"Resumen guardado en {opciones.resumen.resolve()}",
            file=sys.stderr,
        )
        return 0

    if opciones.resumen is not None:
        parser.error("--resumen solo se puede usar con --desde-manifiesto")

    if opciones.resumir is not None:
        if opciones.variante is not None:
            parser.error("--variante solo se puede usar con --evaluar")
        ruta = opciones.resumir
        if not ruta.is_file():
            parser.error(f"no existe la tabla de evaluación: {ruta}")
        etiqueta = opciones.etiqueta or ruta.stem
        tabla = pd.DataFrame([resumir(pd.read_csv(ruta), etiqueta)])
    else:
        if opciones.etiqueta is not None:
            parser.error(
                "--etiqueta solo se puede usar con --resumir o "
                "--desde-manifiesto"
            )
        if opciones.variante is not None:
            os.environ[VARIABLE_VARIANTE] = opciones.variante
        modulo_variante = os.getenv(VARIABLE_VARIANTE, MODULO_BASELINE).strip()
        ruta_progreso = opciones.progreso or _ruta_progreso_automatica(
            opciones.evaluar,
            modulo_variante=modulo_variante,
        )
        if opciones.reiniciar_progreso and ruta_progreso.exists():
            ruta_progreso.unlink()
        print(
            f"Progreso reanudable: {ruta_progreso.resolve()}",
            file=sys.stderr,
            flush=True,
        )
        tabla = evaluar(
            opciones.evaluar,
            salida=opciones.salida,
            ruta_progreso=ruta_progreso,
        )

    if opciones.resumir is not None and opciones.salida is not None:
        opciones.salida.parent.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(opciones.salida, index=False)
    print(tabla.to_string(index=False))
    if opciones.salida is not None:
        print(
            f"Tabla guardada en {opciones.salida.resolve()}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
