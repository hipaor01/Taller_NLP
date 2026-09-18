"""CLI de la interfaz entregable, independiente de la variante elegida."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .interfaz import VARIABLE_VARIANTE, evaluar, resumir


def _texto_no_vacio(valor: str) -> str:
    normalizado = valor.strip()
    if not normalizado:
        raise argparse.ArgumentTypeError("el valor no puede estar vacío")
    return normalizado


def _crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agente",
        description=(
            "Evalúa la variante configurada del agente o resume una tabla "
            "de evaluación existente."
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
    parser.add_argument(
        "--salida",
        type=Path,
        metavar="RUTA_CSV",
        help="Guarda en esta ruta la tabla compatible con el notebook.",
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
            "Nombre de versión para --resumir; por defecto se usa el nombre "
            "del fichero CSV."
        ),
    )
    return parser


def main(argumentos: Sequence[str] | None = None) -> int:
    parser = _crear_parser()
    opciones = parser.parse_args(argumentos)
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
            parser.error("--etiqueta solo se puede usar con --resumir")
        if opciones.variante is not None:
            os.environ[VARIABLE_VARIANTE] = opciones.variante
        tabla = evaluar(opciones.evaluar, salida=opciones.salida)

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
