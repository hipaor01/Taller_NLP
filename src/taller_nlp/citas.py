"""Utilidades compartidas para cargar y validar evidencia textual."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .chunking import FragmentoCorpus


CARACTERES_CITA_COMPROBADOS = 120
_ESPACIOS = re.compile(r"\s+")


def normalizar_texto(texto: str) -> str:
    """Normaliza espacios y caja sin alterar el contenido semántico."""
    return _ESPACIOS.sub(" ", texto).strip().casefold()


def cita_esta_respaldada(
    cita: str,
    texto: str,
    *,
    caracteres: int = CARACTERES_CITA_COMPROBADOS,
) -> bool:
    """Comprueba que el prefijo normalizado de una cita aparece en un texto."""
    if isinstance(caracteres, bool) or not isinstance(caracteres, int):
        raise TypeError("caracteres debe ser un entero.")
    if caracteres <= 0:
        raise ValueError("caracteres debe ser mayor que cero.")
    prefijo = normalizar_texto(cita)[:caracteres]
    return bool(prefijo) and prefijo in normalizar_texto(texto)


def cargar_fragmentos(ruta: Path) -> dict[str, FragmentoCorpus]:
    """Carga un JSONL de chunks y exige identificadores únicos y válidos."""
    chunks: dict[str, FragmentoCorpus] = {}
    with ruta.open(encoding="utf-8") as fichero:
        for numero_linea, linea in enumerate(fichero, start=1):
            if not linea.strip():
                continue
            try:
                fila = json.loads(linea)
                fragmento = FragmentoCorpus.model_validate(fila)
                chunk_id = fragmento.chunk_id
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Chunk inválido en {ruta}:{numero_linea}."
                ) from exc
            if chunk_id in chunks:
                raise ValueError(f"chunk_id duplicado: {chunk_id}")
            chunks[chunk_id] = fragmento
    return chunks
