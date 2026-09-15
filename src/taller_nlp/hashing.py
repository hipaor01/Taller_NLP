"""Utilidades de huella compartidas por los artefactos del corpus."""

from __future__ import annotations

import hashlib
from pathlib import Path


def calcular_sha256(ruta: Path) -> str:
    """Calcula el SHA-256 de un fichero por bloques."""
    digest = hashlib.sha256()
    with ruta.open("rb") as fichero:
        for bloque in iter(lambda: fichero.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()
