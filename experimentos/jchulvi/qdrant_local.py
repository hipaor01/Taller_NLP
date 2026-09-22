"""Infraestructura Qdrant del experimento; importar no arranca Docker ni modelos."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

import numpy as np
import pandas as pd

URL = "http://127.0.0.1:6339"
CONTENEDOR = "taller-nlp-jchulvi-v002"
IMAGEN = "qdrant/qdrant@sha256:12364fe851b9f17356fc88189fc06d1b521262e04659ec7345975b00c9246a10"
_SESIONES_LOCK = RLock()
_SESIONES_ACTIVAS = 0


def validar_vectores(vectores, filas, dimension=None):
    if (vectores.ndim != 2 or vectores.shape[0] != filas or not vectores.shape[1]
            or (dimension is not None and vectores.shape[1] != dimension)
            or vectores.dtype != np.float32 or not np.isfinite(vectores).all()
            or not np.allclose(np.linalg.norm(vectores, axis=1), 1, atol=1e-5)):
        raise ValueError("Índice cloud inválido: filas, dimensión, finitud o normalización.")


def _peticion(url, ruta, cuerpo=None, metodo=None):
    peticion = urllib.request.Request(
        url.rstrip("/") + ruta,
        data=None if cuerpo is None else json.dumps(cuerpo).encode(),
        headers={"Content-Type": "application/json"}, method=metodo,
    )
    with urllib.request.urlopen(peticion, timeout=30) as respuesta:
        return json.load(respuesta)


def consultar(url, coleccion, vector, *, limite=5, ticker=None, fiscal_year=None, item=None):
    """Búsqueda exacta con filtros; no arranca Docker ni genera embeddings."""
    if not coleccion or limite < 1:
        raise ValueError("Se necesita una colección y un límite positivo.")
    filtros = [{"key": campo, "match": {"value": valor}}
               for campo, valor in (("ticker", ticker), ("fiscal_year", fiscal_year), ("item", item))
               if valor is not None]
    cuerpo = {"query": np.asarray(vector).tolist(), "limit": limite, "with_payload": True,
              "params": {"exact": True}, "filter": {"must": filtros}}
    return _peticion(url, f"/collections/{coleccion}/points/query", cuerpo)["result"]["points"]


class QdrantLocal:
    """Start/load/query/stop del contenedor propio y del índice cloud existente.

    La construcción solo lee el manifiesto local. ``sesion`` garantiza la parada
    si falla la carga o el trabajo del cuaderno. ``detener`` permite recuperación
    manual tras perder el kernel, sin borrar contenedores, índices ni datos.
    """
    url = URL

    def __init__(self, directorio, *, ruta_indice, corpus, modelo, contrato):
        self.directorio = Path(directorio)
        self.ruta_indice = Path(ruta_indice)
        self.corpus = corpus
        self.modelo = modelo
        self.contrato = contrato
        manifiesto = self.ruta_indice / "manifest.json"
        self.metadatos = json.loads(manifiesto.read_text()) if manifiesto.is_file() else None
        self.coleccion = ("jchulvi_cloud_" + self.ruta_indice.name.removeprefix("embeddings_cloud_")
                          + "_" + self.metadatos["sha256_vectores"][:16]) if self.metadatos else None

    @staticmethod
    def _verificar_propietario(info):
        if ((info["Config"]["Labels"] or {}).get("taller.owner") != "jchulvi-v002"
                or info["Config"]["Image"] != IMAGEN):
            raise RuntimeError("El contenedor no pertenece a este estudio; no se modifica.")

    @property
    def en_marcha(self):
        info = json.loads(subprocess.check_output(["docker", "inspect", CONTENEDOR], text=True))[0]
        self._verificar_propietario(info)
        return info["State"]["Running"]

    def iniciar(self):
        """Arranca solo el contenedor propio, sin reutilizar una sesión en curso."""
        inspeccion = subprocess.run(["docker", "inspect", CONTENEDOR], capture_output=True, text=True)
        if inspeccion.returncode == 0:
            info = json.loads(inspeccion.stdout)[0]
            self._verificar_propietario(info)
            if info["State"]["Running"]:
                raise RuntimeError("El contenedor ya está en uso; termina esa ejecución primero.")
            subprocess.run(["docker", "start", CONTENEDOR], check=True, capture_output=True)
        else:
            subprocess.run([
                "docker", "run", "-d", "--name", CONTENEDOR,
                "--label", "taller.owner=jchulvi-v002", "--cpus", "2", "--memory", "2g",
                "-e", "QDRANT__TELEMETRY_DISABLED=true",
                "-p", "127.0.0.1:6339:6333", IMAGEN,
            ], check=True, capture_output=True)
        try:
            for intento in range(60):
                try:
                    _peticion(self.url, "/collections")
                    return self
                except (urllib.error.URLError, ConnectionError):
                    if intento == 59:
                        raise
                    time.sleep(0.25)
        except BaseException:
            self.detener()
            raise

    def cargar(self):
        """Verifica y sube el índice existente; reanuda lotes mediante IDs estables."""
        manifiesto = self.ruta_indice / "manifest.json"
        if not manifiesto.is_file():
            raise FileNotFoundError("Prepara primero los embeddings cloud; no se cargarán modelos locales.")
        meta = json.loads(manifiesto.read_text())
        vectores = np.load(self.ruta_indice / "vectores.npy", allow_pickle=False)
        filas = pd.read_json(self.corpus.ruta_chunks, lines=True)
        assert meta == self.metadatos, "El índice cambió: recrea QdrantLocal con el manifiesto nuevo."
        assert meta["contrato"] == self.contrato
        assert meta["sha256_chunks"] == self.corpus.sha256_chunks
        assert meta["modelo"] == self.modelo
        validar_vectores(vectores, len(filas), meta["dimension"])
        assert meta["chunk_ids"] == filas.chunk_id.tolist()
        assert meta["sha256_vectores"] == hashlib.sha256((self.ruta_indice / "vectores.npy").read_bytes()).hexdigest()
        ruta = "/collections/" + self.coleccion
        existentes = _peticion(self.url, "/collections")["result"]["collections"]
        if self.coleccion not in {c["name"] for c in existentes}:
            _peticion(self.url, ruta, {"vectors": {"size": vectores.shape[1], "distance": "Dot"}}, "PUT")
        info = _peticion(self.url, ruta)["result"]
        esquema = info["config"]["params"]["vectors"]
        assert esquema["size"] == vectores.shape[1] and esquema["distance"] == "Dot"
        if info["points_count"] != len(vectores):
            for inicio in range(0, len(vectores), 128):
                puntos = [{
                    "id": i, "vector": vectores[i].tolist(),
                    "payload": {"chunk_id": str(filas.iloc[i].chunk_id), "ticker": str(filas.iloc[i].ticker),
                                "fiscal_year": int(filas.iloc[i].fiscal_year), "item": str(filas.iloc[i]["item"])},
                } for i in range(inicio, min(inicio + 128, len(vectores)))]
                _peticion(self.url, ruta + "/points?wait=true", {"points": puntos}, "PUT")
        cuenta = _peticion(self.url, ruta)["result"]["points_count"]
        assert cuenta == len(vectores), "Índice incompleto; no se evalúa con datos parciales."
        return cuenta

    def consultar(self, vector, *, limite=5, ticker=None, fiscal_year=None, item=None):
        """Devuelve puntos con score y payload; el vector lo prepara el llamante."""
        return consultar(self.url, self.coleccion, vector, limite=limite,
                         ticker=ticker, fiscal_year=fiscal_year, item=item)

    def detener(self):
        """Parada normal o recuperación manual; no elimina almacenamiento."""
        if self.en_marcha:
            subprocess.run(["docker", "stop", "--time", "10", CONTENEDOR], check=True, capture_output=True)
        if self.en_marcha:
            raise RuntimeError("Qdrant no se ha detenido.")
        self.directorio.mkdir(parents=True, exist_ok=True)
        (self.directorio / "docker_estado.json").write_text(json.dumps({
            "contenedor": CONTENEDOR, "imagen": IMAGEN, "running": False,
            "comprobado_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

    @contextmanager
    def sesion(self, *, cargar_indice=True):
        """Comparte sesiones anidadas y garantiza la parada de la última."""
        global _SESIONES_ACTIVAS
        with _SESIONES_LOCK:
            if _SESIONES_ACTIVAS == 0:
                self.iniciar()
            _SESIONES_ACTIVAS += 1
        try:
            if cargar_indice:
                self.cargar()
            yield self
        finally:
            with _SESIONES_LOCK:
                _SESIONES_ACTIVAS -= 1
                if _SESIONES_ACTIVAS == 0:
                    self.detener()
