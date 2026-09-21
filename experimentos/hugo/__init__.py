"""Variantes del agente de Hugo.

Al importarse fija tres variables de entorno ANTES de que se carguen faiss y
torch (este paquete se importa antes que cualquier variante). Son una
precaución: faiss-cpu, torch y scikit-learn traen cada uno su propia copia de
``libomp.dylib`` en macOS. NO eran la causa del ``segmentation fault`` del
2026-09-20: ``faulthandler`` mostró que el fallo venía de dos búsquedas en
paralelo codificando a la vez (se corrige en ``retrievers.RetrieverFaissSeguro``).
Son valores por defecto: si ya están definidas, se respetan.

- ``KMP_DUPLICATE_LIB_OK=TRUE``: tolera las copias duplicadas de OpenMP.
- ``OMP_NUM_THREADS=1``: sin hilos OpenMP. Solo se codifica una consulta
  corta cada vez, así que no hay pérdida de velocidad apreciable.
- ``TOKENIZERS_PARALLELISM=false``: evita que el tokenizador abra procesos
  (el aviso de «leaked semaphore» al terminar).
"""

import os

for _variable, _valor in (
    ("KMP_DUPLICATE_LIB_OK", "TRUE"),
    ("OMP_NUM_THREADS", "1"),
    ("TOKENIZERS_PARALLELISM", "false"),
):
    os.environ.setdefault(_variable, _valor)
