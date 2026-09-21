"""v000 · Baseline del grupo con el modelo de Hugo (Claude).

Réplica exacta de ``experimentos/baseline.py`` (retriever denso plano, sin
filtros; mismas tools, prompt, límites y evaluación) cambiando solo el
modelo. Es la referencia contra la que se comparan las versiones de Hugo:
el baseline del grupo está medido con Gemini y comparar contra él mezclaría
el efecto del modelo con el de los cambios.

Ejecución:
    python -m agente --variante experimentos.hugo.agente_v000 \
        --evaluar golden_set.jsonl \
        --salida experimentos/hugo/resultados/v000_propio.csv
"""

from __future__ import annotations

from taller_nlp import ConstructorAgente

from experimentos.baseline import crear_corpus_baseline, crear_fabrica_baseline
from experimentos.hugo.comun import crear_constructor_hugo
from experimentos.hugo.retrievers import RetrieverFaissSeguro

NOMBRE = "hugo-v000-baseline-claude"


def crear_constructor() -> ConstructorAgente:
    corpus = crear_corpus_baseline()
    # Igual que crear_retriever_baseline (denso plano, sin filtros), con el
    # cerrojo de codificación que evita el segfault de las búsquedas en
    # paralelo. Mismo ranking.
    retriever = RetrieverFaissSeguro(
        corpus,
        nombre="faiss-baseline-s1",
        aplicar_filtros_metadatos=False,
    )
    fabrica = crear_fabrica_baseline(corpus, retriever)
    return crear_constructor_hugo(NOMBRE, corpus, fabrica)
