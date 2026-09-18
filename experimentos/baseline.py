"""Baseline reproducible del notebook de la sesión 1.

Este módulo es la referencia común del equipo. Está dividido en pequeñas
factorías para que una variante pueda reutilizar la configuración estable y
sustituir únicamente el componente que pretende estudiar.

No lee ficheros de claves ni realiza llamadas de red al importarse.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel

from taller_nlp import (
    AgenteFinanciero,
    ConfiguracionAgente,
    ControlPeticionesModelo,
    ErrorTransitorioModeloAgotado,
    ConstructorAgente,
    CorpusVariant,
    FabricaHerramientas,
    JuezCitas,
    JuezCitasLangChain,
    ManifiestoExperimento,
    ProgresoConsolaMiddleware,
    RegistroTelemetriaAuxiliar,
    RetrieverFaiss,
    VERSION_PROTOCOLO_CITAS,
)
from taller_nlp.hashing import calcular_sha256


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
DIRECTORIO_CORPUS = RAIZ_PROYECTO / "corpus"
DIRECTORIO_RESULTADOS = Path(__file__).resolve().parent / "resultados"

MODELO_AGENTE = os.getenv(
    "TALLER_MODELO_AGENTE",
    "openrouter:google/gemini-3.8-flash",
)
MODELO_JUEZ = os.getenv("TALLER_MODELO_JUEZ", MODELO_AGENTE)

# Tarifas docentes fijadas en el notebook S2 (USD por millón de tokens).
# Para un modelo no incluido se conserva coste=None antes que estimar con una
# tarifa que no le corresponde.
PRECIOS_MODELOS: dict[str, tuple[float, float]] = {
    "openrouter:google/gemini-3.5-flash-lite": (0.30, 2.50),
    "openrouter:google/gemini-3.8-flash": (0.75, 3.75),
    "openrouter:anthropic/claude-opus-5": (5.00, 25.00),
    "openrouter:anthropic/claude-fable-5.1": (10.00, 50.00),
}

SYSTEM_PROMPT = """Eres un analista financiero que responde preguntas sobre informes
10-K usando ÚNICAMENTE las herramientas disponibles.

Reglas:
- Para cualquier CIFRA, usa get_xbrl_fact. Nunca leas un número de la prosa.
- Para riesgos, estrategia o comentarios de la dirección, usa search_filings.
- Si no sabes si una compañía o un ejercicio están en el corpus, empieza por
  list_available.
- El corpus está en inglés: escribe las consultas de búsqueda en inglés.
- Cita el chunk_id del fragmento en el que te apoyes.
- Si el dato no está en el corpus, dilo. No lo estimes.
"""


def crear_corpus_baseline() -> CorpusVariant:
    """Declara los artefactos originales repartidos con el notebook."""
    return CorpusVariant(
        nombre="baseline-notebook-s1",
        ruta_secciones=DIRECTORIO_CORPUS / "secciones.jsonl",
        ruta_xbrl=DIRECTORIO_CORPUS / "xbrl_facts.parquet",
        ruta_chunks=DIRECTORIO_CORPUS / "chunks.jsonl",
        sha256_secciones=(
            "823272082bcf84fb14f6de415d087f2ee708b77dc475bcf3c3ff1a2d1e487dd1"
        ),
        sha256_xbrl=(
            "f802fc89c2dba96dfef3dd2ef5025e5540620358fc2944e5303093a929127610"
        ),
        sha256_chunks=(
            "388ff3671742c2248e8f5cb1c75afbc786edfa0dc6f72a62d1fadf2310ec82b2"
        ),
        troceador="baseline-proporcionado-s1",
        parametros_troceado={},
        ruta_indice_faiss=DIRECTORIO_CORPUS / "indice" / "corpus.faiss",
        ruta_chunks_meta=(
            DIRECTORIO_CORPUS / "indice" / "chunks_meta.parquet"
        ),
        modelo_embeddings="BAAI/bge-small-en-v1.5",
        dimension_embeddings=384,
        prefijo_consulta=(
            "Represent this sentence for searching relevant passages: "
        ),
    )


def crear_configuracion_baseline() -> ConfiguracionAgente:
    """Conserva modelo y prompt del notebook y añade límites de seguridad."""
    precios = PRECIOS_MODELOS.get(MODELO_AGENTE)
    return ConfiguracionAgente(
        modelo=MODELO_AGENTE,
        system_prompt=SYSTEM_PROMPT,
        temperatura=0,
        precio_entrada_usd_millon_tokens=precios[0] if precios else None,
        precio_salida_usd_millon_tokens=precios[1] if precios else None,
        max_iteraciones=6,
        max_llamadas_total=24,
        limites_por_herramienta={},
        timeout_s=60,
        # La cuenta nueva admite 20 RPM. Usamos 18 para dejar margen frente
        # a la ventana móvil del proveedor.
        solicitudes_modelo_por_minuto=18,
        max_reintentos_rate_limit=3,
        espera_inicial_rate_limit_s=5,
        factor_espera_rate_limit=2,
        espera_maxima_rate_limit_s=20,
        jitter_rate_limit_s=0.5,
    )


def crear_retriever_baseline(corpus: CorpusVariant) -> RetrieverFaiss:
    """Reproduce el denso plano, sin el arreglo de metadatos de S2."""
    return RetrieverFaiss(
        corpus,
        nombre="faiss-baseline-s1",
        aplicar_filtros_metadatos=False,
    )


def crear_fabrica_baseline(
    corpus: CorpusVariant,
    retriever: RetrieverFaiss,
) -> FabricaHerramientas:
    """Usa las implementaciones baseline de las cuatro herramientas."""
    return FabricaHerramientas(corpus, retriever=retriever)


def crear_constructor_baseline(
    *,
    juez_citas: JuezCitas | None = None,
    middlewares: Sequence[AgentMiddleware] = (),
    modelo: BaseChatModel | None = None,
    control_peticiones: ControlPeticionesModelo | None = None,
    telemetria_auxiliar: RegistroTelemetriaAuxiliar | None = None,
    ruta_progreso: str | Path | None = None,
) -> ConstructorAgente:
    """Ensambla el baseline permitiendo inyectar dobles en tests."""
    corpus = crear_corpus_baseline()
    retriever = crear_retriever_baseline(corpus)
    fabrica = crear_fabrica_baseline(corpus, retriever)
    configuracion = crear_configuracion_baseline()
    control = control_peticiones or ControlPeticionesModelo.desde_configuracion(
        configuracion
    )
    juez = (
        juez_citas
        if juez_citas is not None
        else JuezCitasLangChain(MODELO_JUEZ, control_peticiones=control)
    )
    return ConstructorAgente(
        nombre="baseline-notebook-s1",
        configuracion=configuracion,
        corpus=corpus,
        fabrica_herramientas=fabrica,
        juez_citas=juez,
        middlewares=middlewares,
        modelo=modelo,
        control_peticiones=control,
        telemetria_auxiliar=telemetria_auxiliar,
        k_retrieval=5,
        tolerancia_absoluta=0.0,
        tolerancia_relativa=0.01,
        # Se deja abierto para poder usar tanto las 20 preguntas propias como
        # las 10 preguntas ciegas sin construir otro agente.
        numero_esperado=None,
        minimo_comparativas=0,
        ruta_progreso=ruta_progreso,
    )


def crear_constructor() -> ConstructorAgente:
    """Factoría común usada por la interfaz configurable del notebook."""
    return crear_constructor_baseline()


def crear_agente_baseline(
    *,
    juez_citas: JuezCitas | None = None,
    middlewares: Sequence[AgentMiddleware] = (),
    modelo: BaseChatModel | None = None,
    control_peticiones: ControlPeticionesModelo | None = None,
    telemetria_auxiliar: RegistroTelemetriaAuxiliar | None = None,
    ruta_progreso: str | Path | None = None,
) -> tuple[ConstructorAgente, AgenteFinanciero]:
    """Devuelve constructor y fachada para responder o evaluar."""
    constructor = crear_constructor_baseline(
        juez_citas=juez_citas,
        middlewares=middlewares,
        modelo=modelo,
        control_peticiones=control_peticiones,
        telemetria_auxiliar=telemetria_auxiliar,
        ruta_progreso=ruta_progreso,
    )
    return constructor, constructor.construir()


def _ruta_manifiesto_automatica() -> Path:
    marca = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DIRECTORIO_RESULTADOS / f"baseline_{marca}.json"


def _ruta_progreso_automatica(ruta_golden: Path) -> Path:
    huella = calcular_sha256(ruta_golden)[:12]
    return (
        DIRECTORIO_RESULTADOS
        / "progreso"
        / (
            f"baseline-notebook-s1_{ruta_golden.stem}_{huella}_"
            f"citas-v{VERSION_PROTOCOLO_CITAS}.json"
        )
    )


def _informar_reintento(
    numero: int, espera_s: float, error: BaseException
) -> None:
    causa = (
        "rate limit"
        if type(error).__name__ == "TooManyRequestsResponseError"
        else "error transitorio del proveedor"
    )
    print(
        f"[baseline] {causa}: reintento {numero} en {espera_s:.1f} s...",
        file=sys.stderr,
        flush=True,
    )


def _crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ejecuta el baseline reproducible del notebook.",
    )
    accion = parser.add_mutually_exclusive_group(required=True)
    accion.add_argument("--pregunta", help="Pregunta individual que responder.")
    accion.add_argument(
        "--evaluar",
        type=Path,
        metavar="RUTA_JSONL",
        help="Golden set que se evaluará completo.",
    )
    parser.add_argument(
        "--manifiesto",
        type=Path,
        help=(
            "Ruta de salida del manifiesto. En evaluación se genera una "
            "automática si se omite."
        ),
    )
    parser.add_argument(
        "--progreso",
        type=Path,
        help=(
            "Fichero de progreso reanudable. Solo se usa con --evaluar; si "
            "se omite se crea uno automático en experimentos/resultados/."
        ),
    )
    parser.add_argument(
        "--reiniciar-progreso",
        action="store_true",
        help="Descarta explícitamente el progreso previo de esta evaluación.",
    )
    parser.add_argument(
        "--revision",
        help="Commit, etiqueta o identificador de la revisión del código.",
    )
    return parser


def main(argumentos: Sequence[str] | None = None) -> int:
    parser = _crear_parser()
    opciones = parser.parse_args(argumentos)
    if not os.getenv("OPENROUTER_API_KEY"):
        parser.error(
            "Falta OPENROUTER_API_KEY en el entorno. No guardes la clave "
            "dentro del script."
        )

    if opciones.pregunta is not None and (
        opciones.progreso is not None or opciones.reiniciar_progreso
    ):
        parser.error("--progreso solo puede utilizarse junto con --evaluar.")

    ruta_progreso = None
    if opciones.evaluar is not None:
        ruta_progreso = opciones.progreso or _ruta_progreso_automatica(
            opciones.evaluar
        )
        if opciones.reiniciar_progreso and ruta_progreso.exists():
            ruta_progreso.unlink()
        print(
            f"[baseline] Progreso reanudable: {ruta_progreso}",
            file=sys.stderr,
            flush=True,
        )

    configuracion = crear_configuracion_baseline()
    control = ControlPeticionesModelo(
        solicitudes_por_minuto=configuracion.solicitudes_modelo_por_minuto,
        max_reintentos=configuracion.max_reintentos_rate_limit,
        espera_inicial_s=configuracion.espera_inicial_rate_limit_s,
        factor_espera=configuracion.factor_espera_rate_limit,
        espera_maxima_s=configuracion.espera_maxima_rate_limit_s,
        jitter_s=configuracion.jitter_rate_limit_s,
        al_reintentar=_informar_reintento,
    )

    print("[baseline] Construyendo el agente...", file=sys.stderr, flush=True)
    constructor, agente = crear_agente_baseline(
        middlewares=(ProgresoConsolaMiddleware(),),
        control_peticiones=control,
        ruta_progreso=ruta_progreso,
    )
    print("[baseline] Agente construido.", file=sys.stderr, flush=True)
    informe = None
    if opciones.pregunta is not None:
        respuesta = agente.responder(opciones.pregunta)
        print(respuesta.model_dump_json(indent=2))
    else:
        try:
            informe = agente.evaluar(opciones.evaluar)
        except ErrorTransitorioModeloAgotado as exc:
            print(f"[baseline] {exc}", file=sys.stderr, flush=True)
            print(
                "[baseline] Vuelve a ejecutar el mismo comando para continuar.",
                file=sys.stderr,
                flush=True,
            )
            return 2
        print(informe.model_dump_json(indent=2))

    ruta_manifiesto = opciones.manifiesto
    if informe is not None and ruta_manifiesto is None:
        ruta_manifiesto = _ruta_manifiesto_automatica()
    if ruta_manifiesto is not None:
        ruta_manifiesto.parent.mkdir(parents=True, exist_ok=True)
        manifiesto = ManifiestoExperimento.desde_constructor(
            constructor,
            informe=informe,
            revision_codigo=opciones.revision,
            metadatos={"origen": "experimentos.baseline"},
        )
        guardado = manifiesto.guardar(ruta_manifiesto)
        print(f"Manifiesto guardado en {guardado}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
