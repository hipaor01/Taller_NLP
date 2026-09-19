"""Reescritura de consultas mediante un modelo auxiliar."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from time import perf_counter

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from .auxiliary_telemetry import RegistroTelemetriaAuxiliar
from .component import ComponenteConfigurable
from .contracts import LlamadaModeloAuxiliar
from .corpus import CorpusVariant
from .model_resilience import ControlPeticionesModelo
from .model_usage import estimar_coste_modelo, extraer_uso_modelo
from .retrieval import FragmentoRecuperado, Retriever


class ReescritorConsultaLLM(ComponenteConfigurable):
    """Traduce una consulta al vocabulario del corpus y registra su coste."""

    def __init__(
        self,
        *,
        nombre_modelo: str,
        instruccion: str,
        modelo: BaseChatModel | None,
        control_peticiones: ControlPeticionesModelo,
        telemetria: RegistroTelemetriaAuxiliar,
        reescrituras_respaldo: Mapping[str, str] | None = None,
        precio_entrada_usd_millon_tokens: float | None = None,
        precio_salida_usd_millon_tokens: float | None = None,
        nombre: str = "reescritor-consulta-llm",
    ) -> None:
        if not isinstance(nombre_modelo, str) or not nombre_modelo.strip():
            raise ValueError("nombre_modelo debe ser texto no vacío.")
        if not isinstance(instruccion, str) or not instruccion.strip():
            raise ValueError("instruccion debe ser texto no vacío.")
        if modelo is not None and not isinstance(modelo, BaseChatModel):
            raise TypeError("modelo debe ser BaseChatModel o None.")
        if not isinstance(control_peticiones, ControlPeticionesModelo):
            raise TypeError(
                "control_peticiones debe ser ControlPeticionesModelo."
            )
        if not isinstance(telemetria, RegistroTelemetriaAuxiliar):
            raise TypeError(
                "telemetria debe ser RegistroTelemetriaAuxiliar."
            )
        respaldos = {
            clave.strip(): valor.strip()
            for clave, valor in dict(reescrituras_respaldo or {}).items()
            if clave.strip() and valor.strip()
        }
        if (
            precio_entrada_usd_millon_tokens is None
        ) != (precio_salida_usd_millon_tokens is None):
            raise ValueError("Las dos tarifas deben declararse juntas.")

        super().__init__(
            nombre,
            parametros={
                "modelo": nombre_modelo.strip(),
                "instruccion": instruccion.strip(),
                "tiene_respaldo": bool(respaldos),
            },
        )
        self._nombre_modelo = nombre_modelo.strip()
        self._instruccion = instruccion.strip()
        self._modelo = modelo
        self._control_peticiones = control_peticiones
        self._telemetria = telemetria
        self._respaldos = respaldos
        self._precio_entrada = precio_entrada_usd_millon_tokens
        self._precio_salida = precio_salida_usd_millon_tokens

    @property
    def modelo_disponible(self) -> bool:
        return self._modelo is not None

    def reescribir(
        self,
        pregunta: str,
        id_golden: str | None = None,
    ) -> str:
        """Devuelve una consulta inglesa o, ante un fallo, el respaldo."""
        if not isinstance(pregunta, str) or not pregunta.strip():
            raise ValueError("pregunta debe ser texto no vacío.")
        pregunta_normalizada = pregunta.strip()
        if id_golden is not None and not isinstance(id_golden, str):
            raise TypeError("id_golden debe ser texto o None.")
        id_normalizado = id_golden.strip() if id_golden else ""
        respaldo = self._respaldos.get(id_normalizado, pregunta_normalizada)
        if self._modelo is None:
            return respaldo

        inicio = perf_counter()
        try:
            respuesta = self._control_peticiones.ejecutar(
                lambda: self._modelo.invoke(
                    [
                        {"role": "system", "content": self._instruccion},
                        {"role": "user", "content": pregunta_normalizada},
                    ]
                )
            )
            if not isinstance(respuesta, AIMessage):
                raise TypeError("El reescritor no devolvió un AIMessage.")
            reescrita = str(respuesta.text).strip()
            if not reescrita:
                raise ValueError("El reescritor devolvió una consulta vacía.")
        except Exception as exc:
            self._registrar(
                inicio=inicio,
                tokens_entrada=None,
                tokens_salida=None,
                coste_usd=None,
                error=f"{type(exc).__name__}: {exc}",
            )
            return respaldo

        tokens_entrada, tokens_salida, coste_usd = extraer_uso_modelo(
            (respuesta,)
        )
        if coste_usd is None:
            coste_usd = estimar_coste_modelo(
                tokens_entrada,
                tokens_salida,
                self._precio_entrada,
                self._precio_salida,
            )
        self._registrar(
            inicio=inicio,
            tokens_entrada=tokens_entrada,
            tokens_salida=tokens_salida,
            coste_usd=coste_usd,
            error=None,
        )
        return reescrita

    def _registrar(
        self,
        *,
        inicio: float,
        tokens_entrada: int | None,
        tokens_salida: int | None,
        coste_usd: float | None,
        error: str | None,
    ) -> None:
        self._telemetria.registrar(
            LlamadaModeloAuxiliar(
                componente=self.nombre,
                modelo=self._nombre_modelo,
                latencia_ms=(perf_counter() - inicio) * 1_000,
                tokens_entrada=tokens_entrada,
                tokens_salida=tokens_salida,
                coste_usd=coste_usd,
                error=error,
            )
        )


class RetrieverConReescritura(Retriever):
    """Aplica una reescritura LLM antes de delegar en otro retriever."""

    def __init__(
        self,
        corpus: CorpusVariant,
        *,
        retriever_base: Retriever,
        reescritor: ReescritorConsultaLLM,
        nombre: str = "retriever-con-reescritura-llm",
    ) -> None:
        if not isinstance(retriever_base, Retriever):
            raise TypeError("retriever_base debe ser una instancia de Retriever.")
        if retriever_base.corpus != corpus:
            raise ValueError("El retriever base debe usar el mismo corpus.")
        if not isinstance(reescritor, ReescritorConsultaLLM):
            raise TypeError("reescritor debe ser ReescritorConsultaLLM.")
        super().__init__(
            nombre,
            corpus,
            parametros={
                "retriever_base": retriever_base.nombre,
                "parametros_retriever_base": dict(
                    retriever_base.parametros
                ),
                "reescritor": reescritor.nombre,
                "modelo_reescritor": reescritor.parametros["modelo"],
                "instruccion_reescritor": (
                    reescritor.parametros["instruccion"]
                ),
                "reescrituras_respaldo": (
                    reescritor.parametros["tiene_respaldo"]
                ),
            },
            aplicar_filtros_metadatos=(
                retriever_base.aplica_filtros_metadatos
            ),
        )
        self._retriever_base = retriever_base
        self._reescritor = reescritor

    @property
    def retriever_base(self) -> Retriever:
        return self._retriever_base

    @property
    def reescritor(self) -> ReescritorConsultaLLM:
        return self._reescritor

    def _buscar(
        self,
        *,
        query: str,
        ticker: str | None,
        fiscal_year: int | None,
        item: str | None,
        k: int,
    ) -> Iterable[FragmentoRecuperado]:
        query_reescrita = self._reescritor.reescribir(query)
        return self._retriever_base.buscar(
            query_reescrita,
            ticker=ticker,
            fiscal_year=fiscal_year,
            item=item,
            k=k,
        )
