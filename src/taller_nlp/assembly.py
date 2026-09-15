"""Raíz de composición del agente financiero concreto."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import final

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel

from .agent import AgenteFinanciero
from .config import ConfiguracionAgente
from .corpus import CorpusVariant
from .evaluation import EvaluadorFinanciero, JuezCitas
from .langchain_engine import MotorLangChain
from .model_resilience import ControlPeticionesModelo
from .tool_factory import FabricaHerramientas


@final
class ConstructorAgente:
    """Ensambla una variante completa alrededor de un único corpus.

    Es la raíz de composición compartida por los equipos. Las implementaciones
    variables entran mediante ``FabricaHerramientas``, el modelo y los
    middlewares. El motor y el evaluador se construyen aquí para impedir que
    cada experimento conecte los componentes de forma distinta.
    """

    __slots__ = (
        "_nombre",
        "_configuracion",
        "_corpus",
        "_fabrica_herramientas",
        "_evaluador",
        "_juez_citas",
        "_middlewares",
        "_modelo",
        "_control_peticiones",
    )

    def __init__(
        self,
        nombre: str,
        configuracion: ConfiguracionAgente,
        corpus: CorpusVariant,
        fabrica_herramientas: FabricaHerramientas,
        juez_citas: JuezCitas,
        *,
        middlewares: Sequence[AgentMiddleware] = (),
        modelo: BaseChatModel | None = None,
        control_peticiones: ControlPeticionesModelo | None = None,
        k_retrieval: int = 5,
        tolerancia_absoluta: float = 1.0,
        tolerancia_relativa: float = 1e-6,
        numero_esperado: int | None = None,
        minimo_comparativas: int = 0,
        ruta_progreso: str | Path | None = None,
    ) -> None:
        if not isinstance(nombre, str):
            raise TypeError("nombre debe ser texto.")
        nombre_normalizado = nombre.strip()
        if not nombre_normalizado:
            raise ValueError("nombre no puede estar vacío.")
        if not isinstance(configuracion, ConfiguracionAgente):
            raise TypeError(
                "configuracion debe ser una ConfiguracionAgente."
            )
        if not isinstance(corpus, CorpusVariant):
            raise TypeError("corpus debe ser una CorpusVariant.")
        if not isinstance(fabrica_herramientas, FabricaHerramientas):
            raise TypeError(
                "fabrica_herramientas debe ser una FabricaHerramientas."
            )
        if fabrica_herramientas.corpus != corpus:
            raise ValueError(
                "La fábrica de herramientas y el constructor deben usar la "
                "misma CorpusVariant."
            )
        if not callable(juez_citas):
            raise TypeError("juez_citas debe ser callable.")

        self._nombre = nombre_normalizado
        self._configuracion = configuracion
        self._corpus = corpus
        self._fabrica_herramientas = fabrica_herramientas
        self._juez_citas = juez_citas
        self._middlewares = tuple(middlewares)
        self._modelo = modelo
        self._control_peticiones = control_peticiones or (
            ControlPeticionesModelo.desde_configuracion(configuracion)
        )
        # Crear el evaluador aquí reutiliza su propia validación de opciones y
        # lo liga necesariamente al mismo corpus que el resto de componentes.
        self._evaluador = EvaluadorFinanciero(
            corpus,
            juez_citas,
            k_retrieval=k_retrieval,
            tolerancia_absoluta=tolerancia_absoluta,
            tolerancia_relativa=tolerancia_relativa,
            numero_esperado=numero_esperado,
            minimo_comparativas=minimo_comparativas,
            ruta_progreso=ruta_progreso,
        )

    @property
    def nombre(self) -> str:
        return self._nombre

    @property
    def configuracion(self) -> ConfiguracionAgente:
        return self._configuracion

    @property
    def corpus(self) -> CorpusVariant:
        return self._corpus

    @property
    def fabrica_herramientas(self) -> FabricaHerramientas:
        return self._fabrica_herramientas

    @property
    def evaluador(self) -> EvaluadorFinanciero:
        return self._evaluador

    @property
    def juez_citas(self) -> JuezCitas:
        return self._juez_citas

    @property
    def middlewares(self) -> tuple[AgentMiddleware, ...]:
        return self._middlewares

    @property
    def modelo(self) -> BaseChatModel | None:
        return self._modelo

    @property
    def control_peticiones(self) -> ControlPeticionesModelo:
        return self._control_peticiones

    def construir(self) -> AgenteFinanciero:
        """Crea una fachada lista para responder y evaluar."""
        herramientas = self._fabrica_herramientas.crear()
        motor = MotorLangChain(
            configuracion=self._configuracion,
            herramientas=herramientas,
            corpus=self._corpus,
            middlewares=self._middlewares,
            modelo=self._modelo,
            control_peticiones=self._control_peticiones,
        )
        return AgenteFinanciero(
            nombre=self._nombre,
            motor=motor,
            evaluador=self._evaluador,
        )
