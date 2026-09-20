"""Raíz de composición del agente financiero concreto."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import final

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from .agent import AgenteFinanciero
from .auxiliary_telemetry import RegistroTelemetriaAuxiliar
from .config import ConfiguracionAgente
from .corpus import CorpusVariant
from .evaluation import EvaluadorFinanciero
from .langchain_engine import MotorLangChain
from .model_resilience import ControlPeticionesModelo
from .contracts import RespuestaFinanciera
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
        "_middlewares",
        "_esquema_respuesta",
        "_modelo",
        "_control_peticiones",
        "_telemetria_auxiliar",
    )

    def __init__(
        self,
        nombre: str,
        configuracion: ConfiguracionAgente,
        corpus: CorpusVariant,
        fabrica_herramientas: FabricaHerramientas,
        *,
        middlewares: Sequence[AgentMiddleware] = (),
        esquema_respuesta: type[RespuestaFinanciera] = RespuestaFinanciera,
        modelo: BaseChatModel | None = None,
        control_peticiones: ControlPeticionesModelo | None = None,
        telemetria_auxiliar: RegistroTelemetriaAuxiliar | None = None,
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
        self._nombre = nombre_normalizado
        self._configuracion = configuracion
        self._corpus = corpus
        self._fabrica_herramientas = fabrica_herramientas
        self._middlewares = tuple(middlewares)
        if not isinstance(esquema_respuesta, type) or not issubclass(
            esquema_respuesta,
            RespuestaFinanciera,
        ):
            raise TypeError(
                "esquema_respuesta debe heredar de RespuestaFinanciera."
            )
        self._esquema_respuesta = esquema_respuesta
        self._modelo = modelo
        self._control_peticiones = control_peticiones or (
            ControlPeticionesModelo.desde_configuracion(configuracion)
        )
        if telemetria_auxiliar is not None and not isinstance(
            telemetria_auxiliar,
            RegistroTelemetriaAuxiliar,
        ):
            raise TypeError(
                "telemetria_auxiliar debe ser un RegistroTelemetriaAuxiliar."
            )
        self._telemetria_auxiliar = telemetria_auxiliar
        # Crear el evaluador aquí reutiliza su propia validación de opciones y
        # lo liga necesariamente al mismo corpus que el resto de componentes.
        self._evaluador = EvaluadorFinanciero(
            corpus,
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
    def middlewares(self) -> tuple[AgentMiddleware, ...]:
        return self._middlewares

    @property
    def esquema_respuesta(self) -> type[RespuestaFinanciera]:
        return self._esquema_respuesta

    @property
    def modelo(self) -> BaseChatModel | None:
        return self._modelo

    @property
    def control_peticiones(self) -> ControlPeticionesModelo:
        return self._control_peticiones

    @property
    def telemetria_auxiliar(self) -> RegistroTelemetriaAuxiliar | None:
        return self._telemetria_auxiliar

    def construir_motor(
        self,
        *,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> MotorLangChain:
        """Crea el motor LangChain con los componentes del constructor."""
        herramientas = self._fabrica_herramientas.crear()
        return MotorLangChain(
            configuracion=self._configuracion,
            herramientas=herramientas,
            corpus=self._corpus,
            middlewares=self._middlewares,
            esquema_respuesta=self._esquema_respuesta,
            modelo=self._modelo,
            control_peticiones=self._control_peticiones,
            checkpointer=checkpointer,
            telemetria_auxiliar=self._telemetria_auxiliar,
        )

    def construir(self) -> AgenteFinanciero:
        """Crea una fachada lista para responder y evaluar."""
        motor = self.construir_motor()
        return AgenteFinanciero(
            nombre=self._nombre,
            motor=motor,
            evaluador=self._evaluador,
        )

    def con_ruta_progreso(
        self,
        ruta_progreso: str | Path,
    ) -> "ConstructorAgente":
        """Copia la composición y activa progreso reanudable al evaluar."""
        evaluador = self._evaluador
        return ConstructorAgente(
            nombre=self._nombre,
            configuracion=self._configuracion,
            corpus=self._corpus,
            fabrica_herramientas=self._fabrica_herramientas,
            middlewares=self._middlewares,
            esquema_respuesta=self._esquema_respuesta,
            modelo=self._modelo,
            control_peticiones=self._control_peticiones,
            telemetria_auxiliar=self._telemetria_auxiliar,
            k_retrieval=evaluador.k_retrieval,
            tolerancia_absoluta=evaluador.tolerancia_absoluta,
            tolerancia_relativa=evaluador.tolerancia_relativa,
            numero_esperado=evaluador.numero_esperado,
            minimo_comparativas=evaluador.minimo_comparativas,
            ruta_progreso=ruta_progreso,
        )
