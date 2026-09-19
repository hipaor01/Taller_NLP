"""Contratos públicos compartidos por todas las variantes del agente."""

from statistics import fmean
from typing import Literal
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    computed_field,
    field_validator,
    model_validator,
)


NombreHerramienta = Literal[
    "list_available",
    "get_xbrl_fact",
    "search_filings",
    "read_section",
]

FuenteRespuesta = Literal["xbrl", "texto", "ambas", "ninguna"]
FamiliaPregunta = Literal["extractiva", "numerica", "comparativa"]
VERSION_PROTOCOLO_CITAS = 4


class RespuestaFinanciera(BaseModel):
    """Respuesta trazable a una pregunta sobre informes 10-K."""

    respuesta: str = Field(description="Respuesta en prosa, breve y directa")
    cifra: float | None = Field(
        default=None, description="Valor numérico, si la pregunta pide uno"
    )
    unidad: str | None = Field(
        default=None, description="USD, shares, porcentaje…"
    )
    ticker: str | None = None
    ejercicio: int | None = None
    fuente: FuenteRespuesta = Field(
        description="De dónde sale el dato. 'ninguna' si no está en el corpus"
    )
    cita: str | None = Field(
        default=None,
        description="Texto literal del informe que respalda la respuesta",
    )
    chunk_id: str | None = Field(
        default=None,
        description="Identificador del fragmento citado, para verificar",
    )


class LlamadaHerramienta(BaseModel):
    """Traza normalizada de una llamada a una herramienta del agente."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, description="Identificador de la tool call.")
    nombre: NombreHerramienta
    argumentos: dict[str, JsonValue] = Field(default_factory=dict)
    duracion_ms: float = Field(ge=0)
    chunk_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Identificadores recuperados, en orden de ranking. Solo suele "
            "tener contenido para search_filings."
        ),
    )
    resultado: str | None = Field(
        default=None,
        description="Salida de la herramienta, que puede guardarse truncada.",
    )
    error: str | None = None

    @model_validator(mode="after")
    def validar_resultado_o_error(self) -> "LlamadaHerramienta":
        """Una llamada terminada tiene resultado o error, pero no ambos."""
        if (self.resultado is None) == (self.error is None):
            raise ValueError(
                "La llamada debe contener exactamente uno de resultado o error."
            )
        if self.nombre != "search_filings" and self.chunk_ids:
            raise ValueError(
                "Solo search_filings puede registrar chunk_ids recuperados."
            )
        return self

    @property
    def exitosa(self) -> bool:
        """Indica si la herramienta terminó sin lanzar una excepción."""
        return self.error is None


class LlamadaModeloAuxiliar(BaseModel):
    """Telemetría de una llamada a modelo ajena al bucle principal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    componente: str = Field(min_length=1)
    modelo: str = Field(min_length=1)
    latencia_ms: float = Field(ge=0)
    tokens_entrada: int | None = Field(default=None, ge=0)
    tokens_salida: int | None = Field(default=None, ge=0)
    coste_usd: float | None = Field(default=None, ge=0)
    error: str | None = None

    @field_validator("componente", "modelo", "error")
    @classmethod
    def normalizar_texto_auxiliar(cls, valor: str | None) -> str | None:
        if valor is None:
            return None
        normalizado = valor.strip()
        if not normalizado:
            raise ValueError("El texto no puede estar vacío.")
        return normalizado


class RespuestaAgente(BaseModel):
    """Respuesta final y traza observable de una ejecución del agente."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    respuesta: str = Field(
        description="Texto final presentado al usuario; puede estar vacío si falló."
    )
    cifra: float | None = None
    unidad: str | None = None
    fuente: FuenteRespuesta
    citas: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Identificadores de los fragmentos citados por el agente.",
    )
    cita: str | None = Field(
        default=None,
        description="Extracto textual que el agente atribuye al fragmento citado.",
    )
    llamadas: tuple[LlamadaHerramienta, ...] = Field(default_factory=tuple)
    llamadas_modelo_auxiliares: tuple[LlamadaModeloAuxiliar, ...] = Field(
        default_factory=tuple,
    )
    latencia_ms: float = Field(ge=0)
    coste_usd: float | None = Field(
        default=None,
        ge=0,
        description="Coste de la ejecución; None indica telemetría no disponible.",
    )
    tokens_entrada: int | None = Field(default=None, ge=0)
    tokens_salida: int | None = Field(default=None, ge=0)
    error: str | None = None

    @field_validator("cita")
    @classmethod
    def normalizar_cita(cls, valor: str | None) -> str | None:
        if valor is None:
            return None
        normalizado = valor.strip()
        if not normalizado:
            raise ValueError("cita debe contener texto o ser None.")
        return normalizado

    @model_validator(mode="after")
    def validar_contenido(self) -> "RespuestaAgente":
        """Impide respuestas vacías que tampoco expliquen un fallo."""
        if not self.respuesta.strip() and self.error is None:
            raise ValueError(
                "Una ejecución sin error debe contener una respuesta no vacía."
            )
        if self.cifra is None and self.unidad is not None:
            raise ValueError("No puede indicarse una unidad sin una cifra.")
        return self

    @property
    def exitosa(self) -> bool:
        """Indica si el agente completó la ejecución sin error técnico."""
        return self.error is None

    @property
    def tokens_totales(self) -> int | None:
        """Suma el uso de tokens cuando ambas cantidades están disponibles."""
        if self.tokens_entrada is None or self.tokens_salida is None:
            return None
        return self.tokens_entrada + self.tokens_salida


class ResultadoPregunta(BaseModel):
    """Evaluación desglosada de una pregunta del golden set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id_pregunta: str = Field(min_length=1)
    familia: FamiliaPregunta
    respuesta_agente: RespuestaAgente
    cita_existe: bool | None = None
    cita_respalda: bool | None = None
    justificacion_cita: str | None = None
    cifra_correcta: bool | None = None
    trayectoria_correcta: bool
    recall_at_k: float | None = Field(default=None, ge=0, le=1)
    observaciones: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validar_criterios_de_la_familia(self) -> "ResultadoPregunta":
        """Exige que estén evaluados todos los criterios aplicables."""
        if self.familia in {"extractiva", "comparativa"}:
            if self.cita_existe is None or self.cita_respalda is None:
                raise ValueError(
                    f"La familia {self.familia} requiere evaluar la cita."
                )
        if self.familia in {"numerica", "comparativa"}:
            if self.cifra_correcta is None:
                raise ValueError(
                    f"La familia {self.familia} requiere evaluar la cifra."
                )
        return self

    @computed_field
    @property
    def acierto(self) -> bool:
        """Indica si se cumplen todos los criterios de su familia."""
        criterios = [self.trayectoria_correcta]
        if self.familia in {"extractiva", "comparativa"}:
            criterios.extend([self.cita_existe, self.cita_respalda])
        if self.familia in {"numerica", "comparativa"}:
            criterios.append(self.cifra_correcta)
        return all(criterio is True for criterio in criterios)


class InformeEvaluacion(BaseModel):
    """Informe agregado de la evaluación de una variante del agente."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nombre_agente: str = Field(min_length=1)
    ruta_jsonl: str = Field(min_length=1)
    resultados: tuple[ResultadoPregunta, ...] = Field(min_length=1)
    k_retrieval: int = Field(gt=0)
    tolerancia_absoluta: float = Field(ge=0)
    tolerancia_relativa: float = Field(ge=0)
    metodo_soporte_citas: str = Field(default="no_especificado", min_length=1)
    version_protocolo_citas: Literal[4] = VERSION_PROTOCOLO_CITAS

    @model_validator(mode="after")
    def validar_ids_unicos(self) -> "InformeEvaluacion":
        """Evita contabilizar dos veces una misma pregunta."""
        ids = [resultado.id_pregunta for resultado in self.resultados]
        if len(ids) != len(set(ids)):
            raise ValueError("Los id_pregunta del informe deben ser únicos.")
        return self

    @computed_field
    @property
    def numero_preguntas(self) -> int:
        return len(self.resultados)

    @computed_field
    @property
    def aciertos_totales(self) -> int:
        return sum(resultado.acierto for resultado in self.resultados)

    @computed_field
    @property
    def tasa_acierto_global(self) -> float:
        return self.aciertos_totales / self.numero_preguntas

    @computed_field
    @property
    def preguntas_por_familia(self) -> dict[FamiliaPregunta, int]:
        return {
            familia: sum(r.familia == familia for r in self.resultados)
            for familia in ("extractiva", "numerica", "comparativa")
        }

    @computed_field
    @property
    def aciertos_por_familia(self) -> dict[FamiliaPregunta, int]:
        return {
            familia: sum(
                r.familia == familia and r.acierto for r in self.resultados
            )
            for familia in ("extractiva", "numerica", "comparativa")
        }

    @computed_field
    @property
    def tasa_acierto_por_familia(self) -> dict[FamiliaPregunta, float | None]:
        return {
            familia: (
                self.aciertos_por_familia[familia]
                / self.preguntas_por_familia[familia]
                if self.preguntas_por_familia[familia]
                else None
            )
            for familia in ("extractiva", "numerica", "comparativa")
        }

    @computed_field
    @property
    def recall_at_k_medio(self) -> float | None:
        valores = [
            r.recall_at_k for r in self.resultados if r.recall_at_k is not None
        ]
        return fmean(valores) if valores else None

    @computed_field
    @property
    def cobertura_recall(self) -> float:
        evaluadas = sum(r.recall_at_k is not None for r in self.resultados)
        return evaluadas / self.numero_preguntas

    @computed_field
    @property
    def coste_medio_usd(self) -> float | None:
        costes = [
            r.respuesta_agente.coste_usd
            for r in self.resultados
            if r.respuesta_agente.coste_usd is not None
        ]
        return fmean(costes) if costes else None

    @computed_field
    @property
    def cobertura_coste(self) -> float:
        medidas = sum(
            r.respuesta_agente.coste_usd is not None for r in self.resultados
        )
        return medidas / self.numero_preguntas

    @computed_field
    @property
    def latencia_media_ms(self) -> float:
        return fmean(r.respuesta_agente.latencia_ms for r in self.resultados)

    @computed_field
    @property
    def llamadas_por_pregunta(self) -> float:
        return fmean(len(r.respuesta_agente.llamadas) for r in self.resultados)
