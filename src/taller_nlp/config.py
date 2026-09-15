"""Configuración declarativa de una variante del agente."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import NombreHerramienta


class ConfiguracionAgente(BaseModel):
    """Parámetros comunes de modelo, prompt y límites de ejecución."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    modelo: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    temperatura: float = Field(default=0, ge=0, le=2)
    max_tokens_salida: int | None = Field(default=None, gt=0)
    max_iteraciones: int = Field(default=8, gt=0)
    max_llamadas_total: int = Field(default=8, gt=0)
    limites_por_herramienta: dict[NombreHerramienta, int] = Field(
        default_factory=dict
    )
    timeout_s: float = Field(
        default=120,
        gt=0,
        description="Timeout por llamada al modelo y por superstep de LangGraph.",
    )
    solicitudes_modelo_por_minuto: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Límite cliente compartido por agente y juez; None lo desactiva."
        ),
    )
    max_reintentos_rate_limit: int = Field(default=3, ge=0)
    espera_inicial_rate_limit_s: float = Field(default=5, ge=0)
    factor_espera_rate_limit: float = Field(default=2, ge=1)
    espera_maxima_rate_limit_s: float = Field(default=20, ge=0)
    jitter_rate_limit_s: float = Field(default=0.5, ge=0)

    @field_validator("modelo", "system_prompt")
    @classmethod
    def validar_texto_no_vacio(cls, valor: str) -> str:
        normalizado = valor.strip()
        if not normalizado:
            raise ValueError("El texto no puede estar vacío.")
        return normalizado

    @field_validator("limites_por_herramienta")
    @classmethod
    def validar_limites_no_negativos(
        cls, limites: dict[NombreHerramienta, int]
    ) -> dict[NombreHerramienta, int]:
        if any(limite < 0 for limite in limites.values()):
            raise ValueError(
                "Los límites por herramienta deben ser mayores o iguales a cero."
            )
        return limites

    @model_validator(mode="after")
    def validar_coherencia_de_limites(self) -> "ConfiguracionAgente":
        if any(
            limite > self.max_llamadas_total
            for limite in self.limites_por_herramienta.values()
        ):
            raise ValueError(
                "Ningún límite por herramienta puede superar "
                "max_llamadas_total."
            )
        if (
            self.espera_inicial_rate_limit_s
            > self.espera_maxima_rate_limit_s
        ):
            raise ValueError(
                "espera_inicial_rate_limit_s no puede superar "
                "espera_maxima_rate_limit_s."
            )
        return self
