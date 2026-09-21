"""Middlewares propios de Hugo: comprobación de cifras y reparación de citas.

Dos piezas independientes, cada una con su versión:

- ``ComprobadorCifrasXBRL`` (v004) contrasta la cifra de la respuesta
  estructurada con los hechos XBRL del corpus y, si no cuadra, devuelve al
  modelo el desajuste concreto para que lo corrija. Es el middleware
  obligatorio del enunciado.
- ``ReparadorCitas`` (v005) no habla con el modelo: reescribe la cita con el
  texto literal del fragmento cuando el modelo la ha copiado con apóstrofos
  rectos, comillas añadidas o elipsis. Nunca inventa una cita: solo puede
  sustituirla por una subcadena exacta de un fragmento que el propio agente
  recuperó.

Ninguna de las dos toca código común: se enchufan al motor con el parámetro
``middlewares`` de ``ConstructorAgente`` (ver ``comun.py``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from pydantic import JsonValue

from taller_nlp import CorpusVariant, RespuestaFinanciera
from taller_nlp.citas import cargar_fragmentos, cita_esta_respaldada
from taller_nlp.retrieval import extraer_chunk_ids_formateados

TOLERANCIA_RELATIVA = 0.01
MARCA_CIFRAS = "COMPROBACIÓN XBRL"
_COLUMNAS_XBRL = ("ticker", "fiscal_year", "concept", "value", "unit")
_MAXIMO_HECHOS_LISTADOS = 15


# ---------------------------------------------------------------------------
# Lectura de la traza (mensajes del grafo)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class HechoXBRL:
    """Un hecho del parquet XBRL, con su valor exacto."""

    ticker: str
    ejercicio: int
    concepto: str
    valor: float
    unidad: str

    def __str__(self) -> str:
        return (
            f"{self.ticker} FY{self.ejercicio} {self.concepto} = "
            f"{self.valor:,.15g} {self.unidad}"
        )


def llamadas_por_nombre(mensajes: object, nombre: str) -> tuple[dict[str, Any], ...]:
    """Argumentos de las tool calls de ``nombre`` que aparecen en la traza."""
    if not isinstance(mensajes, (list, tuple)):
        return ()
    llamadas = []
    for mensaje in mensajes:
        if not isinstance(mensaje, AIMessage):
            continue
        for llamada in mensaje.tool_calls:
            if llamada.get("name") != nombre:
                continue
            argumentos = llamada.get("args")
            llamadas.append(argumentos if isinstance(argumentos, dict) else {})
    return tuple(llamadas)


def chunks_recuperados(mensajes: object) -> tuple[str, ...]:
    """IDs devueltos por las búsquedas de esta ejecución, en orden."""
    if not isinstance(mensajes, (list, tuple)):
        return ()
    ids_busqueda = {
        llamada.get("id")
        for mensaje in mensajes
        if isinstance(mensaje, AIMessage)
        for llamada in mensaje.tool_calls
        if llamada.get("name") == "search_filings" and llamada.get("id")
    }
    encontrados: list[str] = []
    for mensaje in mensajes:
        if not isinstance(mensaje, ToolMessage) or mensaje.status == "error":
            continue
        if mensaje.name != "search_filings" and (
            mensaje.tool_call_id not in ids_busqueda
        ):
            continue
        encontrados.extend(extraer_chunk_ids_formateados(mensaje.text))
    return tuple(dict.fromkeys(encontrados))


def contiene_marca(mensajes: object, marca: str) -> bool:
    """Detecta una corrección ya enviada, para no entrar en bucle."""
    if not isinstance(mensajes, (list, tuple)):
        return False
    for mensaje in mensajes:
        if isinstance(mensaje, dict):
            contenido = mensaje.get("content", "")
        else:
            contenido = getattr(mensaje, "text", None)
            if contenido is None:
                contenido = getattr(mensaje, "content", "")
        if marca in str(contenido):
            return True
    return False


def pregunta_de(mensajes: object) -> str | None:
    """Texto de la primera pregunta del usuario en la traza."""
    if not isinstance(mensajes, (list, tuple)):
        return None
    for mensaje in mensajes:
        if isinstance(mensaje, HumanMessage):
            return mensaje.text
    return None


class RegistroIntervenciones:
    """Guarda qué hizo un middleware en cada pregunta, para auditarlo.

    Sin esto, una evaluación solo dice si la pregunta acertó; no si el
    middleware intervino ni qué le dijo al modelo. ``volcar_intervenciones``
    lo escribe junto a los resultados.
    """

    def __init__(self) -> None:
        super().__init__()
        self.intervenciones: list[dict[str, Any]] = []

    def _registrar(self, mensajes: object, tipo: str, **detalle: Any) -> None:
        self.intervenciones.append(
            {
                "middleware": type(self).__name__,
                "pregunta": pregunta_de(mensajes),
                "tipo": tipo,
                **detalle,
            }
        )


def volcar_intervenciones(constructor: object, ruta: str | Path) -> int:
    """Escribe en JSON las intervenciones de los middlewares del constructor.

    Devuelve cuántas hubo. Si la variante no lleva middlewares con registro,
    no escribe nada y devuelve 0. En una evaluación reanudada solo aparecen
    las preguntas ejecutadas en esta tanda.
    """
    registros = [
        middleware
        for middleware in getattr(constructor, "middlewares", ())
        if isinstance(middleware, RegistroIntervenciones)
    ]
    if not registros:
        return 0
    intervenciones = [
        intervencion
        for registro in registros
        for intervencion in registro.intervenciones
    ]
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(intervenciones, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(intervenciones)


def _cuadra(afirmada: float, real: float, tolerancia: float) -> bool:
    """Igual criterio que ``miax_s2.cuadra`` y que el evaluador del proyecto."""
    if real == 0:
        return afirmada == 0
    return abs(afirmada - real) / abs(real) <= tolerancia


# ---------------------------------------------------------------------------
# v004 · Comprobación de cifras contra XBRL
# ---------------------------------------------------------------------------
class ComprobadorCifrasXBRL(RegistroIntervenciones, AgentMiddleware):
    """Devuelve al modelo toda cifra que no sea un hecho XBRL consultado.

    El enunciado dice que acertar una cifra leyéndola del texto cuenta como
    fallo, así que no basta con que la cifra exista en el XBRL: tiene que
    salir de una llamada a ``get_xbrl_fact`` hecha en esta misma ejecución.
    Por eso el middleware distingue cinco desajustes y da un mensaje
    distinto a cada uno:

    1. la cifra cuadra con un hecho consultado, pero de otro ejercicio o de
       otra compañía que la declarada (típico en comparativas, donde hay que
       dar el ejercicio más reciente);
    2. la cifra existe en el XBRL de la compañía y el ejercicio declarados
       pero no se consultó con ``get_xbrl_fact`` (se leyó del texto);
    3. la cifra no cuadra con ningún hecho: se listan los consultados y los
       conceptos disponibles;
    4. no hay ningún hecho XBRL de esa compañía y ejercicio en el corpus, que
       es el caso de las preguntas sin respuesta: se recuerda que entonces la
       respuesta correcta es ``fuente='ninguna'`` con ``cifra=null``;
    5. la cifra es un hecho consultado pero redondeado a entero, porque
       ``get_xbrl_fact`` imprime sin decimales (beneficio por acción): se le
       da el valor exacto.

    Se acepta sin corrección una cifra derivada de hechos consultados (la
    diferencia entre dos ejercicios, su variación porcentual o un margen),
    porque sale igualmente del XBRL.

    Corrige una sola vez por pregunta: si el modelo insiste, se deja pasar su
    respuesta en lugar de gastar iteraciones, que están limitadas.

    Además normaliza la unidad ("dólares", "usd") a la del hecho XBRL cuando
    la cifra ya es correcta; eso no necesita hablar con el modelo.
    """

    def __init__(
        self,
        corpus: CorpusVariant,
        *,
        tolerancia: float = TOLERANCIA_RELATIVA,
        marca: str = MARCA_CIFRAS,
    ) -> None:
        super().__init__()
        if not isinstance(corpus, CorpusVariant):
            raise TypeError("corpus debe ser una instancia de CorpusVariant.")
        if isinstance(tolerancia, bool) or not isinstance(
            tolerancia, (int, float)
        ):
            raise TypeError("tolerancia debe ser un número.")
        if tolerancia < 0:
            raise ValueError("tolerancia no puede ser negativa.")
        if not isinstance(marca, str) or not marca.strip():
            raise ValueError("marca debe ser texto no vacío.")

        self._tolerancia = float(tolerancia)
        self._marca = marca.strip()
        tabla = pd.read_parquet(corpus.ruta_xbrl)
        ausentes = [
            columna
            for columna in _COLUMNAS_XBRL
            if columna not in tabla.columns
        ]
        if ausentes:
            raise ValueError(f"Faltan columnas en el XBRL: {ausentes}.")
        self._hechos: dict[tuple[str, int], tuple[HechoXBRL, ...]] = {}
        for fila in tabla.itertuples(index=False):
            clave = (str(fila.ticker).strip().upper(), int(fila.fiscal_year))
            hecho = HechoXBRL(
                ticker=clave[0],
                ejercicio=clave[1],
                concepto=str(fila.concept),
                valor=float(fila.value),
                unidad=str(fila.unit),
            )
            self._hechos[clave] = self._hechos.get(clave, ()) + (hecho,)
        self._companias = tuple(sorted({clave[0] for clave in self._hechos}))
        self._ejercicios = tuple(sorted({clave[1] for clave in self._hechos}))

    @property
    def parametros(self) -> Mapping[str, JsonValue]:
        return MappingProxyType(
            {
                "tolerancia_relativa": self._tolerancia,
                "exige_get_xbrl_fact": True,
                "acepta_derivadas_de_hechos_consultados": True,
                "normaliza_unidad": True,
                "maximo_correcciones": 1,
                "marca": self._marca,
            }
        )

    @hook_config(can_jump_to=["model"])
    def after_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Comprueba la cifra declarada y, si hace falta, pide corrección."""
        del runtime
        respuesta = state.get("structured_response")
        if not isinstance(respuesta, RespuestaFinanciera):
            return None
        cifra = respuesta.cifra
        if cifra is None:
            return None
        try:
            cifra = float(cifra)
        except (TypeError, ValueError):
            return None

        mensajes = state.get("messages", ())
        consultados = self._hechos_consultados(mensajes)
        ticker = self._normalizar_ticker(respuesta.ticker)
        ejercicio = self._normalizar_ejercicio(respuesta.ejercicio)

        coincidencias = tuple(
            hecho
            for hecho in consultados
            if _cuadra(cifra, hecho.valor, self._tolerancia)
        )
        compatibles = tuple(
            hecho
            for hecho in coincidencias
            if (ticker is None or hecho.ticker == ticker)
            and (ejercicio is None or hecho.ejercicio == ejercicio)
        )
        if compatibles:
            actualizacion = self._normalizar_unidad(respuesta, compatibles)
            if actualizacion is not None:
                self._registrar(
                    mensajes,
                    "unidad_normalizada",
                    antes=respuesta.unidad,
                    despues=actualizacion["structured_response"].unidad,
                )
            return actualizacion
        if self._derivada_de_consultados(cifra, consultados):
            self._registrar(mensajes, "derivada_aceptada", cifra=cifra)
            return None
        if contiene_marca(mensajes, self._marca):
            self._registrar(mensajes, "mantenida_tras_correccion", cifra=cifra)
            return None
        correccion = self._pedir_correccion(
            cifra=cifra,
            ticker=ticker,
            ejercicio=ejercicio,
            consultados=consultados,
            coincidencias=coincidencias,
        )
        self._registrar(
            mensajes,
            "correccion",
            cifra=cifra,
            mensaje=correccion["messages"][0]["content"],
        )
        return correccion

    # -- comprobaciones ----------------------------------------------------
    def _hechos_consultados(self, mensajes: object) -> tuple[HechoXBRL, ...]:
        """Hechos reales detrás de cada ``get_xbrl_fact`` de la traza.

        Se leen los argumentos de la llamada, no su salida en texto: la
        herramienta formatea el valor con ``,.0f`` y redondea, por ejemplo,
        un beneficio por acción de 6,08 a «6».
        """
        consultados: list[HechoXBRL] = []
        for argumentos in llamadas_por_nombre(mensajes, "get_xbrl_fact"):
            ticker = self._normalizar_ticker(argumentos.get("ticker"))
            ejercicio = self._normalizar_ejercicio(argumentos.get("fiscal_year"))
            concepto = argumentos.get("concept")
            if ticker is None or ejercicio is None or not isinstance(concepto, str):
                continue
            for hecho in self._hechos.get((ticker, ejercicio), ()):
                if hecho.concepto == concepto.strip():
                    consultados.append(hecho)
        return tuple(dict.fromkeys(consultados))

    def _derivada_de_consultados(
        self,
        cifra: float,
        consultados: Sequence[HechoXBRL],
    ) -> bool:
        """Acepta diferencias, variaciones porcentuales y márgenes."""
        for primero in consultados:
            for segundo in consultados:
                if primero is segundo:
                    continue
                if primero.ticker != segundo.ticker:
                    continue
                derivadas: list[float] = []
                if primero.concepto == segundo.concepto:
                    derivadas.append(primero.valor - segundo.valor)
                    if segundo.valor:
                        derivadas.append(
                            (primero.valor - segundo.valor) / segundo.valor * 100
                        )
                elif primero.ejercicio == segundo.ejercicio and segundo.valor:
                    derivadas.append(primero.valor / segundo.valor * 100)
                if any(
                    _cuadra(cifra, derivada, self._tolerancia)
                    for derivada in derivadas
                ):
                    return True
        return False

    def _normalizar_unidad(
        self,
        respuesta: RespuestaFinanciera,
        compatibles: Sequence[HechoXBRL],
    ) -> dict[str, Any] | None:
        unidades = {hecho.unidad for hecho in compatibles}
        if len(unidades) != 1:
            return None
        unidad = next(iter(unidades))
        declarada = (respuesta.unidad or "").strip()
        if declarada.casefold() == unidad.casefold():
            return None
        return {
            "structured_response": respuesta.model_copy(
                update={"unidad": unidad}
            )
        }

    # -- mensajes de corrección -------------------------------------------
    def _pedir_correccion(
        self,
        *,
        cifra: float,
        ticker: str | None,
        ejercicio: int | None,
        consultados: Sequence[HechoXBRL],
        coincidencias: Sequence[HechoXBRL],
    ) -> dict[str, Any]:
        identidad = self._describir_identidad(ticker, ejercicio)
        redondeado = self._redondeo_de_consultado(
            cifra, ticker, ejercicio, consultados
        )
        if redondeado is not None:
            detalle = (
                f"esa cifra es {redondeado.concepto} redondeado: la salida de "
                "get_xbrl_fact muestra los valores sin decimales, pero el "
                f"hecho XBRL exacto es {redondeado.valor:,.15g} "
                f"{redondeado.unidad}. Responde con "
                f"cifra={redondeado.valor:.15g} y "
                f"unidad={redondeado.unidad!r}."
            )
        elif coincidencias:
            detalle = (
                f"esa cifra es {coincidencias[0]}, que no es {identidad}. "
                "Si la pregunta compara dos ejercicios, la cifra tiene que "
                "ser la del ejercicio MÁS RECIENTE y los campos ticker y "
                "ejercicio deben corresponder a esa misma cifra."
            )
        elif ticker is not None and ejercicio is not None and (
            (ticker, ejercicio) not in self._hechos
        ):
            detalle = (
                f"no hay ningún hecho XBRL de {identidad} en el corpus "
                f"(compañías: {', '.join(self._companias)}; ejercicios: "
                f"{', '.join(str(anio) for anio in self._ejercicios)}). "
                "Si el dato no está en el corpus, responde fuente='ninguna' "
                "y cifra=null en lugar de estimarlo."
            )
        else:
            sin_consultar = self._hechos_que_cuadran_sin_consultar(
                cifra, ticker, ejercicio, consultados
            )
            if sin_consultar is not None:
                detalle = (
                    f"esa cifra coincide con {sin_consultar}, pero no la has "
                    "pedido con get_xbrl_fact en esta respuesta. Llama a "
                    f"get_xbrl_fact(ticker={sin_consultar.ticker!r}, "
                    f"fiscal_year={sin_consultar.ejercicio}, "
                    f"concept={sin_consultar.concepto!r}) y responde con el "
                    "valor que devuelva: una cifra leída del texto no vale."
                )
            else:
                detalle = (
                    "no coincide con ningún hecho XBRL que hayas consultado. "
                    f"{self._describir_consultados(consultados)} "
                    f"{self._describir_disponibles(ticker, ejercicio)} "
                    "Corrige la cifra con get_xbrl_fact; si es un porcentaje "
                    "o una variación que has calculado a partir de hechos "
                    "XBRL, mantenla y explica el cálculo en la prosa; si el "
                    "dato no está en el corpus, responde fuente='ninguna' y "
                    "cifra=null."
                )
        mensaje = (
            f"{self._marca}: has respondido cifra={cifra:,.15g} para "
            f"{identidad} y {detalle}"
        )
        return {
            "messages": [{"role": "user", "content": mensaje}],
            "jump_to": "model",
        }

    @staticmethod
    def _redondeo_de_consultado(
        cifra: float,
        ticker: str | None,
        ejercicio: int | None,
        consultados: Sequence[HechoXBRL],
    ) -> HechoXBRL | None:
        """Hecho con decimales que el modelo ha copiado redondeado.

        ``get_xbrl_fact`` imprime el valor con ``,.0f``: un beneficio por
        acción de 7,46 le llega al modelo como «7».
        """
        for hecho in consultados:
            if ticker is not None and hecho.ticker != ticker:
                continue
            if ejercicio is not None and hecho.ejercicio != ejercicio:
                continue
            if not hecho.valor.is_integer() and cifra == round(hecho.valor):
                return hecho
        return None

    def _hechos_que_cuadran_sin_consultar(
        self,
        cifra: float,
        ticker: str | None,
        ejercicio: int | None,
        consultados: Sequence[HechoXBRL],
    ) -> HechoXBRL | None:
        if ticker is None or ejercicio is None:
            return None
        for hecho in self._hechos.get((ticker, ejercicio), ()):
            if hecho in consultados:
                continue
            if _cuadra(cifra, hecho.valor, self._tolerancia):
                return hecho
        return None

    @staticmethod
    def _describir_identidad(ticker: str | None, ejercicio: int | None) -> str:
        if ticker is None and ejercicio is None:
            return "una compañía y un ejercicio que no has declarado"
        if ticker is None:
            return f"FY{ejercicio} sin declarar la compañía"
        if ejercicio is None:
            return f"{ticker} sin declarar el ejercicio"
        return f"{ticker} FY{ejercicio}"

    @staticmethod
    def _describir_consultados(consultados: Sequence[HechoXBRL]) -> str:
        if not consultados:
            return "No has consultado ningún hecho XBRL en esta respuesta."
        listado = "; ".join(
            str(hecho) for hecho in consultados[:_MAXIMO_HECHOS_LISTADOS]
        )
        return f"Has consultado: {listado}."

    def _describir_disponibles(
        self,
        ticker: str | None,
        ejercicio: int | None,
    ) -> str:
        if ticker is None or ejercicio is None:
            return ""
        hechos = self._hechos.get((ticker, ejercicio), ())
        if not hechos:
            return ""
        conceptos = ", ".join(
            hecho.concepto for hecho in hechos[:_MAXIMO_HECHOS_LISTADOS]
        )
        return f"Conceptos de {ticker} FY{ejercicio} en el XBRL: {conceptos}."

    @staticmethod
    def _normalizar_ticker(valor: object) -> str | None:
        if not isinstance(valor, str):
            return None
        normalizado = valor.strip().upper()
        return normalizado or None

    @staticmethod
    def _normalizar_ejercicio(valor: object) -> int | None:
        if isinstance(valor, bool):
            return None
        if isinstance(valor, int):
            return valor
        if isinstance(valor, float) and valor.is_integer():
            return int(valor)
        if isinstance(valor, str):
            texto = valor.strip().removeprefix("FY").removeprefix("fy")
            try:
                return int(texto)
            except ValueError:
                return None
        return None


# ---------------------------------------------------------------------------
# v005 · Reparación determinista de citas
# ---------------------------------------------------------------------------
# El evaluador comprueba la cita como subcadena literal (normalizando espacios
# y mayúsculas) del fragmento citado. Estas equivalencias son las diferencias
# que el modelo introduce al copiar y que no cambian el texto: apóstrofos y
# comillas tipográficos, guiones largos y espacios duros. El 25 % de los
# fragmentos del corpus lleva apóstrofo tipográfico (’).
_EQUIVALENTES = {
    "’": "'",
    "‘": "'",
    "ʼ": "'",
    "´": "'",
    "`": "'",
    "′": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "″": '"',
    "«": '"',
    "»": '"',
    "–": "-",
    "—": "-",
    "−": "-",
    "‐": "-",
    "‑": "-",
    "…": "...",
}
_COMILLAS_EXTERIORES = "\"'«»“”‘’ \t "
_SEPARADOR_ELIPSIS = re.compile(r"[\[(]?\s*(?:\.{2,}|…)\s*[\])]?")
_ELIPSIS_EXTERIOR = re.compile(r"^\s*\.\.\.\s*|\s*\.\.\.\s*$")
_HUECO_MAXIMO = 400
_MINIMO_CARACTERES_CITA = 20


def _esqueleto(texto: str) -> tuple[str, tuple[int, ...]]:
    """Texto sin espacios ni variantes tipográficas, con sus posiciones."""
    caracteres: list[str] = []
    posiciones: list[int] = []
    for posicion, caracter in enumerate(texto):
        if caracter.isspace():
            continue
        for canonico in _EQUIVALENTES.get(caracter, caracter).casefold():
            caracteres.append(canonico)
            posiciones.append(posicion)
    return "".join(caracteres), tuple(posiciones)


def buscar_literal(
    cita: str,
    texto: str,
    *,
    hueco_maximo: int = _HUECO_MAXIMO,
    minimo_caracteres: int = _MINIMO_CARACTERES_CITA,
) -> str | None:
    """Devuelve el trozo literal de ``texto`` que la cita reproduce.

    Tolera espacios, mayúsculas, apóstrofos y comillas tipográficos y elipsis
    intermedias; no tolera ninguna palabra distinta. Si la cita no está en el
    texto, devuelve ``None``: este reparador nunca sustituye una cita por otra
    frase parecida.
    """
    segmentos = [
        esqueleto
        for parte in _SEPARADOR_ELIPSIS.split(cita)
        if len(esqueleto := _esqueleto(parte)[0]) >= 4
    ]
    if sum(len(segmento) for segmento in segmentos) < minimo_caracteres:
        return None
    esqueleto_texto, posiciones = _esqueleto(texto)
    inicio: int | None = None
    fin = 0
    for segmento in segmentos:
        encontrado = esqueleto_texto.find(segmento, fin)
        if encontrado < 0:
            return None
        if inicio is None:
            inicio = encontrado
        elif encontrado - fin > hueco_maximo:
            return None
        fin = encontrado + len(segmento)
    if inicio is None:
        return None
    return texto[posiciones[inicio] : posiciones[fin - 1] + 1]


class ReparadorCitas(RegistroIntervenciones, AgentMiddleware):
    """Sustituye la cita por el texto literal del fragmento recuperado.

    No llama al modelo ni gasta iteraciones. Solo actúa cuando la cita no
    pasa la comprobación del evaluador y la misma frase aparece, carácter a
    carácter salvo tipografía y espacios, en un fragmento que el agente citó
    o recuperó con ``search_filings`` en esta ejecución. Si la cita está
    traducida, resumida o inventada, no hace nada: prefiere que la respuesta
    falle a colocar una frase que el modelo no ha usado.

    Casos que arregla, todos vistos en las evaluaciones con Sonnet:
    - apóstrofo recto (') donde el informe lleva el tipográfico (’);
    - comillas añadidas alrededor de la frase;
    - elipsis («…») en mitad de la frase citada;
    - chunk_id equivocado cuando la frase está en otro fragmento recuperado.
    """

    def __init__(self, corpus: CorpusVariant) -> None:
        super().__init__()
        if not isinstance(corpus, CorpusVariant):
            raise TypeError("corpus debe ser una instancia de CorpusVariant.")
        self._chunks = cargar_fragmentos(corpus.ruta_chunks)

    @property
    def parametros(self) -> Mapping[str, JsonValue]:
        return MappingProxyType(
            {
                "solo_fragmentos_recuperados_o_citados": True,
                "tolera": "espacios, mayúsculas, tipografía y elipsis",
                "inventa_citas": False,
                "hueco_maximo_caracteres": _HUECO_MAXIMO,
                "minimo_caracteres_cita": _MINIMO_CARACTERES_CITA,
            }
        )

    def after_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Repara el par cita/chunk_id sin cambiar nada más de la respuesta."""
        del runtime
        respuesta = state.get("structured_response")
        if not isinstance(respuesta, RespuestaFinanciera):
            return None
        cita = respuesta.cita
        if not isinstance(cita, str) or not cita.strip():
            return None

        candidatos = self._candidatos(respuesta.chunk_id, state.get("messages", ()))
        if not candidatos:
            return None
        if self._ya_es_valida(cita, respuesta.chunk_id):
            return None

        for variante in self._variantes(cita):
            # Primero el texto literal del fragmento: así la cita queda tal
            # cual está en el informe, no como la escribió el modelo.
            for chunk_id in candidatos:
                texto = self._chunks[chunk_id].texto
                literal = buscar_literal(variante, texto)
                if literal is not None and cita_esta_respaldada(literal, texto):
                    return self._actualizar(state, respuesta, literal, chunk_id)
            for chunk_id in candidatos:
                texto = self._chunks[chunk_id].texto
                if cita_esta_respaldada(variante, texto):
                    return self._actualizar(state, respuesta, variante, chunk_id)
        self._registrar(
            state.get("messages", ()),
            "cita_no_reparable",
            cita=cita,
            chunk_id=respuesta.chunk_id,
        )
        return None

    def _candidatos(self, chunk_id: str | None, mensajes: object) -> tuple[str, ...]:
        """El fragmento citado primero; luego los recuperados, por ranking."""
        candidatos: list[str] = []
        if isinstance(chunk_id, str) and chunk_id in self._chunks:
            candidatos.append(chunk_id)
        candidatos.extend(
            recuperado
            for recuperado in chunks_recuperados(mensajes)
            if recuperado in self._chunks and recuperado not in candidatos
        )
        return tuple(candidatos)

    def _ya_es_valida(self, cita: str, chunk_id: str | None) -> bool:
        fragmento = self._chunks.get(chunk_id) if isinstance(chunk_id, str) else None
        return fragmento is not None and cita_esta_respaldada(cita, fragmento.texto)

    @staticmethod
    def _variantes(cita: str) -> tuple[str, ...]:
        limpia = cita.strip().strip(_COMILLAS_EXTERIORES).strip()
        limpia = _SEPARADOR_ELIPSIS.sub(" ... ", limpia)
        limpia = _ELIPSIS_EXTERIOR.sub("", limpia).strip()
        variantes = tuple(
            dict.fromkeys(texto for texto in (cita.strip(), limpia) if texto)
        )
        return variantes

    def _actualizar(
        self,
        state: AgentState,
        respuesta: RespuestaFinanciera,
        cita: str,
        chunk_id: str,
    ) -> dict[str, Any] | None:
        if cita == respuesta.cita and chunk_id == respuesta.chunk_id:
            return None
        self._registrar(
            state.get("messages", ()),
            "cita_reparada",
            antes=respuesta.cita,
            despues=cita,
            chunk_antes=respuesta.chunk_id,
            chunk_despues=chunk_id,
        )
        return {
            "structured_response": respuesta.model_copy(
                update={"cita": cita, "chunk_id": chunk_id}
            )
        }


__all__ = [
    "ComprobadorCifrasXBRL",
    "HechoXBRL",
    "MARCA_CIFRAS",
    "ReparadorCitas",
    "RegistroIntervenciones",
    "TOLERANCIA_RELATIVA",
    "buscar_literal",
    "chunks_recuperados",
    "contiene_marca",
    "llamadas_por_nombre",
    "pregunta_de",
    "volcar_intervenciones",
]
