# Agente financiero · jchulvi

## v2 · resultado final

[agente_v2.py](agente_v2.py) define su prompt y una única función
`crear_constructor()`, sin hooks ni validadores propios.
Las instrucciones se han reformulado manteniendo las reglas de cifras, unidades,
citas y comparación entre ejercicios.

Se mantienen **DeepSeek V4 Flash 0731/DeepInfra**, **Voyage 4 Lite**, los **1.749
embeddings cacheados del corpus**, Qdrant, los filtros y el contexto por sección.
El límite nativo es de **6 llamadas** al modelo; no se cachean las consultas.
Cuando se usa mediante la fachada `agente/`, Qdrant se declara como recurso del
`ConstructorAgente` y su sesión se gestiona automáticamente durante la operación.

[validacion_agente_v2.ipynb](validacion_agente_v2.ipynb) está ejecutado y guardado:
**40 respuestas nuevas**, sin reutilizar respuestas de campañas anteriores.

| Conjunto | Aciertos |
| --- | ---: |
| Oficial | **20/20** |
| Equipo | **20/20** |

Los tres fallos históricos (`of-001`, `of-004`, `of-005`) pasan en esta ejecución.
**28/28 citas completas** verificadas, cero errores de ejecución y 7/7 celdas de
código ejecutadas. Coste del agente: **0.0473 USD**, sin embeddings de consulta.
**210 tests y 14 subtests** correctos. Qdrant quedó detenido y conserva sus datos.

Desde la raíz del proyecto, con el entorno, Docker y la clave preparados:

```bash
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 experimentos/jchulvi/validacion_agente_v2.ipynb
```

Campaña: `resultados/estudio_v002/v2_simple_6944a9ccd5b5f92e/`.
Las fuentes y los datos determinan su identidad; los checkpoints permiten
reanudar esa misma configuración. Una ejecución nueva tiene coste cloud.

Los conjuntos son conocidos de desarrollo, no un hold-out. El evaluador común
comprueba cifras/unidades, herramientas y el prefijo de la cita. El notebook
muestra además la literalidad de la cita completa, normalizando espacios y
mayúsculas, sin alterar respuestas ni puntuaciones.

## Referencia v1 conservada

El código y el notebook anteriores siguen disponibles para reproducir la referencia.
La configuración y los resultados que siguen describen **v1**, no la nueva v2.

[agente.py](agente.py) responde preguntas sobre los informes 10-K del corpus.
Consulta herramientas, prioriza XBRL para las cifras y devuelve una
`RespuestaFinanciera` con respuesta, procedencia y evidencia.

## Modelos y funcionamiento

- **Agente y reescritura de consultas:** `deepseek/deepseek-v4-flash-0731`,
  mediante OpenRouter y el proveedor DeepInfra.
- **Embeddings:** `voyageai/voyage-4-lite` mediante OpenRouter, con 1.024
  dimensiones y normalización L2. Solo los vectores del corpus se guardan en [embeddings/](embeddings/).
  Las búsquedas densas e híbridas generan su vector en la nube, sin caché en memoria ni en disco.
- **Evaluación:** comprobaciones locales del framework, protocolo 5. No hay
  modelo juez ni inferencia con modelos locales.

El agente usa el motor compartido LangChain/LangGraph y nuestro recuperador.
Tiene un máximo de **18 llamadas al modelo por pregunta**; la última se reserva
para entregar la respuesta estructurada. Valida campos, cifras XBRL consultadas
y citas literales recuperadas. Si no logra una salida válida, conserva el error.
Las instrucciones permiten cifras textuales cuando no existen en XBRL, indicando su procedencia.
No utiliza un parser de números en prosa.

## Herramientas

Reutilizamos los contratos e implementaciones comunes; `search_filings` conecta
nuestro recuperador y personalizamos las descripciones de búsqueda y XBRL.

| Herramienta | Uso |
| --- | --- |
| `list_available` | Consultar empresas, ejercicios y secciones disponibles. |
| `get_xbrl_fact` | Obtener una magnitud financiera exacta con su unidad. |
| `search_filings` | Buscar evidencia con filtros de empresa, ejercicio y sección. |
| `read_section` | Leer una sección completa cuando falta contexto. |

La búsqueda puede ser BM25, densa o híbrida. El notebook selecciona el modo por
recall@5 y después evalúa al agente, que decide sus propias consultas y filtros.
Los resultados recuperados se amplían a contexto de sección para responder.
Para reproducir la nota, usa el notebook: `crear_constructor()` sin argumentos
conserva BM25 como valor por defecto.

## Qdrant: configuración local

Qdrant es la **base de datos vectorial**, no un modelo local. Los embeddings
del corpus se calculan en la nube y se almacenan en disco para no volver a pagarlos.

- [qdrant_local.py](qdrant_local.py) ofrece `iniciar()`, `cargar()`, `consultar()`
  y `detener()`; el notebook usa `with qdrant.sesion():` para detenerlo al salir.
- Las sesiones anidadas dentro del mismo proceso comparten el contenedor; solo
  la última en cerrarse lo detiene. Un contenedor externo sigue rechazándose.
- Docker publica Qdrant solo en `http://127.0.0.1:6339`, con 2 CPU, 2 GB de RAM
  e imagen fijada por digest en el helper.
- El índice contiene **1.749 vectores de 1.024 dimensiones**, uno por fragmento.
  Usa producto escalar sobre vectores normalizados y búsqueda exacta, con filtros
  `ticker`, `fiscal_year` e `item`.
- `cargar()` comprueba el modelo, los hashes y las dimensiones. Reutiliza la
  colección existente o carga los vectores con identificadores estables.
- Los archivos compartibles son `vectores.npy` y `manifest.json` en `embeddings/`; los datos del contenedor,
  en `resultados/estudio_v002/qdrant_storage/`. Detenerlo no borra ninguno.

## Entorno

Desde la raíz de esta copia del proyecto, con Docker en marcha:

```bash
conda activate /Users/jchulvi/projects/Taller_NLP/.venv
set -a
source .env
set +a
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 experimentos/jchulvi/evaluacion_agente.ipynb
```

En Jupyter: seleccionar ese entorno, reiniciar el kernel y pulsar **Run All**.
La clave `OPENROUTER_API_KEY` se carga fuera del notebook; `.env` está ignorado
por Git. Los modelos se configuran en `agente.py`, no mediante variables de entorno.

[evaluacion_agente.ipynb](evaluacion_agente.ipynb) ejecuta siempre:

1. Comprobaciones de corpus, herramientas y validaciones.
2. Preparación/reutilización del índice y estudio de las seis variantes de búsqueda.
3. Las **20 preguntas del equipo y las 20 oficiales**, con todas las respuestas,
   citas, trazas y resultados visibles y guardados dentro del notebook.

Reanuda checkpoints compatibles; los casos pendientes no cuentan como fallos.
Las llamadas nuevas a modelos y embeddings tienen coste.

## Resultados de la ejecución final

Ejecución del **20-09-2026**, protocolo 5: **40/40 evaluadas, 0 pendientes**.
Notebook guardado con sus 15 celdas de código ejecutadas y sin errores de celda.
Ejecución realizada sin caché de embeddings de consulta; se reutilizó el índice del corpus.

| Conjunto | Aciertos |
| --- | ---: |
| Equipo | 20/20 |
| Oficial | 17/20 |
| **Total** | **37/40 · 92,5%** |

Fallaron `of-001`, `of-004` y `of-005`: el agente terminó sin
respuesta estructurada. Se conservan los fallos y sus trazas, sin repetirlos.
Coste registrado de las 40 respuestas: **0,0877 USD**, sin reescrituras ni embeddings.

Estudio de búsqueda completo: **denso con metadatos, 74,1% de recall@5**,
seleccionado frente a BM25 (29,6%) e híbrido (63,0%). La variante adicional
reescritura + denso obtuvo 81,5%; no es la misma configuración que usa el agente.
Se midieron las seis variantes sobre las 27 preguntas con ancla textual.

Qdrant reutilizó los 1.749 vectores y quedó **detenido, con los datos conservados**.
Una reescritura vacía interrumpió el primer arranque; se reanudó desde los checkpoints.
Artefactos locales: `resultados/estudio_v002/campana_20f07e842e1626f8/` y
`resultados/estudio_v002/retrieval_s2_860c5c920f627a6e/`. El notebook conserva los resultados visibles.
Verificación del proyecto: **178 tests y 12 subtests correctos**.

El evaluador comprueba cifras/unidades, herramientas esperadas y citas cuando
corresponde. En las citas busca el prefijo normalizado de 120 caracteres en el
fragmento: **no verifica semánticamente toda la respuesta**. Estos conjuntos se
usan durante el desarrollo; no son una evaluación ciega ni prueban generalización.

## Resultados históricos

[Historial anterior](resultados/estudio_v002/archivo_notebooks/README_antes_resumen_final_20260920.md).
Se conserva separado porque cambian el código, los conjuntos y el protocolo de evaluación.
