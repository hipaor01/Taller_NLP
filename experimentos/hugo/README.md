# Variantes de Hugo

Todas las versiones usan **Claude Sonnet 5** (API de Anthropic) con la
misma configuración, fijada en `comun.py`. Hasta el 2026-09-20 se usó
Claude Haiku 4.5; esos resultados se conservan con `haiku` en el nombre. Como el baseline del grupo está
medido con Gemini, la referencia de esta carpeta es `agente_v000`: el
baseline idéntico con Claude. Cada versión cambia una sola cosa respecto a
la anterior y se mide sobre el golden set propio y sobre el oficial.

| Versión | Cambio | Estado |
| --- | --- | --- |
| `agente_v000` | Baseline del grupo con el modelo de Hugo (referencia) | Medida (Sonnet 5): 6/20 |
| `agente_v001` | Filtro de metadatos (ticker, ejercicio, item) en `search_filings` | Medida (Sonnet 5): 6/20 |
| `agente_v002` | Formato de la respuesta: cifra completa en USD, último ejercicio en comparativas, cita literal con ’ | Medida (Sonnet 5): 14/20 |
| `agente_v003` | Protocolo de comparativas: XBRL de los dos ejercicios + búsqueda y cita en el Item 7 | Medida (Sonnet 5): 20/20 |
| `agente_v004` | Middleware propio que contrasta la cifra con el XBRL (obligatorio del enunciado) | Medida (Sonnet 5): 20/20 |
| `agente_v005` | Reparación determinista de la cita (tipografía, comillas, elipsis, chunk_id) | Medida (Sonnet 5): 20/20 |

Retrieval (medido aislado en `medir_recall.py`, sin agente): denso plano,
filtro de metadatos, híbrido BM25 + denso (RRF, `retrievers.py`) y
reescritura de la consulta a inglés con el LLM (cacheada en
`resultados/reescrituras_<modelo>.json`).

v003, v004 y v005 necesitan el repositorio en `main` a partir de `8bd78b6`
(protocolo de citas v5 y `taller_nlp.citas`). Probado sobre `1b2c8ae`
(21-sep): 210 tests en verde.

Para ejecutar las variantes hace falta `langchain-anthropic` y la clave de
Anthropic (ver «Requisitos»). Los tests de esta carpeta no los necesitan:
pasan en un clon limpio sin ese paquete.

## Guardarraíles propios (`guardrails.py`)

Dos middlewares, cada uno con su versión, enchufados con el parámetro
`middlewares` de `ConstructorAgente`. No tocan código común.

**`ComprobadorCifrasXBRL` (v004).** Contrasta la cifra de la respuesta
estructurada con los hechos XBRL y devuelve al modelo el desajuste concreto,
una sola vez por pregunta. Cinco casos distintos:

1. la cifra cuadra con un hecho consultado, pero de otro ejercicio o de otra
   compañía que la declarada (el fallo típico de las comparativas);
2. la cifra está en el XBRL de esa compañía y ejercicio pero no se pidió con
   `get_xbrl_fact`: se leyó del texto, que el enunciado cuenta como fallo;
3. la cifra no cuadra con nada: se listan los hechos consultados y los
   conceptos disponibles;
4. no hay ningún hecho de esa compañía y ejercicio en el corpus: se recuerda
   que la respuesta correcta es `fuente="ninguna"` con `cifra=null`, que es
   el caso de las preguntas ciegas sin respuesta;
5. la cifra es un hecho consultado pero redondeado a entero (ver abajo): se
   le da el valor exacto. Añadido el 21-sep, después de medir v004 y v005 en
   el golden propio; en esas pasadas no hay ninguna pregunta de beneficio por
   acción, así que el resultado medido no cambia.

Acepta sin corregir las cifras derivadas de hechos consultados (diferencia,
variación porcentual, margen) y normaliza la unidad («dólares» → `USD`) sin
gastar una llamada al modelo.

Lee el valor del parquet a partir de los **argumentos** de la llamada, no de
su salida en texto: `get_xbrl_fact` imprime el valor con `,.0f`, así que el
beneficio por acción de AAPL FY2024 (6,08 USD/shares) le llega al modelo como
«6». Si una pregunta ciega pide un BPA, la respuesta leída de la herramienta
falla por el redondeo; el middleware lo detecta y devuelve el valor exacto.
(Propuesto a Higinio como arreglo del código común.)

**`ReparadorCitas` (v005).** No habla con el modelo. Si la cita no está
literal en el fragmento citado, busca la misma frase —tolerando espacios,
mayúsculas, apóstrofos y comillas tipográficos y elipsis— en el fragmento
citado y en los que devolvió `search_filings` en esa ejecución, y la
sustituye por el texto literal del corpus, corrigiendo también el `chunk_id`
si la frase estaba en otro fragmento recuperado. Si la cita está traducida,
resumida o inventada, no hace nada: nunca coloca una frase que el modelo no
haya usado.

Los dos middlewares registran cada intervención (qué pregunta, qué tipo,
qué le dijeron al modelo o qué cita cambiaron). `evaluar.py` y
`evaluar_holdout.py` lo guardan en `<etiqueta>_intervenciones.json`, para
poder enseñar en la presentación qué hizo el middleware y no solo el acierto
final.

Tests: `tests/test_hugo_guardrails.py` (unitarios) y
`tests/test_hugo_guardrails_motor.py` (integración con el motor real y un
modelo guionizado: comprueba que la corrección llega al modelo y que la cita
reparada acaba en la respuesta final).

## Casos de estrés (`estres.jsonl`)

Los dos golden sets ya no distinguen v003, v004 y v005 (20/20). Estos 8
casos prueban justo lo que los middlewares tienen que resolver y que las
preguntas ciegas pueden traer:

| id | Qué prueba | Resultado correcto |
| --- | --- | --- |
| est-01, 02, 03 | Beneficio por acción (AAPL, GOOGL, NVDA): `get_xbrl_fact` lo imprime redondeado | la cifra exacta (7,46 · 8,13 · 2,94 USD/shares) |
| est-04 | Compañía fuera del corpus (Tesla) | `fuente="ninguna"`, sin cifra |
| est-05 | Ejercicio fuera del corpus (MSFT FY2021) | `fuente="ninguna"`, sin cifra |
| est-06 | Concepto que la compañía no reporta (GrossProfit de Amazon) | `fuente="ninguna"`, sin cifra |
| est-07, 08 | Extractivas cuyo ancla lleva apóstrofo tipográfico (’) | cita literal |

Las preguntas están redactadas como las del golden, sin pistas para el
agente. Las tres sin respuesta siguen el formato que admite el evaluador común
desde `c6393b4` (numérica con `concept_xbrl` y sin `cifra_esperada` ni
`unidad`), así que el fichero se puede evaluar tanto con el plan B como con
`evaluar.py` o `python -m agente --evaluar`. Las pasadas del 21-sep se
hicieron con el plan B; puntuadas de nuevo con el evaluador común dan lo
mismo (5/8 y 8/8).

```bash
python -m experimentos.hugo.evaluar_holdout experimentos/hugo/estres.jsonl \
  --variante experimentos.hugo.agente_v003 \
  --salida experimentos/hugo/resultados/estres_v003.csv
```

Resultado (Sonnet 5, 21-sep, una pasada):

| Casos | v003 (sin middlewares) | v005 (con los dos) |
| --- | --- | --- |
| Beneficio por acción (3) | 0/3: responde 7, 8 y 3 | **3/3**: 7,46 · 8,13 · 2,94 |
| Sin respuesta en el corpus (3) | 3/3 | 3/3 |
| Extractivas con ’ (2) | 2/2 | 2/2 |
| **Total** | **5/8** | **8/8** |
| Coste de la pasada | 0,19 $ | 0,28 $ |

- Los 3 aciertos que gana v005 son del comprobador de cifras: las tres
  intervenciones registradas son el caso «redondeado» (por ejemplo, «has
  respondido cifra=7 para AAPL FY2025 y esa cifra es EarningsPerShareDiluted
  redondeado [...] el hecho XBRL exacto es 7.46 USD/shares»). En los tres
  casos el modelo corrigió a la primera.
- Las preguntas sin respuesta las resuelve el propio modelo: consulta
  `list_available` o `get_xbrl_fact` y responde `fuente="ninguna"` sin
  cifra. El middleware no tuvo que intervenir.
- El reparador de citas no actuó: Sonnet copió bien el ’ en las dos
  extractivas. En todas las pasadas medidas no ha tenido nada que reparar.
- est-07 en v005 costó 8,8 ¢ y 30 s (3,4 veces más que en v003) con una
  sola búsqueda y sin ninguna intervención registrada: el gasto está en las
  llamadas al modelo, no en los middlewares. No se puede ver más sin la traza
  completa.

## Plan B para las preguntas ciegas (`evaluar_holdout.py`)

Desde `c6393b4` el cargador común acepta las numéricas sin respuesta si
vienen con `concept_xbrl` y sin `cifra_esperada` ni `unidad`. Si el JSONL del
profesor las escribe de otra forma (sin concepto, o una extractiva sin
respuesta), `python -m agente --evaluar` sigue rechazando el fichero entero.
Además puede morir con `segmentation fault` si el agente lanza dos búsquedas
a la vez. Este script evalúa con cualquier variante sin modificar nada común:

```bash
python -m experimentos.hugo.evaluar_holdout holdout.jsonl \
  --variante experimentos.higinio.agente_v009 \
  --salida resultados/holdout_v009.csv
```

- Cada pregunta se clasifica y se evalúa por separado (`holdout.py`):
  `golden` (evaluador de siempre), `sin_respuesta` (acierta si responde
  `fuente="ninguna"` sin cifra) o `no_evaluable` (se ejecuta, no puntúa).
- Solo en ese proceso serializa la codificación de `RetrieverFaiss` (evita el
  segfault); no se edita ningún fichero común.
- Progreso reanudable: repetir el comando sigue donde se quedó.
- Tests: `tests/test_hugo_holdout.py` (7), con la convención de nombres de
  `tests/test_higinio_*.py`.

## Configuración común (`comun.py`)

- Modelo: `anthropic:claude-sonnet-5`, tarifa 2 $ / 10 $ por millón de
  tokens (entrada / salida). Anthropic no devuelve el coste, así que el motor
  lo estima con la tarifa. Haiku 4.5: 1 $ / 5 $.
- `max_tokens_salida = 4096` en todas las versiones.
- Resto (prompt, límites, 18 peticiones por minuto, tolerancia 1 %, k=5):
  igual que `experimentos/baseline.py`.
- Otro modelo sin tocar código: `export HUGO_MODELO=...` (debe estar en
  `PRECIOS`).
- Sonnet 5 rechaza `temperature` (400 «temperature is deprecated for this
  model»). Para los modelos de `MODELOS_SIN_TEMPERATURA` el modelo se
  construye sin ese parámetro; el resto sigue con temperatura 0.

## Requisitos

```bash
pip install langchain-anthropic==1.7.1   # compatible con langchain-core 1.6.1
export ANTHROPIC_API_KEY=...             # en el entorno, nunca en un fichero
```

## Cómo se ejecuta

```bash
# Recall@k del retrieval aislado (sin LLM, sin coste)
python -m experimentos.hugo.medir_recall

# Evaluación completa de una versión
python -m agente --variante experimentos.hugo.agente_v000 \
  --evaluar golden_set.jsonl \
  --salida experimentos/hugo/resultados/v000_propio.csv

# Evaluación con informe completo y diagnóstico de fallos (recomendado)
python -m experimentos.hugo.evaluar agente_v001 golden_set.jsonl v001_propio
python -m experimentos.hugo.evaluar agente_v001 golden_set.jsonl v001_diag --ids gjhh-007 gjhh-010

# Fila resumen para la tabla del informe
python -m agente --resumir experimentos/hugo/resultados/v000_propio.csv \
  --etiqueta v000
```

`experimentos/hugo/resultados/` no se versiona (ver `.gitignore`).

`evaluar.py` guarda el progreso pregunta a pregunta en
`resultados/progreso/<etiqueta>.json`: si se corta, repetir el mismo comando
continúa donde se quedó sin volver a pagar. `--reiniciar` empieza de cero.

**Segfault del 2026-09-20 (golden oficial).** Causa, vista con
`faulthandler`: el modelo pidió dos `search_filings` en el mismo turno,
LangGraph las ejecutó en paralelo y dos hilos codificaron a la vez con el
mismo modelo de embeddings. `RetrieverFaissSeguro` (en `retrievers.py`)
serializa la codificación con un cerrojo; el ranking es idéntico. Todas las
versiones lo usan. Afecta también al baseline del grupo con cualquier modelo
que haga búsquedas en paralelo. Las variables de entorno de `__init__.py`
(OpenMP) quedan como precaución, pero no eran la causa.

## Criterio de recall offline

`medir_recall.py` usa `miax_s2.acierta`, la primitiva del profesor: un
fragmento acierta si contiene el ancla entera y es del mismo ticker y
ejercicio. La consulta es la pregunta en español tal cual y los filtros salen
del golden, como en el notebook S2.

Resultado (BGE-small, k=5; reescritura con Sonnet 5, 2026-09-20):

| Configuración | Oficial (13 con ancla) | Propio (14 con ancla) | Coste |
| --- | --- | --- | --- |
| 1 · Denso plano (baseline) | 30,8 % | 0 % | 0 |
| 2 · + filtro de metadatos | 46,2 % | 50,0 % | 0 |
| 3 · + híbrido BM25 (RRF) | 53,8 % | 42,9 % | 0, +1 índice en memoria |
| 4 · reescritura a inglés + filtro | 76,9 % | 71,4 % | 1 llamada LLM por búsqueda |
| 5 · reescritura + híbrido | **84,6 %** | **92,9 %** | 1 llamada LLM + índice léxico |

- Reproduce las cifras de clase en 1 y 2 (30 % → 46 %). Con la reescritura
  de Sonnet 5 se supera el 69 % que dio el profesor con Gemini.
- El híbrido con preguntas en español empeora el golden propio (50 → 42,9 %)
  y después de reescribir a inglés es lo que más suma (71,4 → 92,9 %): el
  orden en que se prueban los arreglos cambia la conclusión, como se vio en
  clase.
- Es recall del retrieval aislado con la pregunta tal cual. El agente ya
  escribe sus consultas en inglés, así que dentro del agente la reescritura
  explícita puede aportar menos: hay que medirlo con el agente.

## Resultados en el golden propio con Sonnet 5 (2026-09-20)

| Versión | Extractivas | Numéricas | Comparativas | Total | Cita | Cifra | Recall@5 agente | Coste medio | Latencia | Llamadas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline grupo (Gemini 3.8 Flash) | 5/6 | 6/6 | 0/8 | 11/20 | 35,7 % | 78,6 % | 50,0 % | **0,96 ¢** | 16,0 s | 3,65 |
| v000 (Sonnet 5) | 0/6 | 6/6 | 0/8 | 6/20 | 0 % | **100 %** | 28,6 % | 2,41 ¢ | 10,7 s | 2,15 |
| v001 (+ filtro) | 0/6 | 6/6 | 0/8 | 6/20 | 0 % | **100 %** | 35,7 % | 2,32 ¢ | **10,2 s** | **2,10** |
| v002 (+ formato) | 6/6 | 6/6 | 2/8 | 14/20 | 57,1 % | 100 % | 50,0 % | 2,70 ¢ | 10,9 s | 2,20 |
| v003 (+ protocolo comparativas) | **6/6** | **6/6** | **8/8** | **20/20** | **100 %** | **100 %** | **64,3 %** | 3,39 ¢ | 12,7 s | 2,70 |
| v004 (+ middleware XBRL) | **6/6** | **6/6** | **8/8** | **20/20** | **100 %** | **100 %** | 57,1 % | 3,47 ¢ | 12,3 s | 2,75 |
| v005 (+ reparador de citas) | **6/6** | **6/6** | **8/8** | **20/20** | **100 %** | **100 %** | 57,1 % | 3,42 ¢ | 12,0 s | 2,75 |

- En v000 y v001 Sonnet respondió bien las 6 extractivas pero envolvió la
  cita entre comillas (`"If we underestimate...`), así que no aparece
  literal en el fragmento y las 6 fallan. La regla de cita literal de v002 lo
  corrige: 0/6 → 6/6.
- Las 2 comparativas acertadas en v002 (015, 019) son las únicas en que el
  agente buscó por su cuenta en el Item 7 y citó. En las otras 6 solo
  consultó XBRL.
- Coste de las tres pasadas: 1,48 $.
- v003 es el cambio que mueve la nota: comparativas 2/8 → 8/8. Con el
  protocolo, el agente busca en el MD&A y cita en las 8.
- v004 y v005 no cambian nada en este golden porque v003 ya acierta las 20:
  el conjunto no tiene preguntas que los activen (ninguna sin respuesta,
  ningún beneficio por acción, ninguna cita mal copiada). No empeoran nada y
  cuestan lo mismo. Su efecto se mide aparte con `estres.jsonl`.
- v004 y v005 dan la misma cifra en las 20 preguntas y la misma cita en 18;
  el resto es redacción. Todas las citas de v004 ya eran literales (Sonnet
  copia bien el ’ con la regla de v002), así que en este golden el reparador
  de v005 no tenía nada que arreglar. No se puede saber si llegó a actuar: el
  registro de intervenciones se añadió después de estas pasadas.
- El recall del agente baja de 64,3 a 57,1 % por una sola pregunta
  (gjhh-013): en v004 y v005 el agente buscó con otra consulta, no recuperó
  el fragmento del ancla y citó otra frase válida del MD&A. Sigue acertando
  porque la cita se comprueba en el fragmento citado, no contra el ancla. Con
  14 preguntas con ancla, una pregunta son 7 puntos: es variación de la
  consulta del modelo, no efecto de los middlewares.
- Una sola pasada por versión. Sonnet 5 no admite temperatura 0, pero las
  respuestas salen muy estables entre pasadas.

## Resultados en el golden oficial con Sonnet 5 (2026-09-20)

| Versión | Extractivas | Numéricas | Comparativas | Total | Cita | Cifra | Recall@5 agente | Coste medio | Latencia | Llamadas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v000 (Sonnet 5) | 0/6 | 7/7 | 0/7 | 7/20 | 0 % | 64,3 % | 61,5 % | 3,06 ¢ | 12,4 s | 2,60 |
| v002 (+ filtro + formato) | **6/6** | **7/7** | **7/7** | **20/20** | **100 %** | **100 %** | **92,3 %** | 3,21 ¢ | 12,8 s | 2,70 |
| v005 (versión completa) | **6/6** | **7/7** | **7/7** | **20/20** | **100 %** | **100 %** | 84,6 % | 3,88 ¢ | 13,5 s | 2,80 |

- En el oficial las comparativas piden expresamente la explicación de la
  dirección («¿y a qué lo atribuye?»), así que el agente busca en el MD&A sin
  necesidad del protocolo de v003: v002 ya acierta las 20. El oficial no
  distingue entre v002 y las versiones siguientes.
- v000 falla las 6 extractivas por las comillas en la cita y las
  comparativas por dar la variación en vez del último valor; una pregunta
  (of-020) terminó por timeout del proveedor.
- v005 (21-sep) repite el 20/20. El middleware intervino una vez: en of-006
  (extractiva) el agente dio como cifra un 23 % leído del Item 7A; el
  comprobador le devolvió el desajuste, el modelo mantuvo la cifra y la
  respuesta siguió siendo correcta. Es un falso positivo: cuesta una llamada
  al modelo más y explica parte de la subida de coste (3,21 → 3,88 ¢). El
  recall baja una pregunta (13 con ancla, 7,7 puntos cada una).

## Resultados en el golden propio con Haiku 4.5 (2026-09-20)

| Versión | Extractivas | Numéricas | Comparativas | Total | Cita | Cifra | Recall@5 agente | Coste medio | Latencia | Llamadas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline grupo (Gemini 3.8 Flash) | 5/6 | 6/6 | 0/8 | 11/20 | 35,7 % | 78,6 % | 50,0 % | 0,96 ¢ | 16,0 s | 3,65 |
| v000 (Haiku) | 3/6 | 4/6 | 0/8 | 7/20 | 21,4 % | 35,7 % | 28,6 % | 1,48 ¢ | 11,8 s | 2,55 |
| v001 Haiku (+ filtro) | 4/6 | 3/6 | 0/8 | 7/20 | 28,6 % | 28,6 % | 35,7 % | 1,31 ¢ | 11,7 s | 2,55 |

Diagnóstico de v001 (`evaluar.py --ids`): los fallos de numéricas y
comparativas no son de razonamiento sino de formato. La cifra correcta sale
en millones (`12914`, `"millones USD"`), en comparativas a veces se da la
variación en vez del último valor, y una cita falla por el apóstrofo recto
(') frente al tipográfico (’) del informe. De ahí la v002.

Con 6 preguntas por familia, una pregunta arriba o abajo es ruido: la
diferencia v000 → v001 en extractivas y numéricas no es concluyente.

### Diagnóstico sobre 6 preguntas (gjhh-003, 007, 010, 011, 013, 014)

| Versión | Modelo | Aciertos | Cifra correcta | Cita correcta |
| --- | --- | --- | --- | --- |
| v001 | Haiku 4.5 | 0/6 | 0/5 | 0/3 |
| v002 | Haiku 4.5 | 3/6 | 5/5 | 0/3 |

- Las reglas de formato arreglan todas las cifras (numéricas y comparativas).
- La cita de gjhh-003 sigue con apóstrofo recto pese a la instrucción: el
  prompt no basta, hace falta una corrección determinista.
- Las comparativas siguen sin cita porque el agente no busca texto.
- Primer intento con Sonnet 5: 0 llamadas completadas por el error de
  temperature (sin coste).
| v002 | Sonnet 5 | 4/6 | 5/5 | 1/3 |

- Sonnet 5 acierta la cita de gjhh-003, pero es un **falso positivo del
  evaluador**: cita otra frase literal del Item 1A («partners primarily
  located in Asia») y no da los seis países que pide la pregunta. El
  evaluador actual solo comprueba que la cita sea literal, no que respalde la
  respuesta. En lo sustantivo, Sonnet y Haiku empatan en estas 6 preguntas.
- Sonnet redacta mejor las cifras en prosa («12.914 millones de dólares»
  frente al «12.914 mil millones» de Haiku) y cuesta 2,5 veces más
  (2,33 ¢ frente a 0,93 ¢ por pregunta).
