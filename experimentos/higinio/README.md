# Experimentos de Higinio

## `agente_v001`: búsqueda densa con filtros de metadatos

Esta variante mantiene el modelo, prompt, corpus, herramientas y evaluación
del baseline. Solo cambia el recuperador FAISS:

- El baseline recupera el top-k global sin aplicar filtros.
- La v001 busca sobre todo el índice cuando recibe algún filtro, descarta los
  fragmentos cuyo `ticker`, `fiscal_year` o `item` no coinciden y conserva los
  primeros `k` resultados válidos.

La búsqueda sigue siendo densa, exacta y exhaustiva sobre `IndexFlatIP`. El
filtrado se realiza después del ranking, tal como propone el arreglo 1 del
notebook.

Para evaluarla desde la raíz del proyecto:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v001 \
  --evaluar golden_set.jsonl \
  --salida resultados/agente_v001.csv
```

Para generar la fila de métricas que se utilizará en el informe:

```bash
python -m agente \
  --resumir resultados/agente_v001.csv \
  --etiqueta agente_v001 \
  --salida resultados/resumen_agente_v001.csv
```

En Python o en el notebook puede seleccionarse antes de crear el runtime:

```python
import os

os.environ["TALLER_VARIANTE_AGENTE"] = "experimentos.higinio.agente_v001"
```

Hay que reiniciar el proceso o el kernel si ya se había utilizado otra
variante, porque la interfaz conserva el motor construido durante la sesión.

## `agente_v002`: híbrido BM25 + denso mediante RRF

Esta variante acumula el arreglo anterior y añade un ranking léxico BM25. La
lista densa filtrada y la lista BM25 se fusionan mediante Reciprocal Rank
Fusion con `kk=60`; no se suman sus puntuaciones porque usan escalas distintas.

Para evaluarla y guardar el resultado junto al baseline y la v001:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v002 \
  --evaluar golden_set.jsonl \
  --salida resultados/agente_v002.csv
```

Para generar el resumen del informe:

```bash
python -m agente \
  --resumir resultados/agente_v002.csv \
  --etiqueta agente_v002 \
  --salida resultados/resumen_agente_v002.csv
```

## `agente_v003`: reescritura LLM sobre la v001

Esta variante parte directamente de la v001: mantiene la búsqueda densa con
filtros de metadatos y, antes de cada `search_filings`, pide al mismo modelo
que traduzca la consulta al inglés usando vocabulario propio de los informes.
No incorpora BM25 ni RRF.

La llamada auxiliar comparte el limitador y los reintentos del agente. Sus
tokens y coste se suman a los totales de la respuesta, mientras que la
latencia total ya incluye el tiempo de reescritura. Sin clave o si falla el
modelo, se conserva la consulta original o la reescritura de respaldo.
Dentro del agente, `search_filings` no recibe el identificador del golden set,
por lo que su respaldo efectivo es la consulta original; las reescrituras
grabadas se usan cuando se llama directamente a `reescribir(..., id_golden)`.

Para evaluarla:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v003 \
  --evaluar golden_set.jsonl \
  --salida resultados/agente_v003.csv
```

Para generar el resumen del informe:

```bash
python -m agente \
  --resumir resultados/agente_v003.csv \
  --etiqueta agente_v003 \
  --salida resultados/resumen_agente_v003.csv
```

## `agente_v004`: v002 + reescritura LLM

Esta variante combina los dos cambios anteriores: primero reescribe la
consulta mediante el LLM y después entrega la misma consulta inglesa a las
ramas densa filtrada y BM25, que se fusionan mediante RRF con `kk=60`.

Conserva la telemetría auxiliar de v003, por lo que el coste, los tokens y la
latencia de cada reescritura quedan incluidos en la evaluación.

Para evaluarla:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v004 \
  --evaluar golden_set.jsonl \
  --salida resultados/agente_v004.csv
```

Para generar el resumen del informe:

```bash
python -m agente \
  --resumir resultados/agente_v004.csv \
  --etiqueta agente_v004 \
  --salida resultados/resumen_agente_v004.csv
```

## `agente_v005`: v001 + verificación XBRL

Esta variante vuelve a tomar la v001 como referencia: conserva la búsqueda
densa con filtros y añade un middleware `after_model` que contrasta cualquier
cifra de la respuesta estructurada con los hechos XBRL de la misma compañía y
ejercicio.

Si ninguna cifra reportada cuadra con una tolerancia relativa del 1 %, el
middleware devuelve al modelo el valor afirmado, los hechos disponibles y un
salto explícito a `model`. Solo permite una corrección por ejecución para evitar
bucles.

Para evaluarla:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v005 \
  --evaluar golden_set.jsonl \
  --salida resultados/agente_v005.csv
```

Para generar el resumen del informe:

```bash
python -m agente \
  --resumir resultados/agente_v005.csv \
  --etiqueta agente_v005 \
  --salida resultados/resumen_agente_v005.csv
```

## `agente_v006`: v004 + verificación XBRL

Esta variante mantiene íntegramente la combinación de v004 —reescritura LLM,
búsqueda densa con filtros y fusión híbrida BM25 mediante RRF— y añade el mismo
middleware determinista de verificación XBRL utilizado por v005.

Para evaluarla:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v006 \
  --evaluar golden_set.jsonl \
  --salida resultados/agente_v006.csv
```

Para generar el resumen del informe:

```bash
python -m agente \
  --resumir resultados/agente_v006.csv \
  --etiqueta agente_v006 \
  --salida resultados/resumen_agente_v006.csv
```

## `agente_v007`: v005 + reparación determinista de citas

Esta variante conserva íntegramente la v005 —búsqueda densa con filtros de
metadatos y verificación XBRL— y añade un middleware que valida el par
`cita`/`chunk_id` contra los resultados efectivos de `search_filings`.

El middleware no consulta el golden set ni inventa evidencia. Si una cita
literal coincide con un único chunk recuperado, completa su `chunk_id`. Si el
modelo declaró un chunk recuperado pero reformateó una tabla o una frase,
sustituye la cita por una línea literal solo cuando la coincidencia léxica y
numérica es inequívoca. En caso contrario, deja la respuesta intacta.

Para evaluarla:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v007 \
  --evaluar golden_set.jsonl \
  --salida resultados/agente_v007.csv
```

Para generar su resumen:

```bash
python -m agente \
  --resumir resultados/agente_v007.csv \
  --etiqueta agente_v007 \
  --salida resultados/resumen_agente_v007.csv
```

## `agente_v008`: v005 + guardrail comparativo

Esta variante parte directamente de v005 para medir de forma aislada el
efecto del guardrail. Detecta preguntas que comparan explícitamente dos
ejercicios y, antes de aceptar la respuesta estructurada, exige que exista una
llamada exitosa a `search_filings` con resultados y que la respuesta cite uno
de los `chunk_id` recuperados.

Si falta la búsqueda o la cita, devuelve una instrucción de corrección al
modelo. Solo permite una corrección por ejecución para evitar bucles. El
guardrail no consulta el golden set, no ejecuta búsquedas por su cuenta y no
incorpora el reparador de citas de v007.

Para evaluarla:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v008 \
  --evaluar golden_set.jsonl \
  --salida resultados/agente_v008.csv
```

Para generar su resumen:

```bash
python -m agente \
  --resumir resultados/agente_v008.csv \
  --etiqueta agente_v008 \
  --salida resultados/resumen_agente_v008.csv
```

## `agente_v009`: v008 + reparación determinista de citas

Esta variante conserva el guardrail comparativo de v008 y añade el
`VerificadorCitas` probado en v007. En los hooks `after_model`, LangChain
recorre los middlewares en orden inverso: el verificador puede reparar primero
un par `cita`/`chunk_id` usando los resultados reales de `search_filings`; el
guardrail evalúa después la respuesta ya reparada y solo solicita otra vuelta
al modelo cuando sigue faltando evidencia recuperada.

Para evaluarla:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v009 \
  --evaluar golden_set.jsonl \
  --salida resultados/agente_v009.csv
```

Para generar su resumen:

```bash
python -m agente \
  --resumir resultados/agente_v009.csv \
  --etiqueta agente_v009 \
  --salida resultados/resumen_agente_v009.csv
```

## `agente_v010`: v009 + contrato comparativo estricto

Esta variante conserva todos los componentes de v009 y sustituye únicamente
su esquema de respuesta. El nuevo contrato clasifica la salida como
extractiva, numérica o comparativa. En las comparativas exige los ejercicios y
valores inicial y final, además del `ticker`, la unidad y `fuente="ambas"`.

Una vez validada la salida, normaliza de forma determinista `cifra` y
`ejercicio` al valor y ejercicio finales. También recalcula las variaciones
absoluta y porcentual. Así, una variación calculada por el modelo nunca puede
ocupar accidentalmente el campo `cifra` que puntúa el evaluador. Las citas
siguen bajo el guardrail y el verificador de v009, evitando añadir otro
middleware con su propio ciclo de corrección.

Para evaluarla:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v010 \
  --evaluar golden_set.jsonl \
  --salida resultados/agente_v010.csv
```

Para generar su resumen:

```bash
python -m agente \
  --resumir resultados/agente_v010.csv \
  --etiqueta agente_v010 \
  --salida resultados/resumen_agente_v010.csv
```

## `agente_v011`: v009 + contrato comparativo tolerante

Esta variante vuelve a partir de v009 y conserva sus tres middlewares. Añade
los mismos campos comparativos de v010, pero todos son opcionales y no rechaza
la respuesta cuando faltan datos, los periodos están invertidos o la fuente no
es `"ambas"`. Si recibe `cifra_final`, la copia de forma determinista a
`cifra`; cuando también están presentes ambos valores, recalcula las variaciones.

El contrato también puede inferir una comparativa cuando el modelo proporciona
los dos ejercicios y los dos valores aunque omita `tipo_respuesta`. De este
modo se mantiene la desambiguación buscada en v010 sin convertir campos
auxiliares ausentes en reintentos de salida estructurada.

Para evaluarla desde cero:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v011 \
  --evaluar golden_set.jsonl \
  --salida resultados/agente_v011.csv \
  --reiniciar-progreso
```

Para generar su resumen:

```bash
python -m agente \
  --resumir resultados/agente_v011.csv \
  --etiqueta agente_v011 \
  --salida resultados/resumen_agente_v011.csv
```

## `agente_v012`: v011 + instrucciones comparativas preventivas

Esta variante conserva el contrato tolerante y los tres middlewares de v011.
Solo amplía el prompt del sistema para que, antes de responder una comparativa,
el modelo consulte ambos ejercicios con el mismo concepto XBRL, recupere
evidencia mediante `search_filings` y copie una `cita` literal junto con su
`chunk_id`.

Las instrucciones evitan búsquedas innecesarias: si el primer resultado es
pertinente, debe responder sin repetir la consulta; cuando no lo sea, solo
permite una reformulación. La normalización determinista de v011 sigue fijando
la cifra principal al ejercicio final cuando el modelo proporciona ese campo.

Para evaluarla desde cero:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v012 \
  --evaluar golden_set.jsonl \
  --salida resultados/agente_v012.csv \
  --reiniciar-progreso
```

Para generar su resumen:

```bash
python -m agente \
  --resumir resultados/agente_v012.csv \
  --etiqueta agente_v012 \
  --salida resultados/resumen_agente_v012.csv
```

## `agente_v013`: v012 + selección dirigida del fragmento

Esta variante conserva todos los componentes de v012 y amplía solamente su
prompt. La primera búsqueda comparativa deja que compitan Items 7 y 8, usando
el nombre financiero en inglés y los dos valores conocidos. El agente debe
preferir una línea o tabla que contenga la magnitud y ambos importes, en lugar
de citar comentarios genéricos que solo mencionen el tema.

Cuando la primera búsqueda no recupera evidencia suficiente, la única
reformulación permitida se dirige a Item 8 para estados financieros y tablas, o
a Item 7 si la pregunta solicita causas o explicaciones de la dirección.

Para evaluarla desde cero:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v013 \
  --evaluar golden_set.jsonl \
  --salida resultados/agente_v013.csv \
  --reiniciar-progreso
```

Para generar su resumen:

```bash
python -m agente \
  --resumir resultados/agente_v013.csv \
  --etiqueta agente_v013 \
  --salida resultados/resumen_agente_v013.csv
```

## `agente_v014`: v012 + recuperación híbrida BM25-densa

Esta variante conserva la configuración, el prompt, el contrato de salida y
los middlewares de v012. Sustituye únicamente el recuperador denso filtrado por
el recuperador híbrido BM25-denso con fusión RRF ya validado en v002.

El objetivo es mejorar la recuperación de fragmentos con términos literales
relevantes sin perder la similitud semántica ni los filtros de ticker,
ejercicio fiscal e item.

Para evaluarla desde cero:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v014 \
  --evaluar golden_set.jsonl \
  --salida resultados/agente_v014.csv \
  --reiniciar-progreso
```

Para generar su resumen:

```bash
python -m agente \
  --resumir resultados/agente_v014.csv \
  --etiqueta agente_v014 \
  --salida resultados/resumen_agente_v014.csv
```

## `agente_v015`: v013 + recuperación híbrida BM25-densa

Esta variante conserva la selección dirigida de evidencia de v013 y sustituye
únicamente su recuperador denso filtrado por el recuperador híbrido BM25-denso
con fusión RRF de v002. Así combina la elección explícita del Item 8 para tablas
financieras comparativas con la mejora de ranking léxico-semántico observada en
v014.

Para evaluarla desde cero:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v015 \
  --evaluar golden_set.jsonl \
  --salida resultados/agente_v015.csv \
  --reiniciar-progreso
```

Para generar su resumen:

```bash
python -m agente \
  --resumir resultados/agente_v015.csv \
  --etiqueta agente_v015 \
  --salida resultados/resumen_agente_v015.csv
```

## `agente_v016`: v015 + Gemma 4 26B A4B

Esta variante conserva íntegramente el prompt, el contrato, los middlewares y
el recuperador híbrido de v015. Sustituye únicamente el modelo principal por
`openrouter:google/gemma-4-26b-a4b-it` y fija para el experimento unas tarifas
de 0,042 USD por millón de tokens de entrada y 0,22 USD por millón de tokens de
salida.

La configuración explícita impide que `TALLER_MODELO_AGENTE` cambie el modelo
de esta variante y permite comparar su coste y fiabilidad agéntica con v015.

Para evaluarla desde cero:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v016 \
  --evaluar golden_set.jsonl \
  --salida resultados/agente_v016.csv \
  --reiniciar-progreso
```

Para generar su resumen:

```bash
python -m agente \
  --resumir resultados/agente_v016.csv \
  --etiqueta agente_v016 \
  --salida resultados/resumen_agente_v016.csv
```

## Progreso reanudable y análisis detallado

Todas las evaluaciones lanzadas con `python -m agente --evaluar` guardan ahora
un JSON de progreso automático en `experimentos/resultados/progreso/`. Por
ejemplo, la evaluación de v005 crea un fichero con este patrón:

```text
experimentos/resultados/progreso/agente_v005_golden_set_<hash>_citas-v5.json
```

El fichero conserva cada respuesta completa, sus llamadas a herramientas y el
desglose de los evaluadores. También permite continuar una ejecución
interrumpida sin pagar de nuevo las preguntas terminadas.

Se puede elegir una ruta estable explícitamente:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v005 \
  --evaluar golden_set.jsonl \
  --salida resultados/agente_v005.csv \
  --progreso experimentos/resultados/progreso/agente_v005.json
```

Para repetirla desde cero, hay que añadir `--reiniciar-progreso`.
