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
