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
