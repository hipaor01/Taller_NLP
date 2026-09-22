# Experimentos del equipo

Esta carpeta separa el baseline común de las variantes de cada miembro.

```text
experimentos/
├── baseline.py
├── <nombre-miembro>/
│   ├── agente_v001.py
│   ├── agente_v002.py
│   └── README.md
└── resultados/             # generado; no se versiona
```

## Reglas de trabajo

1. `baseline.py` es la referencia común y no se modifica para probar mejoras.
2. Cada miembro crea una subcarpeta con un identificador estable, por ejemplo
   `experimentos/higinio/`.
3. Cada cambio evaluable recibe una versión nueva. No se reutiliza
   `agente_v001.py` para un experimento diferente.
4. Las implementaciones se definen en módulos `.py`, no solo en el notebook.
   Así el manifiesto puede registrar una referencia y una huella de su código.
5. Los artefactos grandes y resultados generados no se versionan. Los
   manifiestos que se quieran conservar se mueven a
   `experimentos/<nombre-miembro>/manifiestos/`, que no está ignorada.
6. Nunca se guarda `OPENROUTER_API_KEY` en esta carpeta ni en un script.

Una variante debe reutilizar todo lo que no pretende estudiar. Por ejemplo,
si se está probando otro retriever, se pueden importar del baseline
`crear_corpus_baseline()` y `crear_configuracion_baseline()` y cambiar solo la
construcción del retriever y de `FabricaHerramientas`.

Toda variante ejecutable desde el notebook debe exponer una factoría común sin
argumentos:

```python
def crear_constructor() -> ConstructorAgente:
    ...
```

La interfaz usa `experimentos.jchulvi.agente_v2` por defecto. Esta variante
declara Qdrant como recurso de ejecución: la fachada lo inicia, valida/carga y
detiene automáticamente alrededor de cada respuesta o evaluación. Para
seleccionar otra variante antes de la primera llamada a `responder()`:

```bash
export TALLER_VARIANTE_AGENTE=experimentos.higinio.agente_v002
```

El módulo indicado debe ser importable y exponer `crear_constructor()`. Como el
motor y su memoria se conservan durante toda la sesión, hay que reiniciar el
proceso o el kernel para cambiar de variante después de la primera respuesta.

La interfaz entregable permite ejecutar el hold-out directamente desde un clon
limpio y guardar la tabla compatible con el notebook:

```python
from agente import evaluar, resumir

tabla = evaluar("holdout.jsonl", salida="resultados/holdout.csv")
fila_informe = resumir(tabla, "jchulvi-v2")
```

`evaluar_informe("holdout.jsonl")` conserva alternativamente el informe
estructurado completo, con el desglose por pregunta y las métricas agregadas.
El resumen incluye los aciertos por familia en formato `correctas/total` para
`extractiva`, `numerica` y `comparativa`.

La misma evaluación puede lanzarse directamente desde la terminal:

```bash
python -m agente \
  --evaluar holdout.jsonl \
  --salida resultados/holdout.csv
```

Una tabla ya generada se puede convertir en la fila resumen del informe. La
etiqueta se deduce del nombre del fichero (`baseline` en este ejemplo):

```bash
python -m agente --resumir resultados/baseline.csv
```

También se pueden indicar la etiqueta y un CSV de salida explícitos:

```bash
python -m agente \
  --resumir resultados/baseline.csv \
  --etiqueta baseline \
  --salida resultados/resumen_baseline.csv
```

Una ejecución ya guardada como manifiesto puede promocionarse a `resultados/`
sin volver a llamar al agente. El CSV detallado y su resumen se derivan del
mismo `InformeEvaluacion`, por lo que representan exactamente la misma
ejecución:

```bash
python -m agente \
  --desde-manifiesto experimentos/resultados/baseline_20260919T071446Z.json \
  --salida resultados/baseline.csv \
  --resumen resultados/resumen_baseline.csv
```

La etiqueta del resumen se deduce de `--salida` (`baseline` en el ejemplo).
Puede sustituirse con `--etiqueta`.

Para ejecutar otra variante sin modificar código:

```bash
python -m agente \
  --variante experimentos.higinio.agente_v002 \
  --evaluar holdout.jsonl \
  --salida resultados/final.csv
```

## Telemetría de modelos auxiliares

Los componentes que hagan llamadas adicionales a un LLM deben compartir un
`RegistroTelemetriaAuxiliar` con `ConstructorAgente`. Después de cada llamada,
el componente registra una `LlamadaModeloAuxiliar` con modelo, tokens, coste,
latencia y posible error. El motor incorpora automáticamente esas llamadas al
desglose y a los totales de cada respuesta.

Si falta cualquiera de las mediciones auxiliares, el total correspondiente se
marca como desconocido (`None`) en vez de presentar una suma parcial como si
fuera completa. La latencia total no se suma manualmente: ya es tiempo de pared
y contiene el tiempo empleado por las llamadas auxiliares.

## Ejecutar el baseline

Desde la raíz del proyecto:

```bash
conda activate ./.venv
export OPENROUTER_API_KEY="..."
python -m experimentos.baseline --pregunta \
  "¿Cuál fue el beneficio neto de Apple en FY2025?"
```

El comando muestra por `stderr` cuándo comienza y termina cada llamada al
modelo y a las herramientas. Cada llamada al modelo tiene un timeout de 60
segundos. El agente usa un límite de 18 peticiones por minuto, sin ráfagas,
para respetar el máximo de 20 RPM de las cuentas nuevas de OpenRouter.
Si aun así reciben un HTTP 429, hacen hasta tres reintentos con esperas de
aproximadamente 5, 10 y 20 segundos. Esos mismos reintentos se aplican al 400
genérico `Provider returned error`, sin reintentar otros errores 400 que sí
indican una petición inválida. En OpenRouter se excluyen además los bloques de
razonamiento opaco del historial para evitar firmas de pensamiento inválidas
en los turnos posteriores a una herramienta. La primera búsqueda textual puede
tardar algo más que las siguientes porque carga BGE en memoria.

Este proyecto usa un entorno Conda almacenado en `.venv`; no se activa con
`source .venv/bin/activate`. También se puede evitar la activación y ejecutar
directamente su intérprete:

```bash
.venv/bin/python -m experimentos.baseline --pregunta \
  "¿Cuál fue el beneficio neto de Apple en FY2025?"
```

## Evaluación y progreso reanudable

Para iniciar una evaluación:

```bash
python -m experimentos.baseline --evaluar golden_set.jsonl
```

El programa crea automáticamente un fichero en
`experimentos/resultados/progreso/` y guarda en él cada pregunta completamente
evaluada. El progreso se conserva si se cancela con `Ctrl+C`, falla la conexión
o se agotan los reintentos de un HTTP 429 o del error genérico del proveedor.

### Continuar una evaluación interrumpida

Ejecuta de nuevo exactamente el mismo comando, con el mismo agente y el mismo
golden set:

```bash
python -m experimentos.baseline --evaluar golden_set.jsonl
```

El evaluador cargará las preguntas ya terminadas y comenzará por la primera
pendiente. Esos casos no vuelven a llamar al agente, por lo que no vuelven a
generar coste.

Los resultados antiguos cuyo error sea `Provider returned error`, HTTP 429 o
`GraphRecursionError` se consideran incompletos y se vuelven a ejecutar
automáticamente. El resto del progreso válido se conserva; no hace falta
utilizar `--reiniciar-progreso` para reparar esos casos.

### Elegir el fichero de progreso

Para compartir una convención de nombres o guardar el progreso en otro lugar:

```bash
python -m experimentos.baseline \
  --evaluar golden_set.jsonl \
  --progreso experimentos/resultados/mi_progreso.json
```

Para continuarla hay que reutilizar también esa misma ruta:

```bash
python -m experimentos.baseline \
  --evaluar golden_set.jsonl \
  --progreso experimentos/resultados/mi_progreso.json
```

### Reiniciar desde cero

Para descartar explícitamente el progreso automático anterior y volver a
ejecutar todas las preguntas:

```bash
python -m experimentos.baseline \
  --evaluar golden_set.jsonl \
  --reiniciar-progreso
```

Si se utilizó una ruta personalizada, deben indicarse ambas opciones:

```bash
python -m experimentos.baseline \
  --evaluar golden_set.jsonl \
  --progreso experimentos/resultados/mi_progreso.json \
  --reiniciar-progreso
```

El reinicio elimina el progreso anterior de esa ruta, por lo que sus preguntas
volverán a ejecutarse y a generar coste. Si el fichero pertenece a otro agente,
otro golden set, otro corpus o una configuración de evaluación incompatible,
el programa no mezclará resultados: pedirá reiniciarlo o elegir otra ruta.

El limitador es compartido dentro de un proceso. Dos procesos o dos ordenadores
que utilicen la misma cuenta de OpenRouter no coordinan su ritmo entre sí y
deben repartirse conjuntamente el límite de la cuenta.

La generación de respuestas hace llamadas reales al modelo y tiene coste. Los
tres evaluadores son locales y deterministas: verifican la cita literal, la
cifra con tolerancia y los nombres de las herramientas usadas. Para indicar
una ruta concreta para el manifiesto:

```bash
python -m experimentos.baseline \
  --evaluar golden_set.jsonl \
  --manifiesto experimentos/resultados/baseline_prueba.json
```

Antes de comparar un manifiesto guardado:

```python
from taller_nlp import ManifiestoExperimento

registro = ManifiestoExperimento.cargar(
    "experimentos/resultados/baseline_prueba.json"
)
print(registro.comprobar_reproducibilidad())
```
