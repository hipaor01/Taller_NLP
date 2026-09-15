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
segundos. Agente y juez comparten un límite de 18 peticiones por minuto, sin
ráfagas, para respetar el máximo de 20 RPM de las cuentas nuevas de OpenRouter.
Si aun así reciben un HTTP 429, hacen hasta tres reintentos con esperas de
aproximadamente 5, 10 y 20 segundos. La primera búsqueda textual puede tardar
algo más que las siguientes porque carga BGE en memoria.

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
o se agotan los reintentos de un HTTP 429.

### Continuar una evaluación interrumpida

Ejecuta de nuevo exactamente el mismo comando, con el mismo agente y el mismo
golden set:

```bash
python -m experimentos.baseline --evaluar golden_set.jsonl
```

El evaluador cargará las preguntas ya terminadas y comenzará por la primera
pendiente. Esos casos no vuelven a llamar al agente ni al juez, por lo que no
vuelven a generar coste.

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

La evaluación hace llamadas reales al modelo y al juez de citas, por lo que
tiene coste. Para indicar una ruta concreta para el manifiesto:

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
