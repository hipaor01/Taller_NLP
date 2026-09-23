# Agente financiero sobre informes SEC 10-K

## Puesta en marcha desde un clon limpio

Los pasos siguientes dejan listo el agente predeterminado,
`experimentos.jchulvi.agente_v3`, tanto para responder una pregunta como para
evaluar un conjunto JSONL. Su código está en
[`experimentos/jchulvi/agente_v3.py`](experimentos/jchulvi/agente_v3.py).
Ejecuta todos los comandos desde la raíz del repositorio.

### Ubicación del código principal

`agente_v2.py` conserva los contratos comunes de las cuatro herramientas, pero
hereda de la versión anterior algunos cambios en `search_filings` y en las
descripciones entregadas al modelo:

- El recuperador personalizado que determina el comportamiento efectivo de
  `search_filings` está en
  [`RecuperadorHibrido`](experimentos/jchulvi/agente.py#L244-L340).
- Las descripciones personalizadas de `get_xbrl_fact` y `search_filings`, junto
  con su conexión a la fábrica, están en la función `crear_constructor()` del
  mismo archivo: [`crear_constructor()`](experimentos/jchulvi/agente.py#L423-L475).
- La fábrica compartida, los contratos de las cuatro tools y sus
  implementaciones comunes se encuentran en
  [`FabricaHerramientas`](src/taller_nlp/tool_factory.py#L77-L366). Ahí están
  las implementaciones base de `list_available`, `get_xbrl_fact`,
  `search_filings` y `read_section`.
- El evaluador común que carga los casos, comprueba cifras, trayectorias, citas
  y `recall@k` está en
  [`EvaluadorFinanciero`](src/taller_nlp/evaluation.py#L37-L365).
- El esquema y validador común de los conjuntos JSONL está en
  [`CasoGolden`](src/taller_nlp/golden.py#L26-L276). Su método
  [`cargar_jsonl()`](src/taller_nlp/golden.py#L238-L276) comprueba el JSON de
  cada línea, los campos obligatorios, las reglas de cada familia, los
  identificadores duplicados y la coherencia con el corpus.
- Las funciones públicas
  [`responder()`](agente/interfaz.py#L306-L308) y
  [`evaluar()`](agente/interfaz.py#L311-L323) están en `agente/interfaz.py`.
  Su exposición mediante `python -m agente --responder` y `--evaluar` se
  implementa en [`agente/__main__.py`](agente/__main__.py#L148-L268).

### 1. Requisitos

- **Git**.
- **Python 3.10 o posterior**, con soporte para crear entornos `venv`.
- **Docker** con el daemon en ejecución. Puede instalarse Docker Desktop en
  macOS, Windows o Linux, o Docker Engine en Linux, siguiendo la
  [documentación oficial](https://docs.docker.com/get-started/).
- Una cuenta de **OpenRouter**, una
  [API key](https://openrouter.ai/settings/keys) y saldo o acceso suficiente
  para los modelos utilizados. Las llamadas al agente tienen coste.
- Acceso a Internet para OpenRouter y, durante el primer arranque, para que
  Docker descargue la imagen fijada de Qdrant si no está instalada.

El repositorio ya incluye el corpus, los hechos XBRL y los embeddings del
corpus. No es necesario descargar informes ni recalcular el índice para usar el
agente predeterminado. Docker debe permitir al contenedor utilizar 2 CPU y 2 GB
de memoria.

Comprueba los requisitos antes de continuar:

```bash
git --version
python3 --version
docker version
```

`docker version` debe mostrar tanto el cliente como el servidor. Si solo muestra
el cliente o devuelve un error de conexión, inicia Docker Desktop o el servicio
de Docker.

### 2. Clonar el repositorio

```bash
git clone https://github.com/hipaor01/Taller_NLP.git
cd Taller_NLP
```

### 3. Crear e instalar el entorno virtual

En macOS o Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

En Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

La instalación editable obtiene las dependencias directamente de
[`pyproject.toml`](pyproject.toml). Para instalar también las dependencias de
pruebas y notebooks:

```bash
python -m pip install -e ".[test,notebook]"
```

### 4. Configurar OpenRouter

Crea una API key de uso normal en la
[página de claves de OpenRouter](https://openrouter.ai/settings/keys) y expórtala
en la terminal desde la que se ejecutará el agente.

En macOS o Linux:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

En Windows PowerShell:

```powershell
$env:OPENROUTER_API_KEY="sk-or-v1-..."
```

La variable solo dura lo que dure esa sesión de terminal. El proyecto ignora
los ficheros `.env`, pero **no los carga automáticamente**: guardar ahí la clave
no sustituye al `export`. No escribas la clave en el código, el README ni un
archivo que pueda terminar en Git.

El agente predeterminado usa a través de OpenRouter el modelo
`deepseek/deepseek-v4-flash-0731`, con el proveedor DeepInfra, y genera los
embeddings de cada consulta con `voyageai/voyage-4-lite`. Los embeddings del
corpus ya están versionados, pero cada búsqueda textual nueva necesita acceso a
la API.

### 5. Responder una pregunta

La opción `--responder` imprime como JSON la respuesta estructurada junto con su
traza y telemetría. Este comando realiza llamadas reales a OpenRouter:

```bash
python -m agente \
  --responder "¿Cuál fue el beneficio neto de Apple en FY2025?"
```

La misma interfaz puede utilizarse desde Python:

```python
from agente import responder

resultado = responder("¿Cuál fue el beneficio neto de Apple en FY2025?")
respuesta = resultado["structured_response"]
print(respuesta.respuesta)
print(respuesta.cifra, respuesta.unidad)
```

El primer arranque puede tardar más porque Docker debe descargar Qdrant, crear
el contenedor `taller-nlp-jchulvi-v002` y cargar el índice. La aplicación expone
Qdrant solo en `127.0.0.1:6339` y detiene el contenedor al terminar la operación;
los datos cargados quedan conservados para ejecuciones posteriores.

### 6. Evaluar el agente predeterminado

Para evaluar las 20 preguntas del conjunto propio y guardar la tabla detallada:

```bash
python -m agente \
  --evaluar golden_set.jsonl \
  --salida resultados/evaluacion_jchulvi_v3.csv
```

Para utilizar el conjunto oficial:

```bash
python -m agente \
  --evaluar golden_set_oficial.jsonl \
  --salida resultados/evaluacion_jchulvi_v3_oficial.csv
```

La evaluación hace llamadas reales al modelo, puede tardar varios minutos y
consume saldo. El comando muestra la tabla, guarda el CSV indicado y crea un
progreso reanudable en `experimentos/resultados/progreso/`. Si la ejecución se
interrumpe, repite el mismo comando para continuar sin pagar de nuevo los casos
terminados. Usa `--reiniciar-progreso` solo cuando quieras descartar ese progreso
y repetir toda la evaluación.

También puede evaluarse desde Python:

```python
from agente import evaluar

tabla = evaluar(
    "golden_set.jsonl",
    salida="resultados/evaluacion_jchulvi_v3.csv",
)
print(tabla)
```

#### Comparación de resultados resumidos

La tabla siguiente compara las columnas comunes de los resúmenes del baseline y
del agente predeterminado sobre `golden_set.jsonl`:

| Agente | Cita | Cifra | Trayectoria | Recall@5 | Aciertos extractiva | Aciertos numérica | Aciertos comparativa | Coste medio (¢) | Latencia media (s) | Llamadas/pregunta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 42,9 % | 85,7 % | 75,0 % | 42,9 % | 5/6 | 6/6 | 0/8 | 0,7592 | 14,87 | 3,10 |
| Jchulvi v2 | 100,0 % | 100,0 % | 100,0 % | 92,9 % | 6/6 | 6/6 | 8/8 | 0,1376 | 25,91 | 3,10 |

Los aciertos por familia del baseline se han reconstruido desde
[`resultados/baseline.csv`](resultados/baseline.csv), ya que su resumen histórico no
incluía esas columnas. La versión actual de `--resumir` sí las calcula.

Los CSV resumidos utilizados como fuente son:

- [`resultados/resumen_baseline.csv`](resultados/resumen_baseline.csv)
- [`resultados/resumen_jchulvi_v2.csv`](resultados/resumen_jchulvi_v2.csv)

Pueden regenerarse a partir de sus evaluaciones detalladas mediante la operación
`--resumir` de la CLI:

```bash
python -m agente \
  --resumir resultados/baseline.csv \
  --etiqueta baseline \
  --salida resultados/resumen_baseline.csv

python -m agente \
  --resumir resultados/evaluacion_jchulvi_v2.csv \
  --etiqueta jchulvi-v2 \
  --salida resultados/resumen_jchulvi_v2.csv
```

Las evaluaciones detalladas de entrada están en
[`resultados/baseline.csv`](resultados/baseline.csv) y
[`resultados/evaluacion_jchulvi_v2.csv`](resultados/evaluacion_jchulvi_v2.csv).
Si se omite `--salida`, el resumen se muestra por pantalla sin escribir ningún
archivo.

### 7. Docker y solución de problemas

No hay que ejecutar `docker run` manualmente: el agente crea, valida, carga y
detiene su propio contenedor. Si el proceso se cerró de manera forzada y una
ejecución posterior informa de que el contenedor sigue en uso, detenlo sin
borrar sus datos y vuelve a ejecutar el agente:

```bash
docker stop taller-nlp-jchulvi-v002
```

- **`docker: command not found`**: instala Docker y abre una terminal nueva.
- **Cannot connect to the Docker daemon**: inicia Docker Desktop o el servicio
  de Docker y comprueba de nuevo `docker version`.
- **Error 401/403 de OpenRouter**: verifica que `OPENROUTER_API_KEY` esté
  exportada en la terminal actual y que la clave sea válida.
- **Error 402/429 de OpenRouter**: revisa el saldo, los límites de la clave y
  los límites de peticiones. El progreso parcial de una evaluación queda
  guardado.
- **El puerto 6339 está ocupado**: detén el proceso o contenedor que lo utiliza
  antes de volver a ejecutar el agente.

Para verificar la instalación sin realizar llamadas al modelo, instala el extra
de pruebas y ejecuta:

```bash
python -m pytest
```
