# Agente financiero sobre informes SEC 10-K

## Puesta en marcha desde un clon limpio

Los pasos siguientes dejan listo el agente predeterminado,
`experimentos.jchulvi.agente_v3`, tanto para responder una pregunta como para
evaluar un conjunto JSONL. Su código está en
[`experimentos/jchulvi/agente_v3.py`](experimentos/jchulvi/agente_v3.py).
Ejecuta todos los comandos desde la raíz del repositorio.

### Ubicación del código principal

`agente_v3.py` conserva los contratos comunes de las cuatro herramientas y
reutiliza la composición previa de `search_filings` y las descripciones
entregadas al modelo:

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
| Jchulvi v3 | 100,0 % | 100,0 % | 100,0 % | 85,7 % | 6/6 | 6/6 | 8/8 | 0,1234 | 16,34 | 2,80 |

Los aciertos por familia del baseline se han reconstruido desde
[`resultados/baseline.csv`](resultados/baseline.csv), ya que su resumen histórico no
incluía esas columnas. La versión actual de `--resumir` sí las calcula.

Los CSV resumidos utilizados como fuente son:

- [`resultados/resumen_baseline.csv`](resultados/resumen_baseline.csv)
- [`resultados/resumen_jchulvi_v3.csv`](resultados/resumen_jchulvi_v3.csv)

Los resultados de la evaluación sobre el holdout están disponibles en:

- [Evaluación detallada del holdout](resultados/evaluacion_holdout.csv)
- [Resumen de la evaluación del holdout](resultados/resumen_holdout.csv)

#### Respuestas del holdout

Las siguientes respuestas finales se han extraído del
[progreso de la evaluación](experimentos/resultados/progreso/agente_v3_holdout_784f694b04ea_citas-v5.json)
y se presentan junto a las preguntas de `holdout.jsonl`:

1. **ho-001 — ¿Qué dice NVIDIA en su 10-K de FY2024 sobre las garantías de
   suministro de obleas y componentes?**

   En su 10-K de FY2024, NVIDIA reconoce que NO dispone de garantías de
   suministro de obleas, componentes ni capacidad. La compañía depende de
   fundiciones y subcontratistas de terceros para fabricar sus obleas de
   semiconductores y ensamblar sus productos, y señala que no se le garantiza
   el suministro de obleas, componentes ni capacidad, lo que reduce su control
   sobre la cantidad, calidad, rendimientos de fabricación y calendarios de
   entrega. También advierte del riesgo de falta de suministro garantizado, de
   precios más altos de obleas y componentes, y de la dependencia de un número
   limitado y geográficamente concentrado de proveedores y fundiciones.

2. **ho-002 — ¿Cuáles son las principales exposiciones a divisas de Alphabet
   según el apartado de riesgo de mercado de su 10-K de 2025?**

   Según el apartado de riesgo de mercado (Item 7A) del 10-K de 2025 de
   Alphabet, las principales exposiciones a divisas son las fluctuaciones de
   los tipos de cambio frente al dólar estadounidense en el dólar australiano,
   la libra esterlina, el dólar canadiense, el euro y el yen japonés. Estas
   exposiciones provienen de los ingresos internacionales, los activos y
   pasivos monetarios denominados en moneda extranjera y las inversiones en
   filiales extranjeras, y se gestionan mediante instrumentos derivados y no
   derivados (forwards, opciones, swaps de divisas cruzadas y deuda en moneda
   extranjera).

3. **ho-003 — ¿Qué ocurrió con las ventas de Apple en la Gran China en 2025 y
   a qué se debió?**

   Las ventas netas de Apple en la Gran China disminuyeron un 4% en 2025,
   pasando de 66.952 millones de USD en 2024 a 64.377 millones de USD en 2025.
   Esta caída se debió principalmente a menores ventas de iPhone, compensadas
   parcialmente por mayores ventas de Mac.

4. **ho-004 — ¿Cuál era el activo total de Microsoft al cierre del ejercicio
   fiscal 2024?**

   El activo total de Microsoft al cierre del ejercicio fiscal 2024 fue de
   512.163 millones de dólares (512.163.000.000 USD).

5. **ho-005 — ¿Cuál fue el beneficio bruto de Amazon en 2025?**

   Amazon no reportó el concepto de beneficio bruto (`GrossProfit`) en su
   informe 10-K del ejercicio 2025. El dato solicitado no está disponible en
   el corpus.

6. **ho-006 — ¿Cuál fue el revenue de NVIDIA en el ejercicio fiscal 2023?**

   El ejercicio fiscal 2023 de NVIDIA no está disponible en el corpus (solo
   existen los ejercicios 2024 y 2025), por lo que no se puede reportar el
   revenue solicitado.

7. **ho-007 — ¿Cuánto crecieron los ingresos de Apple entre FY2024 y FY2025, y
   qué explica la evolución del iPhone?**

   Los ingresos (net sales) de Apple crecieron de 391.035 millones de USD en
   FY2024 a 416.161 millones de USD en FY2025, un aumento de 25.126 millones de
   USD (~6,4%). En cuanto al iPhone, sus ventas netas pasaron de 201.183 a
   209.586 millones de USD (+4%), y la compañía explica que el incremento se
   debió a mayores ventas de los modelos Pro.

8. **ho-008 — ¿Cuánto aumentó el gasto en I+D de Alphabet entre 2024 y 2025, y
   qué advierte la compañía sobre la relación entre el gasto en compensación y
   la plantilla?**

   El gasto en I+D de Alphabet pasó de 49.326 millones de USD en 2024 a 61.087
   millones de USD en 2025, lo que supone un aumento de 11.761 millones de USD
   (unos 11,8 mil millones, +24%). En cuanto a la relación entre compensación y
   plantilla, la compañía advierte que las fluctuaciones en los gastos de
   compensación de los empleados pueden no correlacionarse directamente con
   los cambios en la plantilla, debido a factores como las concesiones anuales
   de SBC (stock-based compensation) que se devengan a lo largo del tiempo.

9. **ho-009 — ¿Cómo cambió el beneficio bruto de NVIDIA entre FY2024 y FY2025,
   y qué margen bruto reportó en cada ejercicio?**

   El beneficio bruto (gross profit) de NVIDIA pasó de 44.301 millones de USD
   en FY2024 a 97.858 millones de USD en FY2025, lo que supone un aumento de
   53.557 millones de USD (más del doble). En cuanto al margen bruto, NVIDIA
   reportó un 72,7% en FY2024 y un 75,0% en FY2025, un incremento de 2,3 puntos
   porcentuales, impulsado principalmente por una mayor proporción de ingresos
   de Data Center.

10. **ho-010 — ¿Cómo evolucionó el beneficio operativo de Microsoft entre
    FY2024 y FY2025, y qué dice la dirección sobre sus segmentos?**

    El beneficio operativo (operating income) de Microsoft pasó de 109.433
    millones de USD en FY2024 a 128.528 millones de USD en FY2025, un aumento
    de 19.095 millones de USD, equivalente a un 17%. La dirección atribuye el
    crecimiento a los tres segmentos reportables: Productivity and Business
    Processes (operating income de 69.773 M USD, +17%), Intelligent Cloud
    (44.589 M USD, +18%, impulsado por Azure) y More Personal Computing (14.166
    M USD, +18%, impulsado por Gaming y Search and news advertising). La
    dirección señala que el margen bruto de Microsoft Cloud bajó al 69% por el
    impacto de escalar la infraestructura de IA, compensado en parte por
    ganancias de eficiencia en Azure, y que los gastos operativos aumentaron
    un 6% por las inversiones en ingeniería de cloud e IA y por Gaming,
    incluido el efecto de la adquisición de Activision Blizzard.

Pueden regenerarse a partir de sus evaluaciones detalladas mediante la operación
`--resumir` de la CLI:

```bash
python -m agente \
  --resumir resultados/baseline.csv \
  --etiqueta baseline \
  --salida resultados/resumen_baseline.csv

python -m agente \
  --resumir resultados/evaluacion_jchulvi_v3.csv \
  --etiqueta jchulvi-v3 \
  --salida resultados/resumen_jchulvi_v3.csv
```

Las evaluaciones detalladas de entrada están en
[`resultados/baseline.csv`](resultados/baseline.csv) y
[`resultados/evaluacion_jchulvi_v3.csv`](resultados/evaluacion_jchulvi_v3.csv).
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
