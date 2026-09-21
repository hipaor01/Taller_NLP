# Agente financiero sobre informes SEC 10-K

Proyecto del Taller de NLP dedicado a construir y evaluar un agente que responde
preguntas sobre informes anuales de empresas, combinando consultas financieras
exactas con búsqueda de evidencia textual.

El agente utiliza herramientas para consultar datos XBRL, localizar fragmentos
de los informes y generar respuestas estructuradas con fuentes verificables.
Su evaluación considera tanto la respuesta como el procedimiento utilizado
para obtenerla.

## Objetivos

- Responder preguntas numéricas, extractivas y comparativas sobre informes 10-K.
- Seleccionar la herramienta adecuada para cada consulta y reconocer cuándo falta información.
- Acompañar las respuestas de cifras, unidades y citas que permitan verificar su procedencia.
- Medir el efecto de los filtros, la búsqueda híbrida y la reescritura de consultas sobre la recuperación.
- Comparar las versiones del agente en calidad, coste y latencia mediante evaluaciones reproducibles.

## Datos

El corpus incluye **seis empresas** —NVIDIA, Microsoft, Apple, Alphabet, Meta y
Amazon— y los ejercicios fiscales **2024 y 2025**. Contiene **48 secciones**,
**1.749 fragmentos de texto** y **135 hechos XBRL**, con información sobre
riesgos, comentarios de la dirección y estados financieros.

El conjunto propio reúne **20 preguntas**, incluidas **8 comparativas**.
Se dispone además de un conjunto oficial de 20 preguntas para validar las
soluciones.

## Estructura del repositorio

```text
├── agente/                 # Interfaz pública: responder() y evaluar()
├── src/taller_nlp/         # Motor, herramientas, recuperación y evaluadores
├── corpus/                 # Informes, fragmentos, hechos XBRL e índice
├── experimentos/           # Baseline y variantes del equipo
├── resultados/             # Evaluaciones y tablas comparativas
├── tests/                  # Pruebas del proyecto
├── golden_set*.jsonl       # Conjuntos de preguntas de evaluación
├── S1_*.ipynb              # Herramientas y bucle del agente
├── S2_*.ipynb              # Robustez y evaluación
└── pyproject.toml          # Dependencias y configuración
```

## Resultados destacados

Los tres integrantes han registrado ejecuciones de **20/20 en el conjunto
propio**, frente al **11/20 del baseline**. Se muestran las mejores variantes
medidas de cada integrante, incluyendo los empates de Higinio y Hugo, y la
validación final de jchulvi. Las tablas distinguen el conjunto propio, el oficial
y los casos de estrés.

### Calidad en el conjunto propio

20 preguntas: **6 extractivas, 6 numéricas y 8 comparativas**.

| Agente | Extractivas | Numéricas | Comparativas | Total | Cita | Cifra | Trayectoria | Recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 5/6 | 6/6 | 0/8 | 11/20 | 42,9 % | 85,7 % | 75,0 % | 42,9 % |
| Higinio v014 | 6/6 | 6/6 | 8/8 | **20/20** | 100,0 % | 100,0 % | 100,0 % | 78,6 % |
| Higinio v015 | 6/6 | 6/6 | 8/8 | **20/20** | 100,0 % | 100,0 % | 100,0 % | 92,9 % |
| Hugo v003 | 6/6 | 6/6 | 8/8 | **20/20** | 100,0 % | 100,0 % | — | 64,3 % |
| Hugo v004 | 6/6 | 6/6 | 8/8 | **20/20** | 100,0 % | 100,0 % | — | 57,1 % |
| Hugo v005 | 6/6 | 6/6 | 8/8 | **20/20** | 100,0 % | 100,0 % | — | 57,1 % |
| jchulvi v2 | 6/6 | 6/6 | 8/8 | **20/20** | 100,0 % | 100,0 % | 100,0 % | 92,9 % |

Cita y cifra se calculan sobre las preguntas donde aplica cada comprobación;
trayectoria, sobre las 20 preguntas. Recall@5 mide la recuperación del ancla
textual en las 14 preguntas que la tienen. Una respuesta puede citar otro
pasaje válido y acertar sin recuperar esa ancla. `—` indica una métrica que la
fuente consultada no desglosa; no equivale a cero.

### Coste y rendimiento en el conjunto propio

| Agente | Coste medio (USD) | Coste de 20 preguntas (USD) | Latencia media (s) | Herramientas por pregunta | Errores de ejecución |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0,007592 | 0,15184 | 14,87 | 3,10 | 2 |
| Higinio v014 | 0,008494 | 0,16987 | 19,91 | 3,15 | 0 |
| Higinio v015 | 0,008208 | 0,16415 | 16,15 | 2,85 | 0 |
| Hugo v003 | ≈ 0,0339 | ≈ 0,678 | 12,7 | 2,70 | — |
| Hugo v004 | ≈ 0,0347 | ≈ 0,694 | 12,3 | 2,75 | — |
| Hugo v005 | ≈ 0,0342 | ≈ 0,684 | 12,0 | 2,75 | — |
| jchulvi v2 | 0,001316 | 0,02632 | 19,63 | 2,95 | 0 |

Los importes están expresados en **USD**, convertidos desde céntimos cuando la
fuente los publica así. El coste total es la media por 20 preguntas. `≈` señala
valores publicados con redondeo y totales derivados de ellos. Las llamadas
corresponden a **herramientas**, no a turnos del modelo.

### Validación en el conjunto oficial

20 preguntas: **6 extractivas, 7 numéricas y 7 comparativas**.

| Agente | Extractivas | Numéricas | Comparativas | Total | Recall@5 | Coste medio (USD) | Latencia media (s) | Herramientas por pregunta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hugo v002 | 6/6 | 7/7 | 7/7 | **20/20** | 92,3 % | ≈ 0,0321 | 12,8 | 2,70 |
| Hugo v005 | 6/6 | 7/7 | 7/7 | **20/20** | 84,6 % | ≈ 0,0388 | 13,5 | 2,80 |
| jchulvi v2 | 6/6 | 7/7 | 7/7 | **20/20** | 100,0 % | 0,001051 | 23,85 | 2,80 |

Las tres ejecuciones registran **100 % en cita y cifra**. El recall se mide sobre
13 preguntas con ancla. No se ha localizado una evaluación oficial de Higinio
en los archivos revisados. Hugo v002 se incluye por su 20/20 oficial, aunque
obtiene 14/20 en el propio. jchulvi v2 suma **40/40**, **28/28 citas completas
verificadas**, cero errores y **0,04733 USD** de coste del agente entre ambos
conjuntos.

### Configuración de las variantes

| Integrante | Variantes destacadas | Modelo | Recuperación | Controles propios |
| --- | --- | --- | --- | --- |
| Higinio | [v014](experimentos/higinio/agente_v014.py), [v015](experimentos/higinio/agente_v015.py) | Gemini 3.8 Flash, configuración heredada del baseline | BGE-small; híbrida BM25 + densa, RRF y filtros de metadatos | Verificación XBRL, guardrail comparativo, reparación de citas y contrato comparativo tolerante |
| Hugo | [v003](experimentos/hugo/agente_v003.py), [v004](experimentos/hugo/agente_v004.py), [v005](experimentos/hugo/agente_v005.py) | Claude Sonnet 5, API de Anthropic | BGE-small; densa con filtros de metadatos | v003: instrucciones comparativas; v004: añade comprobador XBRL; v005: añade reparador de citas |
| jchulvi | [v2](experimentos/jchulvi/agente_v2.py) | DeepSeek V4 Flash 0731, OpenRouter/DeepInfra | Voyage 4 Lite, 1.024 dimensiones; Qdrant, filtros y contexto de sección | Salida estructurada e instrucciones; sin middlewares propios; caché de vectores del corpus |

La configuración de Higinio se obtiene del código y puede cambiarse por entorno;
los CSV no conservan el identificador del modelo de cada ejecución. Los modelos
de Hugo constan en sus tablas publicadas y el de jchulvi en el manifiesto de la
campaña. Higinio v015 mejora el recall, coste y latencia observados de v014
manteniendo el 20/20. Hugo v003, v004 y v005 empatan en el propio: la prueba de
estrés permite distinguir su comportamiento fuera de esas preguntas.

### Casos de estrés documentados

Ocho preguntas adicionales de Hugo: tres de beneficio por acción, tres sin
respuesta en el corpus y dos extractivas con apóstrofos tipográficos.

| Agente | Beneficio por acción | Datos ausentes | Citas tipográficas | Total | Coste total (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hugo v003 | 0/3 | 3/3 | 2/2 | 5/8 | ≈ 0,19 |
| Hugo v005 | **3/3** | 3/3 | 2/2 | **8/8** | ≈ 0,28 |

El comprobador XBRL corrigió tres cifras redondeadas. No consta esta misma
prueba para Higinio ni jchulvi; estos ocho casos no se suman a las notas de los
conjuntos de 20 preguntas.

### Fuentes y lectura de los resultados

- **Baseline y Higinio:** [CSV del baseline](resultados/baseline.csv),
  [v014](resultados/agente_v014.csv), [v015](resultados/agente_v015.csv) y
  [resúmenes](resultados/). Aciertos recalculados exigiendo todas las
  comprobaciones aplicables y ausencia de error.
- **Hugo:** [tablas de evaluación y estrés](experimentos/hugo/README.md).
  Los CSV de esas ejecuciones no están disponibles en este checkout; se conserva
  la precisión publicada. Su README recoge otra pasada del baseline, por lo que
  aquí se utiliza únicamente el baseline del CSV enlazado.
- **jchulvi:** [notebook de validación ejecutado](experimentos/jchulvi/validacion_agente_v2.ipynb),
  campaña `v2_simple_6944a9ccd5b5f92e`. Métricas contrastadas con sus manifiestos.

Son ejecuciones con distintos modelos y configuraciones, no una comparación
controlada del efecto de un único cambio. El baseline utiliza tarifas docentes,
Hugo estima el coste por tokens y jchulvi registra el coste del proveedor,
**sin incluir embeddings de consulta**. Las latencias dependen de las condiciones
de cada ejecución; no se dispone de repeticiones suficientes para estimar su
variabilidad.

El evaluador de citas comprueba coincidencia literal, no el respaldo semántico
completo de la respuesta. Los conjuntos propio y oficial son conocidos durante
el desarrollo: **20/20 no garantiza el mismo resultado en preguntas nuevas**.
La evaluación ciega de la práctica queda pendiente.

Los [notebooks](S2_Robustez_y_Evaluacion_Alumno.ipynb) y la documentación de
[experimentos](experimentos/README.md) recogen el procedimiento de evaluación
y el detalle de las variantes.
