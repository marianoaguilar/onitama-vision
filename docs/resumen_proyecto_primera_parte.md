# Resumen del Proyecto: Motor de Onitama con IA

## 1. Introduccion
Este documento resume el trabajo implementado hasta la fecha en el proyecto **onitama-vision**.  
El objetivo principal es construir un motor de juego de Onitama en Python, con reglas correctas, estado inmutable, interfaz de consola y agentes de IA basados en busqueda alfa-beta.

El proyecto esta orientado a dos usos:
- Uso interactivo (partidas por terminal entre humano y/o IA).
- Uso experimental (torneos y benchmarks para comparar heuristicas y configuraciones de busqueda).

## 2. Objetivos tecnicos del proyecto
- Implementar las reglas de Onitama de forma fiel.
- Garantizar transiciones de estado puras (sin mutaciones no controladas).
- Separar claramente motor, IA y capa de presentacion (CLI).
- Incorporar una IA configurable y extensible.
- Permitir comparacion reproducible entre evaluadores mediante scripts de experimentacion.

## 3. Estructura general del repositorio
Estructura principal:
- `src/onitama/engine/`: logica del juego, reglas, estado y tipos base.
- `src/onitama/ai/`: evaluacion heuristica, busqueda y controlador IA.
- `src/onitama/cli/`: interfaz de linea de comandos.
- `scripts/`: herramientas para torneos y benchmarks.
- `tests/`: pruebas de regresion y consistencia.
- `pyproject.toml`: metadatos del proyecto y version minima de Python (>=3.10).

## 4. Motor de juego (`onitama.engine`)
### 4.1 Modelo de datos
Tipos base:
- `Player`: `RED` y `BLUE`.
- `PieceType`: `MASTER` y `STUDENT`.
- `Piece`: pieza con `owner` y `kind`.
- `Move`: accion con origen, destino e indice de carta usada.
- `Pass`: accion especial cuando no hay movimientos legales, pero se elige carta (indice 0 o 1).

### 4.2 Estado inmutable
`GameState` (en `state.py`) contiene:
- Tablero 5x5.
- Jugador al turno (`to_move`).
- Cartas de RED, cartas de BLUE y carta lateral (`side_card`).
- Cache de posicion de maestro rojo y azul (`red_master_pos`, `blue_master_pos`).

Detalles relevantes:
- En `__post_init__`, el tablero se normaliza a `tuple` de `tuple` para hacerlo hashable e inmutable.
- Si no se pasan posiciones de maestros, se reconstruyen escaneando tablero.
- `GameState.initial(seed)` crea estado inicial completo:
  - Seleccion aleatoria reproducible de 5 cartas entre 16 oficiales.
  - 2 cartas para RED, 2 para BLUE y 1 lateral.
  - El jugador inicial lo marca el sello (`stamp`) de la carta lateral.

### 4.3 Cartas y orientacion
En `cards.py`:
- Se modelan las 16 cartas del juego base.
- Los deltas se definen desde perspectiva RED.
- Para BLUE se aplica rotacion de 180 grados (`dr, dc -> -dr, -dc`).

### 4.4 Reglas y transiciones
En `rules.py`:
- `generate_legal_actions(state)`:
  - Genera movimientos legales segun cartas activas del jugador al turno.
  - Descarta salidas del tablero y capturas de pieza propia.
  - Si no hay `Move`, devuelve exactamente `[Pass(0), Pass(1)]`.
- `apply_action(state, action)`:
  - Aplica `Move` o `Pass` y devuelve un **nuevo** `GameState`.
  - Cambia turno al oponente.
  - Ejecuta intercambio de cartas segun `card_index`.
  - Actualiza cache de maestros correctamente en movimiento/captura.
- `winner(state)`:
  - Victoria por captura de maestro ("Capture of Master").
  - Victoria por llegada al templo rival ("Reach Temple").
- `is_terminal(state)` verifica si la partida es terminal.

## 5. Capa de IA (`onitama.ai`)
### 5.1 Evaluadores heuristicos
Archivo: `evaluate.py`.

Se define `WIN_SCORE = 100000` para estados terminales y tres evaluadores registrados:
- `v1`: base (material, progreso del maestro, amenaza al maestro, movilidad opcional, estudiantes colgando).
- `v2`: version parametrizable con pesos (incluye bonus por amenazar maestro rival y rasgos posicionales de centralidad/avance).
- `v3`: evolucion con mezcla por fase de partida (opening/endgame), movilidad y progreso ponderados por fase, y PST (piece-square tables) para estudiantes y maestro.

Registro central:
- `EVALUATORS = {"v1": ..., "v2": ..., "v3": ...}`
- `get_evaluator(name)` valida y devuelve el evaluador solicitado.

### 5.2 Busqueda
Archivo: `search.py`.

Se implementa:
- Negamax con poda alfa-beta (`alphabeta`).
- Quiescence search (`quiescence`) para extender en capturas.
- Tabla de transposiciones (TT) con banderas `EXACT`, `LOWER`, `UPPER`.
- Ordenacion de movimientos combinando:
  - Mejor accion TT.
  - Killer moves.
  - Capturas.
  - History heuristic.

### 5.3 Seleccion de accion
Archivo: `agent.py`.

`choose_action(...)` ofrece:
- Profundidad configurable.
- Iterative deepening opcional.
- Aspiration window opcional.
- Activacion de TT (local o persistente si se comparte).
- `q_depth` configurable para quiescence.

Tambien ordena acciones de raiz por prioridad tactica:
- Ganancia inmediata.
- Captura.

### 5.4 Controlador IA
Archivo: `ai/controllers.py`.

`AIController` encapsula:
- Profundidad.
- Nombre de evaluador.
- TT persistente por controlador.

## 6. Interfaz de consola (`onitama.cli`)
### 6.1 Flujo principal
Archivo: `cli/play.py`.

Permite los 4 modos:
- Human vs Human.
- Human (RED) vs AI (BLUE).
- AI (RED) vs Human (BLUE).
- AI vs AI.

En modo IA, se configuran profundidad y evaluador por jugador.

### 6.2 Controlador humano
Archivo: `cli/controllers.py`.

Caracteristicas:
- Lista acciones legales numeradas.
- Comandos de control: `help`, `quit`, `restart`.
- Seleccion robusta de entrada por rango.

### 6.3 Renderizado
Archivo: `cli/render.py`.

Incluye:
- Render del tablero en texto.
- Conversor de coordenadas internas a formato tipo ajedrez (`a1..e5`).
- Formateo legible de acciones (incluyendo `PASS` con carta asociada).

## 7. Scripts de experimentacion (`scripts/`)
### 7.1 Torneos
Archivo: `scripts/ai/tournament.py`.

Modos:
- `match`: enfrentamiento A vs B con semillas pareadas y cambio de color (justicia experimental).
- `roundrobin`: todos contra todos entre varios agentes.

Funcionalidades:
- Limite de plies para declarar tablas.
- Export a CSV por partida.
- Export a JSON con resumen agregado.
- Registro opcional del commit git para trazabilidad.
- Parametros de busqueda (TT, iterative deepening, aspiration window, q-depth).

### 7.2 Benchmark de busqueda
Archivo: `scripts/ai/bench_search.py`.

Objetivo:
- Medir coste temporal de `choose_action` en estados de medio juego.

Incluye:
- Modos de TT (`none`, `local`, `persistent`).
- Warmup + runs.
- Estadisticos de tiempo (`avg`, `min`, `max`, `stdev`).
- Modo comparativo entre configuraciones.

## 8. Pruebas implementadas (`tests/`)
Cobertura actual de regresion:
- `test_initial_state.py`: despliegue inicial y unicidad de cartas.
- `test_apply_action_purity.py`: `apply_action` no muta el estado original.
- `test_cards_orientation.py`: rotacion correcta de deltas para BLUE.
- `test_pass_action.py`: contrato de `Pass` cuando no hay movimientos.
- `test_winner.py`: deteccion de victoria por captura y por templo.
- `test_card_swap.py`: intercambio de cartas correcto tras `Pass` y tras `Move`.

## 9. Decisiones de diseno destacables
- Inmutabilidad del estado para seguridad, depuracion y uso de TT.
- Separacion por capas (engine, ai, cli) para facilitar mantenimiento y extension.
- Evaluadores versionados y registrados por nombre para experimentacion sistematica.
- Infraestructura de torneos reproducibles con control de semillas y swap de colores.

## 10. Limitaciones actuales
- Cobertura de tests centrada en reglas base; se puede ampliar hacia casos tacticos mas complejos.
- No hay interfaz grafica; la interaccion es exclusivamente por terminal.
- Ajuste de pesos heuristicos mayoritariamente manual (potencial de tuning automatizado).
- No se explotan paralelizacion ni tecnicas avanzadas tipo MCTS/NN.

## 11. Lineas de trabajo futuro
- Ampliar suite de tests (propiedades, escenarios tacticos, regresion de bugs).
- Afinar o aprender pesos de evaluacion con experimentacion automatica.
- Instrumentar metricas de nodos/tiempo por profundidad durante busqueda.
- Incorporar interfaz visual (web o escritorio) sobre el motor existente.
- Analizar mejoras de rendimiento adicionales y comparar CPython vs PyPy en mas escenarios.

## 12. Comandos utiles para demostracion
Ejecucion de tests:
```bash
.venv/bin/pytest -q
```

Juego por CLI:
```bash
python -m onitama.cli.play
```

Ayuda del torneo:
```bash
python scripts/ai/tournament.py --help
```

Benchmark de busqueda:
```bash
python scripts/ai/bench_search.py --help
```
