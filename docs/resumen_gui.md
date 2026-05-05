# Resumen de la GUI

## Idea general

La GUI de Onitama tiene un centro claro: `MainWindow`.

Desde ahi se controla todo:
- la pantalla de configuracion inicial,
- la pantalla de juego,
- la calibracion,
- la ventana auxiliar de camara,
- y el hilo que ejecuta la vision y la logica de partida.

## Papel de cada archivo

### `src/onitama/gui/vision_app.py`
Punto de entrada de la interfaz.

Su trabajo es:
- crear `QApplication`,
- crear `MainWindow`,
- mostrarla,
- entrar en el bucle de eventos de Qt.

### `src/onitama/gui/main_window.py`
Es el orquestador de toda la GUI.

Se encarga de:
- montar la ventana principal,
- alternar entre setup y juego con `QStackedWidget`,
- abrir y cerrar ventanas auxiliares,
- arrancar y parar el `RuntimeWorker`,
- recibir estados nuevos del runtime,
- actualizar tablero, cartas y mensajes.

Es el archivo principal del flujo de la interfaz.

### `src/onitama/gui/setup_page.py`
Es la pantalla de configuracion inicial embebida dentro de `MainWindow`.

Permite:
- elegir color del humano,
- elegir dificultad de la IA,
- lanzar calibracion de tablero,
- lanzar calibracion de cartas,
- comprobar el estado de calibracion,
- iniciar la partida.

No ejecuta logica de vision. Solo muestra opciones y emite señales.

### `src/onitama/gui/runtime_worker.py`
Es un `QThread` que ejecuta el runtime real sin bloquear la interfaz.

Se encarga de:
- crear `VisionGameRuntime`,
- llamar repetidamente a `runtime.step()`,
- emitir el estado actualizado a la GUI,
- emitir frames de camara si el visor de camara esta abierto.

Es el puente entre la GUI y la capa de vision/integracion.

### `src/onitama/gui/view_logic.py`
Traduce el estado interno del runtime a mensajes claros para la interfaz.

Por ejemplo:
- turno del humano,
- turno de la IA,
- lectura invalida,
- movimiento rechazado,
- partida terminada.

Su objetivo es que `MainWindow` no tenga toda la logica textual mezclada.

### `src/onitama/gui/widgets.py`
Contiene widgets visuales reutilizables.

Ahora mismo incluye:
- `BoardWidget`: dibuja el tablero y las piezas,
- `CardWidget`: dibuja una carta,
- `MessageBanner`: muestra el mensaje superior de estado.

No controla el flujo de la app. Solo representa datos.

### `src/onitama/gui/camera_window.py`
Es una ventana auxiliar independiente.

Sirve para:
- mostrar el frame de camara en directo,
- reescalarlo para que se vea bien.

No toma decisiones de vision ni de juego. Solo visualiza.

### `src/onitama/gui/calibration/board_calibration_dialog.py`
Dialogo para calibrar el tablero.

Permite:
- capturar una imagen,
- marcar las 4 esquinas,
- guardar `data/vision/calibration.json`.

### `src/onitama/gui/calibration/card_rois_calibration_dialog.py`
Dialogo para calibrar las cartas.

Permite:
- capturar una imagen,
- seleccionar cada slot de carta,
- marcar sus 4 puntos,
- guardar `data/vision/card_rois.json`.

### `src/onitama/gui/calibration/calibration_common.py`
Codigo compartido por la calibracion.

Incluye:
- captura de camara,
- visor interactivo con zoom y desplazamiento,
- dialogo base,
- helpers de dibujo de overlays.

## Flujo normal de partida

### 1. Arranque
- se ejecuta `vision_app.py`,
- se crea `QApplication`,
- se crea `MainWindow`,
- `MainWindow` muestra `SetupPage`.

### 2. Setup
En `SetupPage`, el usuario:
- elige color,
- elige dificultad,
- calibra si hace falta,
- pulsa `Iniciar partida`.

`MainWindow` comprueba antes que existan y sean validos:
- `data/vision/calibration.json`
- `data/vision/card_rois.json`

Si falta algo, no deja empezar.

### 3. Calibracion
Si el usuario pulsa calibrar:
- `SetupPage` emite una señal,
- `MainWindow` abre el dialogo correspondiente,
- el dialogo guarda su JSON,
- `MainWindow` refresca el estado de calibracion.

### 4. Inicio de la partida
Cuando el usuario pulsa `Iniciar partida`:
- `MainWindow` crea un `VisionRuntimeConfig`,
- crea `RuntimeWorker`,
- conecta sus señales,
- cambia a la pantalla de juego,
- arranca el hilo.

### 5. Ejecucion del runtime
`RuntimeWorker`:
- crea `VisionGameRuntime`,
- llama a `step()` en bucle,
- recibe `VisionRuntimeState`,
- lo manda a `MainWindow`.

### 6. Actualizacion visual
Cada vez que llega un estado:
- `MainWindow` llama a `_apply_state()`,
- actualiza `BoardWidget`,
- actualiza `CardWidget`,
- actualiza `MessageBanner` usando `build_status_view()`.

### 7. Ventana de camara
Si el usuario pulsa `Camara`:
- `MainWindow` abre `CameraWindow`,
- pide a `RuntimeWorker` que emita frames,
- cada frame recibido se manda a `CameraWindow`.

### 8. Reinicio o fin
- `Reiniciar` pide al runtime que se reinicie.
- `Finalizar partida` detiene el worker y vuelve al setup.
- al cerrar la app, `MainWindow` intenta parar el worker y cerrar dialogos abiertos.

## Resumen corto de responsabilidades

- `vision_app.py`: entrada
- `main_window.py`: coordinacion general
- `setup_page.py`: configuracion inicial
- `runtime_worker.py`: hilo de ejecucion
- `view_logic.py`: mensajes de estado
- `widgets.py`: renderizado reutilizable
- `camera_window.py`: visor auxiliar de camara
- `gui/calibration/`: calibracion integrada
