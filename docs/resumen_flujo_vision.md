# Resumen del flujo de vision

## Arquitectura

La capa `vision` ahora esta separada en piezas claras:

#### `homography.py`

- geometria del tablero
- warp y rotacion
- mapeo de coordenadas a celdas

#### `piece_detector.py`

- deteccion de piezas con YOLO
- conversion de detecciones a `VisionBoard`

#### `card_rois.py`

- carga y guardado de ROIs de cartas
- recorte de las 5 cartas

#### `card_classifier.py`

- preprocesado a `320x320` con padding negro
- clasificacion de las 5 cartas
- salida como `CardClassificationResult`

#### `board.py`

- representacion visual discreta del tablero: `VisionBoard`

#### `snapshot.py`

- observacion visual completa: tablero + cartas
- serializacion
- conversion a `GameState`

#### `vision_pipeline.py`

- orquestador del flujo completo desde un frame

## Flujo de piezas

1. Llega un `frame`.
2. `YoloPieceDetector` usa `homography.py` para:
   - hacer warp del tablero
   - aplicar la rotacion calibrada
3. YOLO detecta piezas sobre esa imagen canonica.
4. Cada deteccion se convierte en `PieceDetection`.
5. Se calcula una celda `(row, col)` usando el anchor de la deteccion.
6. `detections_to_board(...)` resuelve conflictos por celda y produce un `VisionBoard`.

## Flujo de cartas

1. Llega el mismo `frame`.
2. `card_rois.py` recorta las 5 cartas usando ROIs calibrados.
3. `card_classifier.py` prepara cada crop a `320x320`.
4. YOLO de clasificacion predice una carta por slot.
5. El resultado queda en `CardClassificationResult`.
6. `cards_layout()` devuelve:
   - `red_cards`
   - `blue_cards`
   - `side_card`

## Convergencia

Los dos flujos se unen en `vision_pipeline.py`:

#### `snapshot_from_frame(frame)`

- detecta piezas
- clasifica cartas
- construye un `VisionSnapshot`

Ese snapshot ya representa solo observacion visual:

- `board`
- `red_cards`
- `blue_cards`
- `side_card`

## Paso al motor

Despues:

#### `snapshot.to_game_state(to_move=...)`

Aqui `to_move` entra desde fuera, porque:

- no es dato visual
- es contexto del estado de juego
- el motor lo seguira actualizando turno a turno

## Flujo completo final

La forma mental final es:

1. `frame`
2. `VisionPipeline.snapshot_from_frame(frame)`
3. `VisionSnapshot`
4. `VisionSnapshot.to_game_state(to_move=...)`
5. `GameState`

O, en una sola llamada:

1. `frame`
2. `VisionPipeline.game_state_from_frame(frame, to_move=...)`
3. `GameState`

## Decisiones importantes que quedaron cerradas

- `to_move` salio de `VisionSnapshot`
- `VisionSnapshot` ya no ejecuta inferencia desde el frame
- esa orquestacion vive en `VisionPipeline`
- piezas y cartas siguen como clases porque cargan estado, configuracion y modelos
- `bridge.py` desaparecio y la conversion al motor quedo en `VisionSnapshot.to_game_state(...)`
- se eliminaron del nucleo varios datos que eran solo debug
