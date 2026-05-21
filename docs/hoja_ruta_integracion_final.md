# Hoja de ruta para la integracion final: vision + motor + IA

## Objetivo final

El objetivo de esta fase es integrar todo lo ya construido para poder jugar una partida real de Onitama con este flujo:

1. Un jugador fisico realiza su movimiento sobre el tablero real.
2. La camara observa el tablero y las cartas.
3. La capa de vision reconstruye un estado discreto del juego.
4. El sistema valida que ese estado observado sea legal respecto al turno esperado.
5. La IA calcula su mejor jugada a partir de ese estado.
6. El sistema muestra la jugada de la IA.
7. El tablero fisico se actualiza manualmente con la jugada de la IA.
8. El sistema espera a que el tablero coincida con ese nuevo estado y continua con el siguiente turno.

La prioridad de esta fase es, por tanto, la sincronizacion robusta entre:

- estado visual observado,
- estado logico del motor,
- y flujo real de una partida por turnos.

## Punto de partida actual

El proyecto ya tiene una separacion bastante buena:

- La vision ya puede convertir `frame -> VisionSnapshot -> GameState`.
- El motor ya sabe representar, validar y avanzar estados de Onitama.
- La IA ya puede elegir acciones desde un `GameState`.
- El CLI actual ya sirve como referencia del bucle general de una partida.

En la implementacion final, esta integracion no vive en un unico archivo. Se
divide en dos capas:

- `session/`: logica pura de la partida asistida. Trabaja con estados ya
  observados, estabiliza lecturas, valida sucesores legales y gestiona fases de
  turno.
- `runtime/`: ejecucion en vivo. Abre la camara, invoca `VisionPipeline`,
  convierte snapshots segun la fase actual y expone `VisionRuntimeState` a GUI
  o CLI.

Esta separacion permite probar la logica de sincronizacion sin camara, y a la
vez mantener aislada la parte operativa del bucle en vivo.


## Idea central de la integracion

No basta con leer un `GameState` desde una imagen.

En una partida real, el sistema debe decidir si el estado observado:

- es estable,
- es coherente con el turno actual,
- y corresponde a un sucesor legal del estado anterior.

Esa es la diferencia entre una demo de vision y un sistema jugable de extremo a extremo.

## Flujo recomendado de una partida real

El flujo objetivo deberia ser este:

1. El sistema arranca con camara, calibracion y pipeline de vision listos.
2. Se obtiene un estado inicial observado del tablero.
3. Ese estado se estabiliza temporalmente para evitar ruido.
4. El sistema lo convierte en `GameState` y lo toma como estado actual.
5. Cuando es turno humano, el sistema espera a detectar un nuevo estado estable.
6. Comprueba si el estado detectado coincide con uno de los sucesores legales del estado anterior.
7. Si coincide, acepta la jugada humana y actualiza el estado interno.
8. Si no coincide, lo rechaza y sigue esperando una observacion valida.
9. Cuando es turno de la IA, se llama al motor de busqueda sobre ese estado.
10. La jugada calculada se muestra al usuario.
11. El sistema conoce el estado esperado tras la jugada de la IA.
12. Espera a que el tablero fisico coincida con ese estado esperado.
13. Una vez confirmado, avanza al siguiente turno.

## Bloques principales de la integracion

### 1. Runtime de partida con vision

La ejecucion en vivo esta concentrada en `runtime/`.

Esta capa, separada del CLI y de la GUI, es responsable de:

- capturar frames,
- invocar `VisionPipeline`,
- convertir snapshots visuales en estados candidatos,
- alimentar la sesion logica,
- gestionar errores recuperables o fatales,
- y producir un estado consumible por la interfaz.

No decide por si misma si una jugada es legal: delega esa parte en
`session/`.

### 2. Estabilizacion temporal

La estabilizacion temporal vive en `session/stabilizer.py`.

La idea minima necesaria es:

- aceptar un estado solo si se repite durante `N` frames consecutivos,
- ignorar cambios espurios por blur, oclusion o pequenas inestabilidades,
- y mas adelante, opcionalmente, congelar inferencia si aparece una mano en escena.

Sin esta capa, la integracion completa sera fragil aunque la vision individual funcione bien.

### 3. Sincronizador entre estados observados y estados legales

El sincronizador vive en `session/synchronizer.py` y es el nucleo logico de
la integracion.

El sistema debe trabajar siempre con:

- un estado anterior confirmado,
- un estado observado por vision,
- y el conjunto de sucesores legales del motor.

La regla base deberia ser:

- si el estado observado coincide con un sucesor legal del estado anterior, se acepta;
- si no, se rechaza y se sigue esperando.

Esto convierte la vision en una entrada controlada por las reglas del juego, en vez de aceptar ciegamente cualquier lectura.

### 4. Inferencia de la accion humana

Aunque el motor puede funcionar solo con estados, conviene derivar tambien que accion concreta ha realizado el jugador humano:

- que pieza se ha movido,
- a que casilla,
- si ha habido captura,
- y que carta se ha usado.

Esto aporta trazabilidad, simplifica depuracion y mejora la robustez conceptual de la integracion.

### 5. Integracion del turno de la IA

Una vez aceptado el nuevo estado del jugador humano:

- se ejecuta `choose_action(...)`,
- se renderiza o muestra la jugada elegida,
- se calcula el estado esperado tras esa accion,
- y el sistema espera a que el tablero fisico refleje ese estado.

Esta responsabilidad esta en `session/vision_session.py`: la sesion conoce el
estado confirmado, solicita la accion al controlador de IA y guarda el estado
esperado tras esa accion.

### 6. Flujo de UX minimo

El sistema final deberia manejar estados de interfaz simples pero claros:

- calibracion lista,
- esperando tablero estable,
- turno humano,
- validando jugada,
- pensando IA,
- mostrando jugada IA,
- esperando confirmacion fisica del estado esperado,
- partida terminada.

No hace falta una interfaz compleja al principio, pero si un flujo claro para no perder el control de la sesion.

En la implementacion actual, `runtime/vision_models.py` define el estado que
consume la GUI (`VisionRuntimeState`) y `gui/view_logic.py` traduce ese estado a
mensajes de interfaz.

### 7. Logging y depuracion

Esta fase necesita observabilidad desde el primer momento.

Conviene guardar:

- snapshots aceptados,
- snapshots rechazados,
- predicciones de cartas y piezas con confianza,
- motivo de rechazo de un estado,
- y transiciones confirmadas entre estados.

Sin esto, depurar errores de integracion sera lento y poco fiable.

### 8. Tests de integracion sin camara

Antes de conectar webcam en directo, conviene probar el flujo con snapshots o fixtures guardados:

- secuencias de estados validos,
- estados invalidos,
- transiciones legales e ilegales,
- y respuestas de la IA sobre estados confirmados.

La webcam en vivo debe llegar despues, no antes.

## Orden recomendado de implementacion

El orden historico recomendado para construir esta fase fue:

1. Orquestador de partida en modo offline.
2. Sincronizador de estados legales.
3. Estabilizacion temporal.
4. Integracion con IA y visualizacion de jugadas.
5. Logging y mecanismos de recovery.
6. Conexion con webcam en directo.
7. Mejoras de robustez y UX.

## Primer MVP recomendado

El primer MVP serio de la integracion se centro en construir este bloque:

- `VisionGameSession` como sesion logica principal.
- `StateStabilizer` para confirmar estados repetidos.
- `match_observed_state` para comparar observaciones con sucesores legales.
- un nuevo comando de entrada, por ejemplo `python -m onitama.cli.vision_play`.

Ese MVP ya permitiria validar la arquitectura completa sin meter todavia toda la complejidad del modo en vivo definitivo.

## Criterio de diseño importante

La integracion debe tocar lo minimo posible el motor y la IA existentes.

La razon es simple:

- el motor ya encapsula correctamente las reglas,
- la IA ya opera sobre `GameState`,
- y la vision ya produce una representacion compatible.

Por tanto, la nueva logica debe vivir sobre todo en una capa de coordinacion, no repartida de forma difusa por todo el proyecto.

En el codigo final, esa coordinacion queda repartida de forma intencionada:

- `session/` contiene la coordinacion logica, testeable sin camara.
- `runtime/` contiene la coordinacion operativa del bucle en vivo.

La frontera entre ambas evita mezclar validacion de turnos con detalles de
camara, lectura de frames o estado de interfaz.

## Resumen ejecutivo

La fase final del proyecto consiste en transformar una pipeline de vision que ya produce estados utiles en un sistema jugable de extremo a extremo.

La pieza principal no es la deteccion visual en si, sino una capa de control de partida que:

- estabilice observaciones,
- valide legalidad con el motor,
- sincronice turnos reales,
- y conecte esas transiciones con la toma de decisiones de la IA.

En la estructura final, esa capa de control se divide en logica de sesion
(`session/`) y ejecucion en vivo (`runtime/`). Con esa separacion, el
proyecto pasa de tener piezas tecnicas solidas por separado a tener un sistema
completo y demostrable.
