# Resumen del proceso de entrenamiento del clasificador de cartas de Onitama

## 1. Objetivo

El objetivo de esta fase del proyecto fue entrenar un modelo de **clasificación de imágenes** capaz de reconocer automáticamente cuál de las **16 cartas de movimiento de Onitama** aparece en cada recorte de carta.

Este clasificador forma parte del sistema de percepción visual del proyecto. La idea general es:

- detectar o definir la zona donde se encuentra cada carta en la imagen del tablero,
- recortar esa región,
- y usar un clasificador para identificar la carta concreta.

Por tanto, en esta fase no se entrenó un detector de cartas sobre la imagen completa, sino un **clasificador sobre recortes individuales** de cartas.

---

## 2. Construcción del dataset

### 2.1. Adquisición de imágenes

Se partió de fotografías del tablero real de Onitama. Cada fotografía contenía las cinco cartas visibles en la partida:

- dos cartas de un jugador,
- dos cartas del otro jugador,
- y una carta central.

A partir de cada fotografía del tablero se obtuvieron **cinco recortes**, uno por cada carta. De esta forma, una sola imagen del tablero generaba cinco muestras para el clasificador.

### 2.2. Número total de imágenes

Tras el proceso de captura y recorte se construyó un dataset final con **2000 imágenes de cartas**, distribuidas entre las **16 clases** del juego.

El reparto por clases quedó razonablemente equilibrado, con una distribución aproximada alrededor de 125 imágenes por clase, sin desequilibrios graves.

### 2.3. División en subconjuntos

La versión final del dataset utilizada en el experimento principal quedó dividida en tres subconjuntos:

- **train**: 1400 imágenes
- **validation**: 400 imágenes
- **test**: 200 imágenes

La división se realizó **por grupos**, no de forma completamente aleatoria por imagen individual, para evitar fugas de información entre entrenamiento y validación/test. Esto era importante porque varias imágenes procedían de disposiciones muy parecidas del tablero.

Es decir la técnica a seguir fue:
1. Colocar 5 cartas
2. Tomar 3 fotos modificando ángulo, orientacion side card.
3. Rotar las 5 cartas, sacar la última y meter otra nueva en la primera posición.

---

## 3. Preprocesado de las imágenes

### 3.1. Problema inicial

Los recortes de cartas no eran cuadrados y no todos tenían el mismo tamaño. Además, la carta no ocupaba siempre exactamente la misma posición dentro del ROI, ya que podía desplazarse a lo largo del ancho del tablero.

Esto impedía usar recortes muy ajustados y obligaba a mantener cierto margen alrededor de la carta.

### 3.2. Estrategia de resize

Se descartó deformar las imágenes mediante un estirado directo (`stretch`), porque eso alteraría las proporciones reales de la carta y podría perjudicar el aprendizaje.

La estrategia elegida fue:

- fijar un tamaño de entrada común,
- **mantener la relación de aspecto**,
- y completar el resto con **padding negro**.

Finalmente, las imágenes se prepararon a **320 × 320 píxeles**.

### 3.3. Justificación del tamaño 320×320

Inicialmente se estudió usar 224×224, por ser un tamaño típico en clasificación, pero al inspeccionar los recortes se observó que:

- las cartas perdían algo de detalle,
- y el padding negro ocupaba una fracción importante de la imagen.

Teniendo en cuenta que el tamaño mediano de las imágenes del dataset era aproximadamente:

- **377 px de ancho**
- **251 px de alto**

se decidió usar **320×320**, que ofrecía un mejor compromiso entre:

- conservación de detalle,
- coste computacional,
- y tamaño final uniforme para el modelo.

---

## 4. Estrategia de augmentación

### 4.1. Augmentación previa en Roboflow

Se estudió la posibilidad de aplicar augmentations offline en Roboflow, valorando opciones como:

- brightness
- exposure
- rotation
- flips
- crop
- blur
- saturation

Sin embargo, tras varias pruebas se observó que no era necesario aumentar artificialmente el dataset antes del entrenamiento. Además, algunas transformaciones podían generar ejemplos poco realistas para este problema.

Por ello, en la configuración final se optó por **no aplicar augmentación previa en Roboflow** y usar únicamente:

- el preprocesado estructural,
- el resize a 320×320,
- el mantenimiento de proporción,
- y el padding negro.

### 4.2. Augmentaciones internas de YOLO

Durante el entrenamiento se observó que Ultralytics aplicaba además sus propias augmentations internas por defecto, entre ellas:

- `fliplr=0.5`
- `translate=0.1`
- `hsv_h=0.015`
- `hsv_s=0.7`
- `hsv_v=0.4`
- `erasing=0.4`

Estas transformaciones se aplican **online**, es decir, durante el entrenamiento, sin generar nuevas imágenes físicas en el dataset.

A la vista de los resultados, estas augmentations internas resultaron suficientes, por lo que no fue necesario añadir una capa adicional de augmentación offline en Roboflow.

---

## 5. Modelo utilizado

Se comparó la posibilidad de usar YOLOv8 y YOLO11 para clasificación.

Finalmente se eligió:

**YOLO11n-cls**

Las razones principales fueron:

- pertenece a la familia más reciente de Ultralytics,
- mantiene el mismo ecosistema y flujo de trabajo ya utilizado en detección,
- es un modelo ligero,
- y resultó suficiente para un problema de clasificación con solo 16 clases y un entorno visual relativamente controlado.

El modelo cargado fue:

```python
model = YOLO("yolo11n-cls.pt")
```

Esto implica que se partió de pesos preentrenados, lo cual mejora la convergencia y aprovecha conocimiento visual previo aprendido sobre datasets grandes.

## 6. Entorno de entrenamiento

El entrenamiento se realizó en Google Colab con GPU. En concreto, se utilizó una Tesla T4.

La instalación se hizo con el paquete ultralytics, sin necesidad de cambiar de framework respecto a trabajos anteriores de detección.

## 7. Configuración de entrenamiento

La configuración principal fue la siguiente:

```python
from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")

results = model.train(
    data=dataset.location,
    epochs=30,
    imgsz=320,
    batch=32,
    device=0,
    project="/content/runs_classify",
    name="onitama_yolo11n_cls_320"
)
```

### Significado de los parámetros principales

- `data`: ruta al dataset de clasificación exportado desde Roboflow.
- `epochs`: número de vueltas completas sobre el dataset.
- `imgsz=320`: tamaño de entrada del modelo.
- `batch=32`: número de imágenes procesadas simultáneamente antes de actualizar los pesos.
- `device=0`: uso de la GPU principal.
- `project` y `name`: ruta y nombre del experimento.

## 8. Elección del número de épocas

Inicialmente se entrenó el modelo durante 50 épocas. Sin embargo, al analizar las métricas se observó que:

- la precisión subía muy rápido,
- y el modelo alcanzaba valores prácticamente máximos mucho antes del final.

Por ello se repitió el entrenamiento con 30 épocas, obteniendo resultados equivalentes.

Esto permitió concluir que 30 épocas eran suficientes y que aumentar el número de epochs no aportaba una mejora apreciable para este problema concreto.

## 9. Evolución del entrenamiento

Durante el entrenamiento se monitorizaron principalmente:

- `loss`
- `top1_acc`
- `top5_acc`

### 9.1. Loss

La loss representa el error interno que el modelo intenta minimizar. En este caso descendió rápidamente desde valores altos al inicio hasta valores bajos al final, lo que indica una convergencia correcta.

### 9.2. Top-1 accuracy

Es la métrica más importante en este problema. Representa el porcentaje de imágenes en las que la clase predicha con mayor probabilidad coincide exactamente con la clase real.

En el experimento final con el dataset original, la evolución fue muy rápida:

- Epoch 1: top1 ≈ 0.11
- Epoch 2: top1 ≈ 0.837
- Epoch 3: top1 ≈ 0.940
- Epoch 4: top1 ≈ 0.960
- Epoch 7: top1 ≈ 0.988
- Epoch 15: top1 ≈ 0.998
- Epoch 17: top1 = 1.000

Esto muestra que, aunque el arranque es más modesto que en un entrenamiento con más imágenes, el modelo converge igualmente a un rendimiento excelente.

### 9.3. Top-5 accuracy

Representa el porcentaje de imágenes en las que la clase correcta aparece entre las cinco clases más probables.

En este problema alcanzó 1.000 muy pronto, lo cual era esperable dado que:

- solo había 16 clases,
- y el problema visual resultó bastante separable para el modelo.

## 10. Resultados finales

Tras 30 épocas, el modelo obtuvo:

```text
top1_acc = 1.000
top5_acc = 1.000
```

tanto en validación final como en la evaluación posterior del mejor modelo guardado (best.pt).

Además, la inferencia fue muy rápida:

- aproximadamente 1.1 ms por imagen

Esto confirma que el modelo no solo clasifica muy bien dentro del split utilizado, sino que además es suficientemente ligero para integrarse en un pipeline práctico de percepción.

## 11. Comparación entre entrenar con y sin augmentación previa en Roboflow

Se realizaron dos tipos de experimentos:

### Experimento A: dataset ampliado con augmentación previa en Roboflow

En este caso se entrenó con una versión del dataset aumentada offline antes del entrenamiento.

El modelo obtuvo un rendimiento excelente, alcanzando también:

```text
top1_acc = 1.000
top5_acc = 1.000
```

### Experimento B: dataset original sin augmentación previa en Roboflow

En este caso se utilizó únicamente el dataset original, sin generar imágenes adicionales antes del entrenamiento. Se mantuvo el mismo preprocesado y se dejaron activas las augmentations online por defecto de Ultralytics.

El modelo volvió a obtener:

```text
top1_acc = 1.000
top5_acc = 1.000
```

### Conclusión de la comparación

La comparación entre ambos experimentos permite extraer una conclusión importante:

- la augmentación previa en Roboflow no aportó una mejora observable en el rendimiento final del clasificador.

Por tanto, el dataset original ya era suficientemente bueno para resolver la tarea, siempre que se mantuvieran:

- el preprocesado adecuado,
- el resize a 320×320 con padding negro,
- y las augmentations internas de YOLO durante el entrenamiento.

## 12. Interpretación de los resultados

Los resultados indican que:

- el problema de clasificación de cartas está bien planteado,
- el dataset construido es suficientemente representativo,
- el preprocesado elegido (320×320 con padding negro y sin deformación) es válido,
- el dataset original ya contiene suficiente información para entrenar con éxito,
- y el modelo YOLO11n-cls tiene capacidad suficiente para resolver la tarea con alto rendimiento.

Además, los experimentos realizados sugieren que:

- aumentar artificialmente el dataset en Roboflow no mejora el resultado final en este caso,
- mientras que sí incrementa el tamaño del dataset y el tiempo de entrenamiento.

No obstante, desde el punto de vista metodológico, es importante señalar que un resultado perfecto en validación y test debe complementarse con pruebas sobre imágenes nuevas no pertenecientes al dataset, para verificar la capacidad de generalización en condiciones reales de uso.

## 13. Decisión metodológica final

A partir de los experimentos realizados, la configuración final seleccionada para el proyecto fue:

- dataset original
- sin augmentación extra en Roboflow
- resize a 320×320
- mantenimiento de proporción
- padding negro
- YOLO11n-cls
- 30 epochs
- batch = 32
- augmentations por defecto de Ultralytics
- entorno de entrenamiento: Google Colab con Tesla T4

Esta configuración ofrece:

- rendimiento excelente,
- entrenamiento relativamente corto,
- pipeline más simple,
- y menor complejidad metodológica.

## 14. Conclusión técnica

La fase de entrenamiento del clasificador permitió validar con éxito el enfoque de:

**recorte de carta + clasificación supervisada**

frente a enfoques más complejos de detección directa de cartas sobre la imagen completa.

La solución final adoptada fue:

- modelo: YOLO11n-cls
- entrada: 320×320
- preprocesado: conservación de proporción + padding negro
- epochs: 30
- batch: 32
- dataset: versión original sin augmentación previa en Roboflow
- augmentación durante entrenamiento: augmentations online por defecto de Ultralytics
- entorno: Google Colab con GPU Tesla T4

Los resultados obtenidos fueron excelentes y muestran que el clasificador es una base sólida para integrarlo posteriormente en el sistema completo de percepción del tablero.
