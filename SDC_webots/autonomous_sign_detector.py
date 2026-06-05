# =============================================================================
# Controlador Manual con Detección de Señales (Actividad 4.1)
#
# Basado en simple_controller_act_2_1.py: conserva el control manual completo
# (UP/DOWN velocidad, LEFT/RIGHT giro, 'A' captura de pantalla). Sobre esa
# base añade el pipeline de percepción de la Actividad 4.1: por cada frame
# se construye una máscara HSV de los tres colores dominantes de las señales
# de tránsito (rojo, azul y amarillo), se filtran los contornos por tamaño y
# aspecto, y cada ROI candidata se clasifica con una CNN (32x32 RGB) entrenada
# en Colab. Las detecciones con confianza >= 0.7 se anotan sobre el display
# embebido del BMW X5 y se acumula el conjunto de etiquetas únicas vistas
# durante la sesión para verificar la cobertura del video de demostración.
# =============================================================================

import os
import json
import time
from datetime import datetime

import numpy as np
import cv2

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver

# =============================================================================
# Constantes de control (heredadas de Actividad 2.1)
# =============================================================================
DEBOUNCE_TIME = 0.1   # 100 ms para evitar rebote del teclado
MAX_ANGLE = 0.5
MAX_SPEED = 250
SPEED_INCR = 5
ANGLE_INCR = 0.05

# =============================================================================
# Constantes de detección de señales (Actividad 4.1)
# =============================================================================
MIN_CONTOUR_SIDE = 8           # 15 -> 8: la máscara HSV produce ~6-10% de píxeles pero
                               # en blobs pequeños y dispersos; con 15 px se descartaban
                               # incluso señales reales lejanas. 8 px captura ROIs distantes.
ASPECT_MIN = 0.7               # aspect ratio mínimo (las señales son cuadradas)
ASPECT_MAX = 1.4               # aspect ratio máximo

CNN_INPUT_SIZE = 32            # 32x32 RGB - arquitectura fijada en entrenamiento
CNN_NORMALIZATION = 255.0      # /255 para mapear [0,255] -> [0,1] - igual que el training
CONFIDENCE_THRESHOLD = 0.65    # 0.85 -> 0.65: con la cámara de 128x64 las ROIs son
                               # tan pequeñas que la CNN (entrenada en 32x32 GTSRB de
                               # fotos reales) produce confianzas moderadas para señales
                               # renderizadas en Webots. Las máscaras de cofre y cielo
                               # ahora bloquean los falsos positivos dominantes, así que
                               # podemos bajar el umbral sin volver a aceptar fondo.
CNN_RUN_EVERY_N = 3            # throttle: corremos la inferencia 1 de cada N frames

# Rangos HSV para los tres colores dominantes de las señales de tránsito.
# El rojo envuelve el círculo HSV, por lo que se modela con dos rangos
# (uno cerca de 0° y otro cerca de 180°) y se unen con un OR lógico.
HSV_RANGES = {
    "rojo_bajo": ((0,   100, 80), (10,  255, 255)),
    "rojo_alto": ((160, 100, 80), (180, 255, 255)),
    "azul":      ((100, 100, 80), (130, 255, 255)),
    "amarillo":  ((20,  100, 80), (35,  255, 255)),
}

# Estilos para dibujar las anotaciones sobre el display del cockpit.
BBOX_COLOR_BGR = (0, 255, 0)       # verde brillante para el rectángulo
TEXT_COLOR_BGR = (0, 255, 255)     # amarillo intenso para el texto
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICK = 1

LOG_EVERY_N = 30                   # frecuencia de impresión del resumen acumulado


# =============================================================================
# FUNCIONES DE CAPTURA Y DISPLAY (heredadas de Actividad 2.1, adaptadas a color)
# =============================================================================
def get_image(camera):
    """Lee un frame BGRA desde la cámara onboard (igual que en 2.1)."""
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image


def display_color_image(display, bgra):
    """Envía el frame BGRA al widget Display convirtiéndolo a RGB."""
    # El widget acepta RGB plano; descartamos el canal alfa al convertir.
    rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    ref = display.imageNew(
        rgb.tobytes(),
        Display.RGB,
        width=rgb.shape[1],
        height=rgb.shape[0],
    )
    display.imagePaste(ref, 0, 0, False)
    # Liberar la imagen para no fugar handles en cada paso de simulación.
    display.imageDelete(ref)


# =============================================================================
# CARGA DEL MODELO CNN (Stage 4.1)
# Resolución de rutas: directorio donde vive este .py, no el cwd.
# =============================================================================
def load_sign_model(controller_dir):
    """Carga el CNN y el mapa de etiquetas una sola vez al arranque."""
    # Importación diferida: TensorFlow es pesado y solo se requiere al inicio.
    from tensorflow.keras.models import load_model

    model_path = os.path.join(controller_dir, "traffic_sign_cnn.h5")
    labels_path = os.path.join(controller_dir, "label_map.json")

    # compile=False evita reconstruir el optimizador (no entrenamos aquí).
    model = load_model(model_path, compile=False)
    with open(labels_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    print(
        f"[INIT] CNN cargada desde {controller_dir} "
        f"(clases={len(label_map)}, input={CNN_INPUT_SIZE}x{CNN_INPUT_SIZE} RGB, "
        f"umbral={CONFIDENCE_THRESHOLD})"
    )
    return model, label_map


# =============================================================================
# PIPELINE DE PERCEPCIÓN: HSV -> CONTORNOS -> CNN
# =============================================================================
def build_color_mask(bgr):
    """Construye una máscara HSV combinada para rojo, azul y amarillo (señales de tránsito)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Acumulamos las cuatro máscaras de color en una sola con OR lógico.
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in HSV_RANGES.values():
        m = cv2.inRange(
            hsv,
            np.array(low, dtype=np.uint8),
            np.array(high, dtype=np.uint8),
        )
        combined = cv2.bitwise_or(combined, m)

    # Morfología para limpiar ruido (open) y cerrar huecos pequeños (close).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    # Anular el 25% inferior: el cofre rojo del BMW cae en esa zona y dispararía
    # la máscara de rojo generando candidatos falsos cada frame.
    h, w = combined.shape
    combined[int(h * 0.75):, :] = 0
    # Anular el 40% superior: la cámara es de 128x64 (muy baja resolución),
    # el horizonte cae cerca de y=30 y el cielo azul saturado sigue dominando
    # los candidatos. 15% no era suficiente; 40% deja la franja central
    # (rows ~26-48, donde están las señales a la altura de los ojos).
    combined[:int(h * 0.40), :] = 0
    return combined


def propose_candidates(mask):
    """Extrae bboxes candidatas (x, y, w, h) a partir de los contornos externos."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filtro de tamaño: descartar blobs muy pequeños (ruido o señales muy lejanas).
        if w < MIN_CONTOUR_SIDE or h < MIN_CONTOUR_SIDE:
            continue
        # Filtro de aspecto: las señales son aproximadamente cuadradas (0.7-1.4).
        ratio = w / float(h)
        if ratio < ASPECT_MIN or ratio > ASPECT_MAX:
            continue
        candidates.append((x, y, w, h))
    return candidates


def classify_roi(model, label_map, roi_bgr):
    """Clasifica un ROI con la CNN. Devuelve (etiqueta_en_espanol, confianza)."""
    # Redimensionar a la entrada nativa de la red (32x32 px).
    roi_resized = cv2.resize(
        roi_bgr, (CNN_INPUT_SIZE, CNN_INPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    # OpenCV trabaja en BGR; el modelo se entrenó con RGB de PIL -> convertir.
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    # Normalizar a [0, 1] igual que en el pipeline de entrenamiento.
    x = roi_rgb.astype(np.float32) / CNN_NORMALIZATION
    # Inferencia silenciosa (verbose=0 evita spam por frame).
    probs = model.predict(x[None, ...], verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    # Las claves del label_map.json son strings (json.dump las castea).
    label = label_map[str(idx)]
    return label, conf


def draw_detection(bgra, x, y, w, h, label, conf):
    """Dibuja el bbox y la etiqueta de la detección sobre el frame BGRA."""
    cv2.rectangle(bgra, (x, y), (x + w, y + h), BBOX_COLOR_BGR, 2)
    text = f"{label} ({conf:.2f})"
    # Ubicamos el texto justo arriba del bbox; si no cabe, lo movemos abajo.
    text_y = y - 5 if y - 5 > 10 else y + h + 15
    cv2.putText(
        bgra, text, (x, text_y), FONT, FONT_SCALE,
        TEXT_COLOR_BGR, FONT_THICK, cv2.LINE_AA,
    )


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Resolución de rutas: directorio donde vive este .py, no el cwd.
    controller_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Carga del modelo CNN (Stage 4.1) ---
    model, label_map = load_sign_model(controller_dir)

    # --- Estado de control manual (heredado de 2.1) ---
    speed = 10
    angle = 0.0
    last_press = {}

    # --- Estado de detección de señales (Actividad 4.1) ---
    unique_signs_seen = set()   # etiquetas distintas vistas durante la sesión
    frame_count = 0
    # Detecciones aceptadas en la última corrida de la CNN. Se re-dibujan en los
    # frames intermedios (cuando saltamos la inferencia por throttle).
    last_detections = []
    # Bandera para el volcado de diagnóstico (se dispara una sola vez).
    _snapshot_done = False

    # Instancia del vehículo y del driver (igual que 2.1).
    robot = Car()
    driver = Driver()

    # Paso de simulación del mundo cargado.
    timestep = int(robot.getBasicTimeStep())

    # Cámara onboard montada al frente del BMW X5.
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Widget de display embebido en el cockpit (sensorsSlotTop del .wbt).
    display_img = Display("display_image")

    # Teclado para el control manual.
    keyboard = Keyboard()
    keyboard.enable(timestep)

    while robot.step() != -1:
        frame_count += 1

        # --- Captura de imagen ---
        # .copy(): get_image() devuelve una vista del buffer interno de Webots,
        # que es de solo lectura; cv2.rectangle/putText necesitan un arreglo escribible.
        bgra = get_image(camera).copy()
        # Trabajamos en BGR para HSV y para los dibujos de OpenCV.
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

        # --- Pipeline de percepción (throttled cada CNN_RUN_EVERY_N frames) ---
        # En los frames intermedios reutilizamos las últimas detecciones para
        # que el display no parpadee y para no saturar la inferencia de la CNN.
        if frame_count % CNN_RUN_EVERY_N == 0:
            mask = build_color_mask(bgr)
            candidates = propose_candidates(mask)
            refreshed = []
            # DBG: acumulamos (label, conf) de cada candidata para diagnóstico,
            # incluso si no superan el umbral, así sabemos qué está viendo la CNN.
            dbg_top = []
            for (x, y, w, h) in candidates:
                roi = bgr[y:y + h, x:x + w]
                if roi.size == 0:
                    continue
                label, conf = classify_roi(model, label_map, roi)
                dbg_top.append((label, conf))
                if conf > CONFIDENCE_THRESHOLD:
                    refreshed.append((x, y, w, h, label, conf))
                    unique_signs_seen.add(label)
            last_detections = refreshed
            # Instrumentación de diagnóstico (cada 30 frames de simulación):
            # % de píxeles encendidos en la máscara HSV, nº de candidatas
            # tras filtrar contornos, nº aceptadas y top-5 (clase, confianza).
            if frame_count % 30 == 0:
                mask_pct = float((mask > 0).sum()) / mask.size * 100.0
                top_str = ", ".join(f"{lbl}@{c:.2f}" for lbl, c in dbg_top[:5])
                print(
                    f"[DBG] frame={frame_count} mask%={mask_pct:.1f} "
                    f"cands={len(candidates)} acc={len(refreshed)} "
                    f"top=[{top_str}]"
                )
            # Volcado one-shot: en el primer frame de inferencia con candidatas
            # guardamos vista completa, máscara HSV y cada ROI a disco para
            # poder inspeccionar visualmente qué está clasificando la CNN.
            if not _snapshot_done and len(candidates) > 0:
                snap_dir = os.path.join(controller_dir, "debug_snapshot")
                os.makedirs(snap_dir, exist_ok=True)
                cv2.imwrite(os.path.join(snap_dir, "frame_full.png"), bgr)
                cv2.imwrite(os.path.join(snap_dir, "mask.png"), mask)
                for i, (cx, cy, cw, ch) in enumerate(candidates):
                    roi_dump = bgr[cy:cy + ch, cx:cx + cw]
                    if roi_dump.size == 0:
                        continue
                    lbl_i, conf_i = dbg_top[i] if i < len(dbg_top) else ("?", 0.0)
                    # Nombre del archivo embebe pos+tamaño+etiqueta para correlacionar.
                    safe = lbl_i.replace(" ", "_").replace("/", "_")
                    fname = f"roi_{i:02d}_{cx}-{cy}_{cw}x{ch}_{safe}_{conf_i:.2f}.png"
                    cv2.imwrite(os.path.join(snap_dir, fname), roi_dump)
                print(f"[DBG] snapshot guardado en {snap_dir} "
                      f"({len(candidates)} ROIs)")
                _snapshot_done = True

        # Re-dibujar las últimas detecciones aceptadas sobre el frame actual
        # (se ejecuta cada frame, no solo cuando corre la CNN).
        for (x, y, w, h, label, conf) in last_detections:
            draw_detection(bgra, x, y, w, h, label, conf)

        # --- Mostrar el frame anotado en el display del cockpit ---
        display_color_image(display_img, bgra)

        # --- Resumen periódico para el video de demostración ---
        if frame_count % LOG_EVERY_N == 0:
            print(
                f"[4.1] Señales únicas detectadas: {len(unique_signs_seen)} "
                f"-> {sorted(unique_signs_seen)}"
            )

        # --- Bloque de teclado (idéntico a Actividad 2.1) ---
        current_time = time.time()
        key = keyboard.getKey()

        if key in last_press and (current_time - last_press[key] < DEBOUNCE_TIME):
            continue  # Ignore rebound

        # pressed key accepted, update
        last_press[key] = current_time

        if key == keyboard.UP:  # up
            if speed < MAX_SPEED:
                speed += SPEED_INCR
                print("up")
        elif key == keyboard.DOWN:  # down
            if speed >= SPEED_INCR:
                speed -= SPEED_INCR
                print("down")
        elif key == keyboard.RIGHT:  # right
            angle += ANGLE_INCR
            if angle > MAX_ANGLE:
                angle = MAX_ANGLE
            print("right")
        elif key == keyboard.LEFT:  # left
            angle -= ANGLE_INCR
            if angle < -MAX_ANGLE:
                angle = -MAX_ANGLE
            print("left")
        elif key == ord('A'):
            # filename with timestamp and saved in current directory
            current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            file_name = current_datetime + ".png"
            print("Image taken")
            camera.saveImage(os.getcwd() + "/" + file_name, 1)

        # update angle and speed
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()
