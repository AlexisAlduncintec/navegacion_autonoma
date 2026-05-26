"""
================================================================================
Actividad 3.1 - Detección de peatones y barriles con frenado de emergencia
MR4010 - Navegación Autónoma | Tecnológico de Monterrey | MNA
Equipo 18:
  - Alexis Alduncin Barragán          - A01017478
  - David Rodrigo Alvarado Domínguez  - A01797606
  - Abraham Avila Garcia              - A01795305
  - Jorge Luis Ancheyta Segovia       - A01796354
Profesor: Dr. David Antonio Torres
================================================================================

Este controlador externo integra cuatro componentes sobre el BMW X5 del mundo
city_2025a_activity_3_1.wbt:

    1. Seguimiento de carril (HEREDADO de Actividad 2.1): HSV-amarillo -> Canny
       -> ROI -> HoughLinesP -> PID -> Driver.setSteeringAngle()
    2. LiDAR Sick LMS 291 frontal: lectura del sector central (~25°) limitado
       a 20 m para detectar obstáculos por delante del vehículo.
    3. Clasificador HOG + SVM (reentrenado con 600 frames de la cámara del
       BMW hand-etiquetados + hard-negative mining estilo Dalal-Triggs; ver
       retrain_svm_hardneg.py en este directorio. El modelo inicial entrenado
       con Penn-Fudan fue descartado por brecha de dominio entre fotos reales
       y mannequins renderizados de Webots): se ejecuta SOLO cuando el LiDAR
       confirma un obstáculo cercano. Sliding-window horizontal sobre la
       cámara 256x128 con ventana 64x128 tipo Dalal-Triggs.
    4. Máquina de estados de frenado: DRIVING / BRAKE_PEDESTRIAN / BRAKE_BARREL.
       Los barriles activan intermitentes; los peatones NO (se reanuda crucero
       cuando se quitan del frente).

Cambios respecto a Actividad 2.1 (cada cambio justificado en comentarios inline):
    - Ganancias del PID: Kp 0.008 -> 0.010, Kd 0.004 -> 0.002 (feedback del TA:
      el coche anticipaba curvas; bajamos Kd y subimos Kp ligeramente).
    - main._prev_angle (atributo de función) -> variable local prev_angle
      (más limpio, el profesor penaliza código innecesariamente complejo).
    - Se añaden: Sick LMS 291 LiDAR, carga de SVM entrenado, clasificación
      por sliding window, y la máquina de estados de frenado con intermitentes.

Mundo: SDC_webots/worlds/city_2025a_activity_3_1.wbt (controller "<extern>")
Cámara del vehículo: 256x128 (vs 128x64 de la Actividad 2.1)
Validado a velocidad mínima de 50 km/h.

Uso (Mac / VS Code terminal):
    export WEBOTS_HOME=/Applications/Webots.app
    export PYTHONPATH=/Applications/Webots.app/Contents/lib/controller/python
    export WEBOTS_PYTHON_EXECUTABLE=$(which python)
    conda activate webots
    cd /Users/alexisalduncin/ITESM/navegacion_autonoma/SDC_webots
    python autonomous_obstacle_controller.py
"""

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import math
import os
import sys

import cv2
import joblib
import numpy as np
from skimage.feature import hog

# =============================================================================
# CONSTANTES DE CONFIGURACIÓN
# =============================================================================

# --- Velocidad ---
# La actividad exige una velocidad mínima de 50 km/h. Mantenemos constante
# para que el comportamiento del PID sea independiente de la velocidad.
TARGET_SPEED = 50.0          # km/h - cumple el mínimo de la actividad
MAX_STEERING_ANGLE = 0.5     # radianes - límite mecánico del BMW X5

# --- Velocidad de frenado ---
# Se usa cuando la máquina de estados decide detener el vehículo (peatón o barril).
# Driver.setCruisingSpeed(0.0) corta tracción; las ruedas se frenan por inercia
# y la fricción del modelo del BMW X5 (no hace falta freno explícito adicional).
BRAKE_SPEED_KMH = 0.0

# --- Parámetros de Canny --- (sin cambios respecto a Actividad 2.1)
CANNY_LOW = 50
CANNY_HIGH = 150

# --- Parámetros de HoughLinesP --- (sin cambios respecto a Actividad 2.1)
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 20
HOUGH_MIN_LINE_LEN = 15
HOUGH_MAX_LINE_GAP = 20

# --- Filtro de líneas horizontales --- (sin cambios)
HORIZONTAL_SLOPE_THRESHOLD = 0.05

# --- Ganancias del controlador PID (RE-TUNING DE ACTIVIDAD 3.1) ---
# Feedback del TA en Actividad 2.1: el auto anticipaba las curvas demasiado
# pronto. La componente derivativa "predice" el error futuro a partir de la
# tasa de cambio del error; si es grande, el coche reacciona antes de tiempo.
# Solución acordada en clase:
#   - bajar Kd a la mitad para reducir esa anticipación.
#   - subir Kp ~25% para que la corrección proporcional compense la pérdida
#     de amortiguamiento sin volver oscilante el control.
PID_KP = 0.010                # antes 0.008  -> ↑ corrección proporcional ligeramente mayor
PID_KI = 0.0                  # sin cambio   -> lane keeping no se beneficia de integrador
PID_KD = 0.002                # antes 0.004  -> ↓ derivativo a la mitad para no anticipar curvas

# --- Slew rate limit --- (sin cambios)
MAX_ANGLE_CHANGE_PER_FRAME = 0.05

# --- Comportamiento por defecto cuando no hay líneas detectadas ---
DEFAULT_STEERING_ANGLE = 0.0
NO_DETECTION_GRACE_FRAMES = 20

# --- Setpoint del PID en el eje X de la imagen (FIX setpoint - Actividad 3.1) ---
# La cámara es 256 px de ancho; el controlador heredado de Actividad 2.1 usaba
# image_width / 2 = 128 como setpoint, asumiendo que la línea amarilla debía
# caer en el centro del frame. Eso equivale a "el coche se monta sobre la
# línea central de la carretera", no a "el coche va en su carril derecho".
#
# En este mundo (city_2025a_activity_3_1.wbt) la línea central amarilla
# separa los dos sentidos de circulación. Cuando el BMW va correctamente en
# su carril derecho, la línea se ve a la IZQUIERDA del centro de cámara
# (porque la cámara está montada sobre el centro del cofre del coche, y el
# coche está a la derecha del centro de la calle).
#
# Diagnóstico del bug: con setpoint=128 el PID arrastraba al coche hacia
# la izquierda hasta que la línea quedaba bajo el centro de la imagen,
# lo cual significa que el coche estaba pisando la línea central /
# invadiendo el carril contrario / saliéndose a la banqueta izquierda.
#
# Setpoint corregido: estimado a partir de la geometría del carril y la FOV
# de la cámara (FOV ≈ 1 rad, ancho carril ≈ 3.5 m, mitad = 1.75 m a la izq
# de la cámara; a ~10 m de distancia eso corresponde a aprox 87-110 px de
# desplazamiento desde el centro). Punto de partida: 110. Si el coche sigue
# derivando a la izquierda hay que bajar más (probar 100, 95...); si ahora
# deriva a la derecha hay que subir (probar 115, 120...). Iterar con ojos
# en el viewport, NO ajustar Kp/Kd/SVM/LiDAR.
PID_SETPOINT_PX = 110

# --- Robustez en intersecciones (FIX 1 - Actividad 3.1) ---
# El controlador heredado de Actividad 2.1 al perder la línea más de
# NO_DETECTION_GRACE_FRAMES poníaa el ángulo a 0 y reseteaba el PID. En
# intersecciones (donde no hay línea amarilla) eso causaba que el coche
# atravesara recto, se saliera del trazado, y terminara en la calle
# perpendicular sin posibilidad de recuperar la línea.
#
# Fix mínimo: durante la pérdida prolongada se mantiene el ÚLTIMO ÁNGULO
# VÁLIDO (no se resetea a 0) y se baja la velocidad de crucero. Velocidad
# más baja => menos distancia recorrida ciegamente => más probabilidad de
# re-encontrar la línea amarilla antes de salir del camino pintado.
LANE_LOST_SPEED_KMH = 20.0   # velocidad reducida durante intersecciones (vs 50 normal)


# =============================================================================
# CONSTANTES LiDAR (NUEVO - Stage 2)
# =============================================================================
# El BMW X5 monta un Sick LMS 291 en sensorsSlotFront (ver city_2025a_activity_3_1.wbt).
# El nombre del dispositivo es el default del PROTO ("Sick LMS 291"), confirmado
# leyendo simple_controller_lidar.py del ZIP de ejemplos del profesor.
#
# El LMS 291 barre 180° en horizontal con 180 puntos (1° por punto). Para esta
# actividad solo nos importa el sector frontal: aprox. 25° centrados al frente
# del coche. Filtramos en software (sin tocar el .wbt) para no modificar el
# mundo del profesor — esto es lo que sugiere la actividad y mantiene el código
# mínimo.
LIDAR_DEVICE_NAME      = "Sick LMS 291"   # nombre por defecto del PROTO
LIDAR_FOV_CENTRAL_DEG  = 25.0             # sector útil = ~25° al frente
LIDAR_FOV_TOTAL_DEG    = 180.0            # FOV por defecto del SickLms291
LIDAR_MAX_DETECT_M     = 20.0             # umbral de obstáculo (actividad: 20 m)


# =============================================================================
# CONSTANTES DEL CLASIFICADOR HOG + SVM (NUEVO - Stage 3)
# =============================================================================
# El modelo fue entrenado en pedestrian_detection (1).ipynb con cv2.resize a
# (64, 128) (convención OpenCV: (width, height)). En NumPy/skimage el patch
# resultante tiene forma (height, width) = (128, 64). La cámara del BMW X5
# es 256x128, así que basta con barrer horizontalmente: la altura de la
# ventana coincide exactamente con la altura de la imagen.
#
# Las rutas se resuelven con __file__ (no relativas al cwd) porque Webots
# ejecuta el controlador externo desde un cwd distinto al del .py y las
# rutas relativas fallan silenciosamente.
SVM_MODEL_PATH    = "svm_peatones.joblib"
SCALER_PATH       = "scaler_peatones.joblib"
HOG_PARAMS_PATH   = "hog_params.joblib"
SVM_WINDOW_W      = 64        # ancho de la ventana de entrenamiento (px)
SVM_WINDOW_H      = 128       # alto de la ventana = alto de la imagen (px)
SVM_FEATURE_DIM   = 3780      # dim del vector HOG con orientations=9, ppc=(8,8), cpb=(2,2), 64x128
SVM_STRIDE_X      = 16        # paso horizontal del sliding window
SVM_RUN_EVERY_N   = 2         # corre el clasificador cada N frames si hay obstáculo (ahorra CPU)

# --- Toggle de visualización (diagnóstico) ---
# DEBUG: durante el aislamiento del freeze a ~F351 se controla aquí si el
# widget Display de Webots se llena cada frame. Si ENABLE_DISPLAY=False el
# controlador omite las llamadas display.imageNew / imagePaste / imageDelete
# por completo. En la entrega final debe quedar en True.
ENABLE_DISPLAY = True


# =============================================================================
# CONSTANTES DE LA MÁQUINA DE ESTADOS DE FRENADO (NUEVO - Stage 4)
# =============================================================================
# DRIVING            => crucero normal a TARGET_SPEED, sin intermitentes
# BRAKE_PEDESTRIAN   => freno + intermitentes APAGADOS (reanuda cuando se aparta)
# BRAKE_BARREL       => freno + intermitentes ENCENDIDOS (supervisor despawnea
#                       el barril tras ~10 ticks => regresamos a DRIVING)
STATE_DRIVING          = "DRIVING"
STATE_BRAKE_PEDESTRIAN = "BRAKE_PEDESTRIAN"
STATE_BRAKE_BARREL     = "BRAKE_BARREL"


# =============================================================================
# CONTROLADOR PID (sin cambios respecto a Actividad 2.1)
# =============================================================================
class PIDController:
    """PID paralelo con anti-windup y salida acotada. Usa robot.getTime() como
    base de tiempo para que el dt sea correcto incluso en simulación fast."""

    def __init__(self, kp, ki, kd, output_limit=MAX_STEERING_ANGLE,
                 integral_limit=100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def reset(self):
        """Reinicia el estado del controlador. Útil al perder la línea."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def compute(self, error, current_time):
        if self.previous_time is None:
            self.previous_time = current_time
            self.previous_error = error
            return self.kp * error
        dt = current_time - self.previous_time
        if dt <= 0:
            return self.kp * error
        self.integral += error * dt
        self.integral = max(-self.integral_limit,
                            min(self.integral_limit, self.integral))
        derivative = (error - self.previous_error) / dt
        output = (self.kp * error
                  + self.ki * self.integral
                  + self.kd * derivative)
        output = max(-self.output_limit, min(self.output_limit, output))
        self.previous_error = error
        self.previous_time = current_time
        return output


# =============================================================================
# FUNCIONES DE PROCESAMIENTO DE IMAGEN (heredadas de Actividad 2.1)
# =============================================================================

def get_image(camera):
    """Cámara de Webots devuelve BGRA. Devolvemos BGR (sin alpha)."""
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image[:, :, :3]


def to_grayscale(image_bgr):
    """Máscara HSV amarilla + escala de grises (paso 3 de la actividad)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    yellow_lower = np.array([15, 80, 80])
    yellow_upper = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=yellow_mask)
    return cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)


def detect_edges(image_gray):
    """Blur Gaussiano + Canny (paso 4 de la actividad)."""
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    return cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)


def apply_roi(edges):
    """Trapecio cubriendo la mitad inferior de la imagen (paso 5)."""
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(width * 0.25), int(height * 0.50)),
        (int(width * 0.75), int(height * 0.50)),
        (width, height),
        (0, height),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


def detect_lines(edges_roi):
    """HoughLinesP probabilística (paso 6)."""
    return cv2.HoughLinesP(
        edges_roi,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )


def compute_error(lines, image_width):
    """Devuelve (error, num_valid_lines). Filtra líneas casi horizontales.

    El setpoint del PID NO es el centro de la imagen — es PID_SETPOINT_PX,
    una constante calibrada para que cuando el coche va correctamente en
    su carril derecho, la línea amarilla central de la calzada caiga en
    ese pixel (a la izquierda del centro). Ver comentario del constante.
    """
    if lines is None:
        return None, 0
    setpoint = PID_SETPOINT_PX
    smallest_error = None
    valid_lines = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        slope = float('inf') if dx == 0 else (y2 - y1) / dx
        if abs(slope) < HORIZONTAL_SLOPE_THRESHOLD:
            continue
        valid_lines += 1
        midpoint_x = (x1 + x2) / 2.0
        error = midpoint_x - setpoint
        if smallest_error is None or abs(error) < abs(smallest_error):
            smallest_error = error
    return smallest_error, valid_lines


# =============================================================================
# VISUALIZACIÓN (heredadas de Actividad 2.1)
# =============================================================================

def draw_lines_on_image(image_bgr, lines, color=(0, 255, 0), thickness=2):
    output = image_bgr.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), color, thickness)
    return output


def draw_setpoint_line(image_bgr, color=(0, 0, 255)):
    """Dibuja una línea vertical en PID_SETPOINT_PX (NO en el centro de la
    imagen). El operador puede ver dónde el PID está intentando colocar la
    línea amarilla del carril."""
    output = image_bgr.copy()
    x = int(PID_SETPOINT_PX)
    cv2.line(output, (x, 0), (x, output.shape[0]), color, 1)
    return output


def display_image_on_webots(display, image_bgr):
    """Webots Display espera RGB; convertimos.

    NOTA (fix de fuga de recursos): se DEBE llamar imageDelete después de
    imagePaste. Webots reserva un slot interno por cada imageNew(); si no
    se libera, después de ~350 iteraciones (timestep 32 ms = ~11 s) Webots
    se queda sin slots y wbu_driver_step() deadlockea silenciosamente.
    El controlador de Actividad 2.1 tiene la misma fuga; no se nota porque
    típicamente se prueba en sesiones cortas. Verificado con faulthandler.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)
    display.imageDelete(image_ref)


# =============================================================================
# LiDAR — NUEVO Stage 2
# =============================================================================

def init_lidar(robot, timestep):
    """Inicializa el Sick LMS 291 y calcula los índices del sector central.

    Patrón tomado del controlador de ejemplo del profesor
    (simple_controller_lidar.py del ZIP de la clase activa): getDevice +
    enable + enablePointCloud (este último habilita la nube de puntos en
    Webots para visualización aunque nosotros solo usemos getRangeImage()).

    Devuelve (lidar, slice_indices) donde slice_indices es un slice de
    Python que, aplicado a getRangeImage(), recorta el sector frontal útil
    (LIDAR_FOV_CENTRAL_DEG grados).

    Hacer el filtrado en software es preferible a tocar el .wbt: no
    requiere modificar el mundo del profesor y mantiene el controlador
    autocontenido.
    """
    lidar = robot.getDevice(LIDAR_DEVICE_NAME)
    lidar.enable(timestep)
    lidar.enablePointCloud()                         # mismo orden que el profesor
    horiz_res = lidar.getHorizontalResolution()      # típicamente 180
    central_n = max(1, int(horiz_res * LIDAR_FOV_CENTRAL_DEG / LIDAR_FOV_TOTAL_DEG))
    mid       = horiz_res // 2
    half      = central_n // 2
    return lidar, slice(mid - half, mid + half)


def read_forward_distance(lidar, central):
    """Distancia mínima en el sector frontal del LiDAR.

    getRangeImage() devuelve una lista de floats con horizontalResolution
    entradas; los rayos que no devuelven impacto aparecen como 'inf'.
    Filtramos los infinitos ANTES del min() para que un solo rayo perdido
    no enmascare un obstáculo real.
    """
    ranges = lidar.getRangeImage()[central]
    finite = [r for r in ranges if math.isfinite(r)]
    return min(finite) if finite else float("inf")


# =============================================================================
# SVM PEATÓN — NUEVO Stage 3
# =============================================================================

def load_pedestrian_svm(controller_dir):
    """Carga el LinearSVC entrenado + StandardScaler + parámetros HOG.

    Se cargan con ruta absoluta (basada en __file__) en vez de relativa
    al cwd porque Webots ejecuta el controlador externo desde un directorio
    distinto y las rutas relativas fallan en silencio.

    Verificación al cargar:
      - el LinearSVC debe haber sido entrenado con SVM_FEATURE_DIM features.
      - hog_params['feature_dim'] (escrito por el notebook) debe coincidir.
    """
    clf    = joblib.load(os.path.join(controller_dir, SVM_MODEL_PATH))
    scaler = joblib.load(os.path.join(controller_dir, SCALER_PATH))
    hog_p  = joblib.load(os.path.join(controller_dir, HOG_PARAMS_PATH))

    # Verificación temprana: si el modelo o el scaler no concuerdan con la
    # dimensión esperada del HOG, mejor crashear AHORA con un mensaje claro
    # que silenciosamente clasificar mal en el lazo principal.
    n_feats_clf = clf.coef_.shape[1] if hasattr(clf, "coef_") else None
    if n_feats_clf is not None and n_feats_clf != SVM_FEATURE_DIM:
        raise ValueError(
            f"SVM entrenado con {n_feats_clf} features, esperaba {SVM_FEATURE_DIM}. "
            "Re-exporta el modelo desde el notebook."
        )
    if hog_p.get("feature_dim") != SVM_FEATURE_DIM:
        raise ValueError(
            f"hog_params.joblib tiene feature_dim={hog_p.get('feature_dim')}, "
            f"esperaba {SVM_FEATURE_DIM}."
        )
    return clf, scaler, hog_p


# Lista mutable como flag de "primera verificación hecha". Se usa por defecto
# en el argumento para mantener estado entre llamadas sin globales explícitas.
_HOG_DIM_CHECKED = [False]


def classify_pedestrian(bgr_frame, clf, scaler, hog_params):
    """Sliding window HOG + SVM sobre la cámara. Devuelve True si alguna
    ventana se clasifica como peatón.

    Optimización: short-circuit en el primer hit (no necesitamos bounding
    boxes ni NMS, solo la decisión binaria 'hay peatón al frente').

    CRÍTICO: el preprocesamiento debe ser IDÉNTICO al del notebook.
    El notebook hace cv2.resize(patch, (64, 128)) (width=64, height=128).
    En skimage el patch correspondiente tiene forma (128, 64). Si esto
    difiere, el vector HOG tendrá otra dimensión y el SVM dará basura
    silenciosa o scaler.transform lanzará ValueError.

    La verificación de dimensión se hace una sola vez (la primera ventana
    del primer frame con obstáculo) usando _HOG_DIM_CHECKED como bandera.
    """
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < SVM_WINDOW_H:
        return False
    for x in range(0, w - SVM_WINDOW_W + 1, SVM_STRIDE_X):
        patch = gray[0:SVM_WINDOW_H, x:x + SVM_WINDOW_W]
        # Garantía de tamaño: si por algún motivo el recorte no es 128x64,
        # redimensionamos. cv2.resize toma (w, h), inverso a numpy.
        if patch.shape != (SVM_WINDOW_H, SVM_WINDOW_W):
            patch = cv2.resize(patch, (SVM_WINDOW_W, SVM_WINDOW_H))
        feats = hog(
            patch,
            orientations    = hog_params["orientations"],
            pixels_per_cell = hog_params["pixels_per_cell"],
            cells_per_block = hog_params["cells_per_block"],
            block_norm      = hog_params["block_norm"],
            transform_sqrt  = hog_params["transform_sqrt"],
            feature_vector  = True,
        )
        # Verificación de consistencia HOG <-> entrenamiento (una sola vez).
        if not _HOG_DIM_CHECKED[0]:
            if feats.shape[0] != SVM_FEATURE_DIM:
                raise AssertionError(
                    f"HOG dim mismatch: got {feats.shape[0]}, "
                    f"expected {SVM_FEATURE_DIM}. Revisa hog_params.joblib vs notebook."
                )
            _HOG_DIM_CHECKED[0] = True
        feats_scaled = scaler.transform(feats.reshape(1, -1))
        if clf.predict(feats_scaled)[0] == 1:
            return True
    return False


# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

def main():
    # --- Inicialización de Webots ---
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    # Cámara: 256x128 en el mundo de la Actividad 3.1 (vs 128x64 de la 2.1)
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    img_width = camera.getWidth()
    img_height = camera.getHeight()
    print(f"[INIT] Cámara: {img_width}x{img_height}, setpoint={PID_SETPOINT_PX}px "
          f"(offset {PID_SETPOINT_PX - img_width/2:+.0f} px desde el centro)")

    # Display para visualización en el simulador
    display_img = Display("display_image")

    # Teclado (lo conservamos por consistencia con Actividad 2.1)
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # --- Inicialización del PID (con ganancias retunadas para Actividad 3.1) ---
    pid = PIDController(kp=PID_KP, ki=PID_KI, kd=PID_KD)
    print(f"[INIT] PID gains: Kp={PID_KP}, Ki={PID_KI}, Kd={PID_KD}")

    # --- Inicialización del LiDAR (Stage 2) ---
    lidar, lidar_central = init_lidar(robot, timestep)
    print(f"[INIT] LiDAR '{LIDAR_DEVICE_NAME}' habilitado, "
          f"FOV central = {LIDAR_FOV_CENTRAL_DEG}° de {LIDAR_FOV_TOTAL_DEG}°, "
          f"umbral = {LIDAR_MAX_DETECT_M} m")

    # --- Carga del modelo de peatones (Stage 3) ---
    # Resolución de rutas: directorio donde vive este .py, no el cwd.
    controller_dir = os.path.dirname(os.path.abspath(__file__))
    clf, scaler, hog_params = load_pedestrian_svm(controller_dir)
    print(f"[INIT] SVM cargado desde {controller_dir} "
          f"(feature_dim={hog_params['feature_dim']}, "
          f"window={hog_params['window']})")

    # Velocidad inicial (crucero antes de evaluar el primer estado)
    driver.setCruisingSpeed(TARGET_SPEED)

    # --- Variables de estado del loop principal ---
    frame_count = 0
    no_line_count = 0
    last_valid_angle = 0.0
    frames_since_detection = 0
    # Reemplaza el viejo main._prev_angle (atributo de función) por una
    # variable local; cumple con la regla del profesor de evitar código
    # innecesariamente complejo.
    prev_angle = 0.0
    # Estado de la máquina de frenado (Stage 4). None fuerza el primer log
    # de transición al alcanzar el primer estado, incluso si es DRIVING.
    last_state = None
    # Último resultado del SVM (para reutilizar entre frames cuando el SVM
    # se saltea por SVM_RUN_EVERY_N).
    last_svm_is_pedestrian = False
    # Última velocidad de crucero aplicada. Usado para el debounce extendido
    # (FIX 1): la velocidad ahora depende del estado del state-machine AND
    # del estado de la línea (lane_lost_extended), no sólo del estado, así
    # que el debounce no puede engancharse sólo a cambios de estado.
    last_cruising_speed = None

    # --- Loop principal ---
    # IMPORTANTE: usamos driver.step() (no robot.step()). El controlador hereda
    # el patrón del profesor (Car() + Driver() ambos instanciados) pero al
    # cambiar cruising_speed y hazard_flashers cada frame el subsistema Driver
    # acumula comandos pendientes en su buffer interno. wbu_driver_step() es
    # la API correcta para vaciar ese buffer; wb_robot_step() (que es lo que
    # ejecuta robot.step()) solo avanza el tiempo de simulación sin tocar el
    # estado del Driver, lo que provoca un deadlock determinístico tras unos
    # cientos de frames (verificado con faulthandler en GUI y en headless).
    # En Actividad 2.1 esto no se notaba porque el cruising_speed se fijaba
    # una sola vez y los hazard_flashers nunca cambiaban.
    while driver.step() != -1:
        frame_count += 1

        # 1. Captura de imagen (BGR 256x128)
        image_bgr = get_image(camera)

        # 2. Pipeline de detección de carriles (Actividad 2.1 heredada)
        gray = to_grayscale(image_bgr)
        edges = detect_edges(gray)
        edges_roi = apply_roi(edges)
        lines = detect_lines(edges_roi)

        # 3. Cálculo del error de carril
        error, num_valid = compute_error(lines, img_width)

        # 4. PID -> ángulo de dirección. robot.getTime() es tiempo de
        # simulación; correcto incluso si Webots corre en fast-mode.
        current_time = robot.getTime()
        # lane_lost_extended se usa más abajo en el state-machine para reducir
        # la velocidad de crucero cuando la línea lleva mucho rato sin verse
        # (típicamente porque estamos cruzando una intersección).
        lane_lost_extended = False
        if error is None:
            frames_since_detection += 1
            no_line_count += 1
            # FIX 1: Mantener SIEMPRE el último ángulo válido cuando se pierde
            # la línea — el comportamiento previo (resetear a 0 + reset del PID
            # tras el grace period) hacía que el coche atravesara intersecciones
            # en línea recta y terminara fuera del trazado. Si la línea estaba
            # recta justo antes, last_valid_angle ya es ≈ 0; si estaba curva,
            # el coche mantiene la curva, lo cual es lo correcto.
            steering_angle = last_valid_angle
            if frames_since_detection > NO_DETECTION_GRACE_FRAMES:
                lane_lost_extended = True
        else:
            steering_angle = pid.compute(error, current_time)
            last_valid_angle = steering_angle
            frames_since_detection = 0

        # 5. Slew-rate limit (variable local prev_angle, reemplaza al atributo
        # main._prev_angle del controlador de Actividad 2.1)
        max_change = MAX_ANGLE_CHANGE_PER_FRAME
        if steering_angle - prev_angle > max_change:
            steering_angle = prev_angle + max_change
        elif steering_angle - prev_angle < -max_change:
            steering_angle = prev_angle - max_change
        prev_angle = steering_angle

        # 6. Lectura del LiDAR (Stage 2) y clasificación de peatones (Stage 3)
        # Solo invocamos al clasificador SVM cuando el LiDAR confirma un
        # obstáculo cercano. Esto ahorra ~30 inferencias HOG por segundo
        # cuando la vía está despejada, sin perder cobertura cuando importa.
        forward_m = read_forward_distance(lidar, lidar_central)
        obstacle = forward_m < LIDAR_MAX_DETECT_M

        if obstacle:
            # Throttling del SVM: una de cada N veces se vuelve a clasificar;
            # entre tanto se reutiliza el último resultado (los obstáculos
            # cercanos no cambian de tipo entre frames).
            if frame_count % SVM_RUN_EVERY_N == 0:
                last_svm_is_pedestrian = classify_pedestrian(
                    image_bgr, clf, scaler, hog_params
                )
                # Log gated por LiDAR: aparece sólo cuando hay obstáculo
                # cercano, deja rastro claro en consola de cada clasificación.
                tag = "PEATÓN" if last_svm_is_pedestrian else "no peatón (barril?)"
                print(f"[LIDAR {forward_m:.2f} m] {tag}")
            new_state = (STATE_BRAKE_PEDESTRIAN if last_svm_is_pedestrian
                         else STATE_BRAKE_BARREL)
        else:
            # Sin obstáculo cercano: limpiar el resultado para que un peatón
            # detectado hace 3 segundos no siga frenando indefinidamente.
            last_svm_is_pedestrian = False
            new_state = STATE_DRIVING

        # 7. Aplicación de la máquina de estados (Stage 4)
        # En DRIVING usamos crucero normal sin intermitentes.
        # En BRAKE_PEDESTRIAN frenamos sin intermitentes (el peatón puede
        # caminar y reanudaremos en cuanto se aparte del frente).
        # En BRAKE_BARREL frenamos con intermitentes ENCENDIDOS para indicar
        # avería simulada (lo pide la actividad). El supervisor despawnea el
        # barril tras ~10 ticks -> el LiDAR queda libre -> regresamos a DRIVING.
        #
        # FIX 1 (intersecciones): además del estado, la velocidad efectiva en
        # DRIVING depende de si llevamos rato sin ver la línea. Si la línea
        # está perdida más de NO_DETECTION_GRACE_FRAMES, bajamos a
        # LANE_LOST_SPEED_KMH para tener más oportunidad de re-detectar antes
        # de salir del trazado pintado.
        if new_state == STATE_DRIVING:
            target_cruising_speed = (LANE_LOST_SPEED_KMH if lane_lost_extended
                                     else TARGET_SPEED)
            target_hazard = False
        elif new_state == STATE_BRAKE_PEDESTRIAN:
            target_cruising_speed = BRAKE_SPEED_KMH
            target_hazard = False
        else:  # STATE_BRAKE_BARREL
            target_cruising_speed = BRAKE_SPEED_KMH
            target_hazard = True

        # Debounce: solo enviamos por IPC cuando el valor objetivo cambia.
        # Esto cubre el cambio por estado Y el cambio por línea-perdida.
        if target_cruising_speed != last_cruising_speed:
            driver.setCruisingSpeed(target_cruising_speed)
            last_cruising_speed = target_cruising_speed
        # setHazardFlashers solo cambia con el estado (no con la línea),
        # entonces sigue siendo seguro engancharlo al cambio de estado.
        if new_state != last_state:
            driver.setHazardFlashers(target_hazard)
            print(f"[STATE] -> {new_state} (dist={forward_m:.2f} m)")
            last_state = new_state

        # Aplicación del steering (cada frame: el valor depende del PID).
        # El lane keeping sigue activo incluso mientras el coche frena.
        driver.setSteeringAngle(steering_angle)

        # 8. Visualización en el display de Webots
        # Si ENABLE_DISPLAY=False se omite por completo (diagnóstico).
        if ENABLE_DISPLAY:
            debug_image = draw_lines_on_image(image_bgr, lines)
            debug_image = draw_setpoint_line(debug_image)
            display_image_on_webots(display_img, debug_image)

        # 9. Log periódico cada 50 frames (~0.5 s a 10 ms timestep)
        if frame_count % 50 == 0:
            err_str = f"{error:+.2f}" if error is not None else "  N/A"
            n_lines = len(lines) if lines is not None else 0
            print(f"[F{frame_count}] lines={n_lines} valid={num_valid} "
                  f"err={err_str} angle={steering_angle:+.4f} "
                  f"dist={forward_m:.2f} m state={new_state} "
                  f"no_line_total={no_line_count}")


if __name__ == "__main__":
    main()
