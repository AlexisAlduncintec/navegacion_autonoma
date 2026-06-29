"""
================================================================================
Proyecto Final - Conductor autónomo CIL con overrides de seguridad
MR4010 - Navegación Autónoma | Tecnológico de Monterrey | MNA
Equipo 18:
  - Alexis Alduncin Barragán          - A01017478
  - David Rodrigo Alvarado Domínguez  - A01797606
  - Abraham Avila Garcia              - A01795305
  - Jorge Luis Ancheyta Segovia       - A01796354
Profesor: Dr. David Antonio Torres
================================================================================

Controlador externo AUTÓNOMO sobre el BMW X5 del mundo:

    Proyecto_Final/worlds/city_traffic_2025_02.wbt   (controller "<extern>")

Conduce por imitación usando la red de Conditional Imitation Learning (CIL,
Codevilla et al. 2018) entrenada en notebooks/cil_training.ipynb, con una capa
de OVERRIDES DE SEGURIDAD por encima de la predicción del modelo:

    - El modelo CIL predice el ángulo de dirección a partir de la cámara 200x88
      y un comando de navegación de alto nivel (seguir carril / izq / der / recto).
    - Tres sensores nuevos (añadidos al .wbt en la Fase 1) habilitan la seguridad:
        * Camera Recognition  -> clasifica peatones y autobuses por modelo/color.
        * SickLms291 "lidar"  -> distancia frontal (sector central ~25°) para
                                  confirmar peatón cercano y disparar evasión.
        * Radar "radar"        -> rango al vehículo de adelante (car-following).

--------------------------------------------------------------------------------
MÁQUINA DE ESTADOS (mutuamente exclusivos, sólo uno activo por tick)
--------------------------------------------------------------------------------
  CIL_CRUISE        Default. El modelo predice el steering. Velocidad = 25 km/h.
  EMERGENCY_BRAKE   Peatón a < 10 m (Recognition + LiDAR). Velocidad = 0; se
                    mantiene hasta que el peatón despeja el frente.
  EVASION           Autobús/obstáculo al frente. Maniobra basada en yaw (swerve
                    + despeje + restauración de rumbo con el Gyro, estilo 4.2).
                    Reanuda CIL al recuperar el rumbo previo.
  CAR_FOLLOW_SLOW   Vehículo al frente dentro del umbral del radar. Velocidad
                    escalada al gap.
  CAR_FOLLOW_STOP   Vehículo al frente dentro del umbral duro. Velocidad = 0.

PRIORIDAD DE DECISIÓN (cada tick, en este orden; GANA la primera que aplica):
  1. Peatón (Recognition con modelo "pedestrian") Y LiDAR confirma punto frontal
     < 10 m  -> EMERGENCY_BRAKE
  2. Autobús (Recognition por modelo "bus" / color / tamaño) al frente Y LiDAR
     frontal < 8 m -> EVASION (maniobra estilo 4.2)
  3. Radar con blanco a rango < HARD_STOP_DIST (4 m) -> CAR_FOLLOW_STOP
  4. Radar con blanco a rango < SLOW_DIST (10 m) -> CAR_FOLLOW_SLOW
     (velocidad = 25 * (rango - 4) / 6, acotada a [5, 25] km/h)
  5. Nada de lo anterior -> CIL_CRUISE

--------------------------------------------------------------------------------
DECISIONES DE DISEÑO (cada una justificada)
--------------------------------------------------------------------------------
  - Loop usa driver.step() (no robot.step()): el subsistema Driver mantiene un
    buffer interno de comandos que sólo se vacía con driver.step(). Documentado
    desde la Actividad 3.1.
  - display.imageDelete() OBLIGATORIO tras imagePaste(): Webots reserva un slot
    por cada imageNew() y sin liberarlo deadlockea tras ~350 frames. Bug de 3.1.
  - Paridad de preprocesamiento entrenamiento<->inferencia (CRÍTICO): el dataset
    se entrenó con imágenes RGB normalizadas a [0,1] tamaño 200x88 (el notebook
    hace cv2.imread -> BGR2RGB -> resize(200,88) -> /255). Aquí replicamos
    EXACTAMENTE: BGRA -> BGR -> RGB -> resize(200,88) -> /255 -> expand_dims.
    Si se omitiera el BGR2RGB el modelo vería los canales invertidos y daría
    steering basura silenciosa.
  - Índice de comando del modelo: la capa CommandSwitch hace gather sobre una
    salida de 4 ramas, así que el tensor 'command' debe ser el ÍNDICE 0..3, no
    el código alto 2..5. Convertimos cmd_idx = command - 2 (idéntico a la celda 7
    del notebook). Alimentar 2..5 indexaría fuera de rango.
  - EVASION adaptada de la 4.2: el BMW de World #2 NO tiene los 3 DistanceSensor
    laterales que usaba el wall-follower de la 4.2 (la Fase 1 sólo añadió
    Recognition + LiDAR + Radar). Sustituimos el seguimiento de pared por una
    maniobra guiada por yaw: swerve a la izquierda hasta |Δyaw|>=0.6 rad, avanzar
    hasta que el LiDAR frontal despeje (autobús rebasado), y restaurar el rumbo
    con un P sobre el yaw del Gyro (idéntico patrón de captura/restauración de la
    4.2). Documentado como adaptación por falta de sensores laterales.
  - Carga del modelo: se reconstruye la MISMA arquitectura de la celda 5 del
    notebook y se cargan los pesos con load_weights() (formato Keras 3 .weights.h5,
    el mismo patrón que recargaba limpio en la celda 7). No usamos load_model
    porque sólo necesitamos los pesos sobre una arquitectura ya conocida.

--------------------------------------------------------------------------------
MAPA DE TECLADO (el conductor sobreescribe el comando de navegación al vuelo)
--------------------------------------------------------------------------------
  F = FOLLOW_LANE  (command=2, default)   - HUD blanco
  L = TURN LEFT    (command=3)            - HUD cian
  K = TURN RIGHT   (command=4)            - HUD magenta   [NO 'R']
  S = GO STRAIGHT  (command=5)            - HUD amarillo
  SPACE = freno de emergencia manual                     - HUD rojo
  Q = salir limpio

--------------------------------------------------------------------------------
USO (Mac / R2025a)
--------------------------------------------------------------------------------
  # Terminal 1 - Webots
  pkill -KILL -x webots 2>/dev/null; pkill -KILL -f "webots-controller" 2>/dev/null; sleep 2
  cd ~/ITESM/navegacion_autonoma
  /Applications/Webots.app/Contents/MacOS/webots --mode=realtime --port=1234 \
    Proyecto_Final/worlds/city_traffic_2025_02.wbt

  # Terminal 2 - Controlador
  conda activate webots
  export WEBOTS_HOME=/Applications/Webots.app
  export PYTHONPATH=/Applications/Webots.app/Contents/lib/controller/python
  export PYTHONUNBUFFERED=1
  /Applications/Webots.app/Contents/MacOS/webots-controller \
    --port=1234 --robot-name=vehicle \
    $(which python) -u \
    ~/ITESM/navegacion_autonoma/Proyecto_Final/controllers/autonomous_cil_driver.py
================================================================================
"""

import os
import sys
import math
import time

# Silenciamos los logs de C++ de TensorFlow ANTES de importarlo (oneDNN, etc.).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import cv2

from controller import Display, Keyboard, Camera, Lidar, Radar, Gyro
from vehicle import Car, Driver

import tensorflow as tf
import keras
from keras import layers

# =============================================================================
# CONSTANTES DE CONFIGURACIÓN
# =============================================================================

# --- Velocidad ---
TARGET_SPEED_KMH   = 25.0     # crucero CIL (igual que la recolección)
EVADE_SPEED_KMH    = 15.0     # velocidad durante la maniobra de evasión
MAX_STEERING_ANGLE = 0.5      # rad - límite mecánico del BMW X5
MAX_ANGLE_CHANGE_PER_FRAME = 0.05  # slew-rate del volante (anti-jerk)

# --- Modo híbrido PID + CIL (seguimiento de carril robusto en World #2) ---
# El modelo CIL predice steering casi cero en World #2 (brecha de dominio
# World#1->World#2) y el coche se sale en curvas. En el estado CIL_CRUISE
# mezclamos la predicción del modelo con un PID basado en la línea central
# amarilla detectada en la cámara (patrón de la Actividad 2.1).
HYBRID_MODE   = True
LANE_PID_KP   = 0.4    # ganancia P sobre el error normalizado de la línea
HYBRID_W_CIL  = 0.3    # peso de la predicción del modelo CIL
HYBRID_W_PID  = 0.7    # peso del PID de la línea amarilla

# --- Entrada del modelo CIL (Codevilla) ---
IMG_W = 200
IMG_H = 88

# --- Códigos de comando de navegación (paper §V, idénticos al recolector) ---
CMD_FOLLOW_LANE = 2
CMD_TURN_LEFT   = 3
CMD_TURN_RIGHT  = 4
CMD_GO_STRAIGHT = 5
CMD_TEXT = {
    CMD_FOLLOW_LANE: "FOLLOW_LANE",
    CMD_TURN_LEFT:   "TURN LEFT",
    CMD_TURN_RIGHT:  "TURN RIGHT",
    CMD_GO_STRAIGHT: "GO STRAIGHT",
}
CMD_COLOR = {
    CMD_FOLLOW_LANE: 0xFFFFFF,  # blanco
    CMD_TURN_LEFT:   0x00FFFF,  # cian
    CMD_TURN_RIGHT:  0xFF00FF,  # magenta
    CMD_GO_STRAIGHT: 0xFFFF00,  # amarillo
}

# --- Ruta de los pesos del modelo (resuelta con expanduser para portabilidad) ---
WEIGHTS_PATH = os.path.join(
    os.path.expanduser("~"),
    "ITESM", "navegacion_autonoma", "Proyecto_Final",
    "notebooks", "artifacts_full_dataset", "cil_model.weights.h5",
)

# --- LiDAR Sick LMS 291 (name "lidar", añadido en la Fase 1) ---
LIDAR_DEVICE_NAME     = "lidar"
LIDAR_FOV_TOTAL_DEG   = 180.0
LIDAR_FOV_CENTRAL_DEG = 25.0     # sector frontal útil
LIDAR_MAX_DETECT_M    = 30.0     # más allá es ruido/edificios

# --- Radar (name "radar", añadido en la Fase 1) ---
RADAR_DEVICE_NAME = "radar"

# --- Gyro (name por defecto "gyro") ---
GYRO_DEVICE_NAME = "gyro"

# --- Umbrales de la máquina de estados ---
PED_BRAKE_DIST   = 10.0   # m - peatón confirmado por LiDAR -> freno
BUS_EVADE_DIST   = 8.0    # m - autobús al frente -> evasión
HARD_STOP_DIST   = 4.0    # m - radar: paro total de car-following
SLOW_DIST        = 10.0   # m - radar: car-following lento

# --- Detección de autobús por Recognition ---
# Colores de recognitionColors de los autobuses del mundo (los que los traen);
# vehicle(5) no define recognitionColors, así que también detectamos por el
# string del modelo ("bus") y por tamaño en imagen como respaldo.
BUS_COLORS_RGB = [
    (0.898039, 0.647059, 0.0392157),  # naranja
    (0.878431, 0.105882, 0.141176),   # rojo
    (0.149020, 0.635294, 0.411765),   # verde
]
BUS_COLOR_MATCH_TOL = 0.12   # distancia euclidiana en el cubo RGB [0,1]
BUS_CENTRAL_BAND    = 0.30   # fracción a cada lado del centro = "al frente"

# --- Maniobra de evasión guiada por yaw (adaptación de la 4.2) ---
EVADE_INIT_TURN_RAD  = 0.5    # swerve inicial a la izquierda (negativo en código)
EVADE_MIN_YAW_RAD    = 0.6    # rotación mínima antes de pasar a despeje (~34°)
EVADE_SWERVE_MAX_FR  = 60     # fallback de frames del swerve
BUS_CLEAR_DIST       = 14.0   # m - LiDAR frontal "despejado" (autobús rebasado)
CLEAR_HOLD_FRAMES    = 5      # frames consecutivos despejado para confirmar
EVADE_CLEAR_MAX_FR   = 220    # safety del despeje
HEADING_KP           = 1.5    # P de restauración de rumbo (rad/rad)
HEADING_TOL_RAD      = 0.05   # ~3°, tolerancia de "rumbo restaurado"
RESTORE_MAX_FR       = 220    # safety de la restauración

# --- Estados (string-based, estables en logs) ---
STATE_CIL_CRUISE      = "CIL_CRUISE"
STATE_EMERGENCY_BRAKE = "EMERGENCY_BRAKE"
STATE_EVASION         = "EVASION"
STATE_CAR_FOLLOW_SLOW = "CAR_FOLLOW_SLOW"
STATE_CAR_FOLLOW_STOP = "CAR_FOLLOW_STOP"

# Colores del HUD por estado.
STATE_COLOR = {
    STATE_CIL_CRUISE:      0x00FF00,  # verde
    STATE_EMERGENCY_BRAKE: 0xFF0000,  # rojo
    STATE_EVASION:         0xFF8000,  # naranja
    STATE_CAR_FOLLOW_SLOW: 0xFFFF00,  # amarillo
    STATE_CAR_FOLLOW_STOP: 0xFFFF00,  # amarillo
}

# Subfases internas de la evasión.
EVADE_SWERVE  = "SWERVE"
EVADE_CLEAR   = "CLEAR"
EVADE_RESTORE = "RESTORE"


# =============================================================================
# ARQUITECTURA CIL (idéntica a la celda 5 del notebook -> los pesos cargan por
# topología). Dropout 0.3 en las FC; en inferencia el dropout no actúa, pero la
# arquitectura debe coincidir EXACTAMENTE para que load_weights mapee bien.
# =============================================================================

@keras.saving.register_keras_serializable()
class CommandSwitch(layers.Layer):
    """Switch de Codevilla: por muestra del batch selecciona la salida de la
    rama indicada por el comando. Entradas: (outputs (batch,4), command (batch,)).
    Salida: (batch,)."""
    def call(self, inputs):
        outs, cmd = inputs
        batch_size = tf.shape(outs)[0]
        indices = tf.stack([tf.range(batch_size), cmd], axis=1)
        return tf.gather_nd(outs, indices)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][0],)


def build_image_module(input_shape=(IMG_H, IMG_W, 3)):
    """Módulo de imagen de Codevilla §IV.B: 8 conv + 2 FC."""
    inp = layers.Input(shape=input_shape, name='image_input')
    x = inp
    x = layers.Conv2D(32, 5, strides=2, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(32, 3, strides=1, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(64, 3, strides=1, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(128, 3, strides=1, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(256, 3, strides=1, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(256, 3, strides=1, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation=None)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation=None)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    return keras.Model(inp, x, name='image_module')


def build_branch(features_dim=512, name='branch'):
    """Una rama especialista: 2 FC + salida de steering con tanh -> [-1,1]."""
    inp = layers.Input(shape=(features_dim,))
    x = layers.Dense(256, activation=None)(inp)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation=None)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='tanh', name=f'{name}_steering')(x)
    return keras.Model(inp, out, name=name)


def build_cil_model():
    """CIL ramificado: módulo de imagen compartido + 4 ramas + CommandSwitch."""
    image_inp = layers.Input(shape=(IMG_H, IMG_W, 3), name='image')
    command_inp = layers.Input(shape=(), dtype=tf.int32, name='command')
    image_module = build_image_module()
    features = image_module(image_inp)
    branch_follow   = build_branch(name='branch_follow')
    branch_left     = build_branch(name='branch_left')
    branch_right    = build_branch(name='branch_right')
    branch_straight = build_branch(name='branch_straight')
    out_follow   = branch_follow(features)
    out_left     = branch_left(features)
    out_right    = branch_right(features)
    out_straight = branch_straight(features)
    all_outputs = layers.Concatenate(axis=-1)(
        [out_follow, out_left, out_right, out_straight])
    selected = CommandSwitch(name='command_switch')([all_outputs, command_inp])
    return keras.Model(inputs=[image_inp, command_inp], outputs=selected,
                       name='cil_branched')


def load_cil_model(weights_path):
    """Construye la arquitectura y carga los pesos. Si load_weights estricto
    falla (p.ej. desajuste de nombres entre versiones de Keras), reintenta con
    skip_mismatch=True y AVISA RUIDOSAMENTE qué capas se omitieron (eso sería un
    bug real que hay que reportar, no un fallo silencioso)."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"No se encontraron los pesos del modelo en:\n  {weights_path}\n"
            "Mueve cil_model.weights.h5 a notebooks/artifacts_full_dataset/.")
    model = build_cil_model()
    try:
        model.load_weights(weights_path)
        print(f"[MODEL] Pesos cargados (estricto) desde {weights_path}")
    except Exception as e_strict:
        print(f"[MODEL][WARN] load_weights estricto falló: {e_strict}")
        print("[MODEL][WARN] Reintentando con skip_mismatch=True...")
        model.load_weights(weights_path, skip_mismatch=True)
        print("[MODEL][WARN] *** ALGUNAS CAPAS SE OMITIERON (skip_mismatch) — "
              "REVISAR: la arquitectura del .py podría no coincidir 1:1 con la "
              "del notebook. Las predicciones podrían ser parciales. ***")
    return model


# =============================================================================
# PROCESAMIENTO DE IMAGEN
# =============================================================================

def get_bgr_image(camera):
    """Cámara de Webots devuelve BGRA -> devolvemos BGR (sin alpha), o None si
    aún no hay frame disponible (primer paso)."""
    raw = camera.getImage()
    if raw is None:
        return None
    bgra = np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4))
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


def preprocess_for_cil(bgr):
    """Paridad EXACTA con el notebook: BGR -> RGB -> resize(200,88) -> /255 ->
    expand_dims. Devuelve un ndarray (1, 88, 200, 3) float32."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_W, IMG_H))   # cv2 toma (w, h)
    norm = resized.astype(np.float32) / 255.0
    return np.expand_dims(norm, axis=0)


def compute_pid_steer(bgr, img_width):
    """Steering PID basado en la línea central amarilla (patrón fotométrico de
    la Actividad 2.1). Umbral HSV amarillo -> centro de masa de los píxeles
    amarillos en la mitad inferior de la imagen -> error normalizado -> -KP*error.
    Devuelve (steer, lane_seen). lane_seen=False si no se ve la línea (el híbrido
    cae de vuelta al modelo CIL en ese caso)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([40, 255, 255]))
    h = mask.shape[0]
    lower = mask[h // 2:, :]                      # sólo la mitad inferior
    m = cv2.moments(lower)
    if m["m00"] < 1e-3:
        return 0.0, False                          # no hay línea amarilla visible
    cm_x = m["m10"] / m["m00"]
    error = (img_width / 2.0 - cm_x) / (img_width / 2.0)
    steer = -LANE_PID_KP * error
    return max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, steer)), True


# =============================================================================
# SENSORES
# =============================================================================

def init_lidar(robot, timestep):
    """Inicializa el SickLms291 y precalcula el slice del sector central
    frontal. Mismo patrón que 3.1/4.2 (enable + enablePointCloud)."""
    lidar = robot.getDevice(LIDAR_DEVICE_NAME)
    lidar.enable(timestep)
    lidar.enablePointCloud()
    horiz_res = lidar.getHorizontalResolution()
    central_n = max(1, int(horiz_res * LIDAR_FOV_CENTRAL_DEG / LIDAR_FOV_TOTAL_DEG))
    mid = horiz_res // 2
    half = central_n // 2
    print(f"[INIT] LiDAR '{LIDAR_DEVICE_NAME}': horiz_res={horiz_res}, "
          f"sector central={central_n} rayos")
    return lidar, slice(mid - half, mid + half)


def read_forward_distance(lidar, central_slice):
    """Distancia mínima finita en el sector frontal, acotada a LIDAR_MAX_DETECT_M.
    inf si no hay impacto (frente despejado)."""
    ranges = lidar.getRangeImage()[central_slice]
    finite = [r for r in ranges if math.isfinite(r) and r <= LIDAR_MAX_DETECT_M]
    return min(finite) if finite else float("inf")


def init_radar(robot, timestep):
    """Inicializa el Radar frontal."""
    radar = robot.getDevice(RADAR_DEVICE_NAME)
    radar.enable(timestep)
    print(f"[INIT] Radar '{RADAR_DEVICE_NAME}' habilitado")
    return radar


def read_radar_nearest(radar):
    """Rango (m) al blanco más cercano del radar; inf si no hay blancos.
    RadarTarget expone .distance (m) entre otros campos."""
    targets = radar.getTargets()
    if not targets:
        return float("inf")
    dists = []
    for t in targets:
        try:
            dists.append(t.distance)
        except AttributeError:
            continue
    return min(dists) if dists else float("inf")


def init_gyro(robot, timestep):
    gyro = robot.getDevice(GYRO_DEVICE_NAME)
    gyro.enable(timestep)
    print(f"[INIT] Gyro '{GYRO_DEVICE_NAME}' habilitado")
    return gyro


def _safe_pair(maybe_ptr):
    """getPositionOnImage/getSizeOnImage pueden ser lista o ctypes pointer."""
    try:
        return (maybe_ptr[0], maybe_ptr[1])
    except (TypeError, IndexError):
        return (0, 0)


def _color_matches_bus(rgb):
    """True si el color RGB cae dentro de la tolerancia de algún color de bus."""
    r, g, b = rgb
    for cr, cg, cb in BUS_COLORS_RGB:
        if math.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) < BUS_COLOR_MATCH_TOL:
            return True
    return False


def scan_recognition(camera, img_width):
    """Recorre los CameraRecognitionObject y devuelve (ped_ahead, bus_ahead, ped_min_dist).

    ped_ahead: hay al menos un objeto cuyo modelo contiene 'pedestrian'.
    bus_ahead: hay al menos un autobús (modelo 'bus' / color de bus / tamaño
               grande en imagen) cuyo centro cae en la banda central de la
               imagen (= delante del coche, no a un lado).

    getColors() devuelve un ctypes pointer; se lee con getNumberOfColors()."""
    ped_ahead = False
    ped_min_dist = float("inf")
    bus_ahead = False
    central_lo = img_width * BUS_CENTRAL_BAND
    central_hi = img_width * (1.0 - BUS_CENTRAL_BAND)
    for obj in camera.getRecognitionObjects():
        try:
            model = (obj.getModel() or "").lower()
        except Exception:
            model = ""
        if "pedestrian" in model:
            ped_ahead = True
            # Distancia tomada de la posición 3D del objeto de Recognition
            # (más robusta que el LiDAR para un peatón delgado: los rayos del
            # sector central del LiDAR pueden pasar a los lados sin tocarlo).
            try:
                p = obj.getPosition()
                ped_min_dist = min(ped_min_dist,
                                   math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2))
            except Exception:
                pass
            continue
        # ¿Es un autobús? El modelo es EXACTAMENTE "bus" (NO "bus stop", que
        # también contiene la subcadena) o el color coincide con un
        # recognitionColors de autobús. Se ELIMINÓ el fallback por tamaño en
        # imagen: en World #2 (denso en edificios) cualquier edificio grande al
        # frente excedía el umbral de píxeles y disparaba evasiones falsas
        # (hallazgo del arnés de prueba autónomo, iter 1).
        is_bus = (model.strip() == "bus")
        if not is_bus:
            try:
                if obj.getNumberOfColors() >= 1:
                    cptr = obj.getColors()
                    if _color_matches_bus((cptr[0], cptr[1], cptr[2])):
                        is_bus = True
            except Exception:
                pass
        if is_bus:
            cx = _safe_pair(obj.getPositionOnImage())[0]
            if central_lo <= cx <= central_hi:
                bus_ahead = True
    return ped_ahead, bus_ahead, ped_min_dist


# =============================================================================
# VISUALIZACIÓN / HUD
# =============================================================================

def resolve_display(robot):
    """Localiza el Display de forma robusta ("display" por defecto en World #2,
    "display_image" en mundos viejos)."""
    for name in ("display", "display_image"):
        dev = robot.getDevice(name)
        if dev is not None:
            return dev
    return None


def paste_camera_to_display(display, bgr, dw, dh):
    """Pega la cámara (redimensionada al Display) y libera el slot (imageDelete)."""
    resized = cv2.resize(bgr, (dw, dh))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    ref = display.imageNew(rgb.tobytes(), Display.RGB, width=dw, height=dh)
    display.imagePaste(ref, 0, 0, False)
    display.imageDelete(ref)   # evita la fuga de slots de 3.1


def draw_hud(display, dw, dh, command, state, model_steer, speed_kmh,
             steer_applied, radar_m, lidar_m, pid_steer=0.0):
    """Overlay sobre la imagen ya pegada (drawText de Webots, no OpenCV).
    Layout pensado para el Display 200x150 del BMW."""
    # Arriba-izquierda: comando activo (color-coded).
    display.setColor(CMD_COLOR.get(command, 0xFFFFFF))
    display.drawText(f"CMD: {CMD_TEXT.get(command, '?')}", 4, 4)
    # Arriba-derecha: estado (color-coded).
    display.setColor(STATE_COLOR.get(state, 0xFFFFFF))
    display.drawText(f"STATE: {state}", max(4, dw - 116), 4)
    # Centro: predicción cruda del modelo + mezcla híbrida.
    display.setColor(0xFFFFFF)
    display.drawText(f"MODEL_STEER: {model_steer:+.3f}", 4, dh // 2 - 8)
    if HYBRID_MODE:
        display.drawText(f"PID:{pid_steer:+.3f}  CIL:{model_steer:+.3f}", 4, dh // 2 + 6)
    # Abajo-izquierda: velocidad + steering aplicado.
    display.drawText(f"spd:{speed_kmh:4.1f}  steer_applied:{steer_applied:+.3f}",
                     4, dh - 28)
    # Abajo-derecha (en la práctica abajo-izquierda por ancho): radar + lidar.
    radar_str = f"{radar_m:4.1f}" if math.isfinite(radar_m) else " inf"
    lidar_str = f"{lidar_m:4.1f}" if math.isfinite(lidar_m) else " inf"
    display.drawText(f"radar:{radar_str}m  lidar_min:{lidar_str}m", 4, dh - 14)


# =============================================================================
# CONTROL DE LA EVASIÓN (sub-FSM guiada por yaw, adaptación de la 4.2)
# =============================================================================

def restore_heading_steering(saved_yaw, yaw):
    """P sobre el error de yaw (envuelto a [-pi,pi]) para volver al rumbo previo.
    Convención Driver: steering>0 = derecha = yaw decrece."""
    raw = yaw - saved_yaw
    err = math.atan2(math.sin(raw), math.cos(raw))
    cmd = HEADING_KP * err
    return max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, cmd))


def step_evasion(ev, yaw, lidar_front_m):
    """Avanza la sub-FSM de evasión un tick. 'ev' es un dict mutable de contexto.
    Devuelve (steering, speed_kmh, done). done=True -> volver a CIL_CRUISE."""
    ev["frames"] += 1
    phase = ev["phase"]

    if phase == EVADE_SWERVE:
        # Swerve a la izquierda (negativo) hasta rotar EVADE_MIN_YAW_RAD o timeout.
        raw = yaw - ev["saved_yaw"]
        yaw_delta = abs(math.atan2(math.sin(raw), math.cos(raw)))
        if yaw_delta >= EVADE_MIN_YAW_RAD or ev["frames"] >= EVADE_SWERVE_MAX_FR:
            ev["phase"] = EVADE_CLEAR
            ev["frames"] = 0
            ev["clear_streak"] = 0
            print(f"[EVASION] SWERVE->CLEAR (yaw_delta={yaw_delta:.2f} rad)")
        return -EVADE_INIT_TURN_RAD, EVADE_SPEED_KMH, False

    if phase == EVADE_CLEAR:
        # Avanzar (volante neutro) hasta que el LiDAR frontal despeje N frames.
        if lidar_front_m >= BUS_CLEAR_DIST:
            ev["clear_streak"] += 1
        else:
            ev["clear_streak"] = 0
        if ev["clear_streak"] >= CLEAR_HOLD_FRAMES or ev["frames"] >= EVADE_CLEAR_MAX_FR:
            ev["phase"] = EVADE_RESTORE
            ev["frames"] = 0
            print(f"[EVASION] CLEAR->RESTORE (lidar={lidar_front_m:.1f} m)")
        return 0.0, EVADE_SPEED_KMH, False

    # EVADE_RESTORE: P de rumbo hasta volver al yaw guardado.
    raw = yaw - ev["saved_yaw"]
    err = abs(math.atan2(math.sin(raw), math.cos(raw)))
    if err < HEADING_TOL_RAD or ev["frames"] >= RESTORE_MAX_FR:
        print(f"[EVASION] RESTORE done (err={err:.3f} rad) -> CIL_CRUISE")
        return 0.0, EVADE_SPEED_KMH, True
    return restore_heading_steering(ev["saved_yaw"], yaw), EVADE_SPEED_KMH, False


# =============================================================================
# LÓGICA DE DECISIÓN DE ESTADO (prioridad documentada arriba)
# =============================================================================

def decide_state(state, ev_active, manual_brake, ped_ahead, bus_ahead,
                 lidar_front_m, radar_near_m, ped_dist_m=float("inf")):
    """Decide el estado del tick. La evasión es 'sticky': una vez activa corre
    su sub-FSM hasta terminar (salvo que un peatón fuerce el freno: la seguridad
    del peatón gana siempre). El freno manual (SPACE) también gana sobre todo."""
    if manual_brake:
        return STATE_EMERGENCY_BRAKE
    # Prioridad 1: peatón cercano -> freno (gana incluso en evasión).
    # La distancia es la mínima entre la posición 3D del Recognition y el LiDAR
    # (el LiDAR puede no tocar un peatón delgado; el Recognition sí lo ubica).
    if ped_ahead and min(ped_dist_m, lidar_front_m) < PED_BRAKE_DIST:
        return STATE_EMERGENCY_BRAKE
    # Si hay una evasión en curso, mantenerla hasta que su sub-FSM diga 'done'.
    if ev_active:
        return STATE_EVASION
    # Prioridad 2: autobús al frente -> evasión.
    if bus_ahead and lidar_front_m < BUS_EVADE_DIST:
        return STATE_EVASION
    # Prioridad 3/4: car-following por radar.
    if radar_near_m < HARD_STOP_DIST:
        return STATE_CAR_FOLLOW_STOP
    if radar_near_m < SLOW_DIST:
        return STATE_CAR_FOLLOW_SLOW
    # Prioridad 5: crucero CIL.
    return STATE_CIL_CRUISE


def car_follow_slow_speed(radar_near_m):
    """Velocidad escalada al gap: 25*(rango-4)/6, acotada a [5, 25] km/h."""
    speed = TARGET_SPEED_KMH * (radar_near_m - HARD_STOP_DIST) / (SLOW_DIST - HARD_STOP_DIST)
    return max(5.0, min(TARGET_SPEED_KMH, speed))


# =============================================================================
# TECLADO
# =============================================================================

def handle_keyboard(keyboard, st, now):
    """Lee las teclas del tick. Devuelve (manual_brake, quit)."""
    keys = set()
    k = keyboard.getKey()
    while k != -1:
        keys.add(k & 0xFFFF)
        k = keyboard.getKey()

    manual_brake = ord(' ') in keys
    quit_requested = False

    def pressed(code):
        if code in keys and (now - st["last_press"].get(code, 0.0) >= 0.1):
            st["last_press"][code] = now
            return True
        return False

    if pressed(ord('F')):
        st["command"] = CMD_FOLLOW_LANE
    if pressed(ord('L')):
        st["command"] = CMD_TURN_LEFT
    if pressed(ord('K')):
        st["command"] = CMD_TURN_RIGHT
    if pressed(ord('S')):
        st["command"] = CMD_GO_STRAIGHT
    if pressed(ord('Q')):
        quit_requested = True
    return manual_brake, quit_requested


# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

def main():
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())
    dt_s = timestep / 1000.0

    # --- Cámara + Recognition ---
    camera = robot.getDevice("camera")
    if camera is None:
        print("[ERROR] Cámara 'camera' no encontrada. Abortando.")
        sys.exit(1)
    camera.enable(timestep)
    camera.recognitionEnable(timestep)
    img_width = camera.getWidth()
    img_height = camera.getHeight()
    print(f"[INIT] Cámara {img_width}x{img_height}, recognition habilitado")

    # --- Display ---
    display = resolve_display(robot)
    if display is None:
        print("[ERROR] Display no encontrado (HUD imposible). Abortando.")
        sys.exit(1)
    dw, dh = display.getWidth(), display.getHeight()
    print(f"[INIT] Display {dw}x{dh}")

    # --- Sensores nuevos (Fase 1) ---
    lidar, lidar_central = init_lidar(robot, timestep)
    radar = init_radar(robot, timestep)
    gyro = init_gyro(robot, timestep)

    # --- Teclado ---
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # --- Modelo CIL ---
    print(f"[INIT] Cargando modelo CIL desde {WEIGHTS_PATH} ...")
    model = load_cil_model(WEIGHTS_PATH)
    print(f"[INIT] Modelo CIL listo ({model.count_params():,} parámetros)")

    # --- Estado del controlador ---
    st = {"command": CMD_FOLLOW_LANE, "last_press": {}}
    state = STATE_CIL_CRUISE
    prev_angle = 0.0
    yaw_z = 0.0           # yaw integrado del gyro (rad)
    ev = None             # contexto de la evasión (None = sin evasión activa)
    frame_count = 0

    driver.setCruisingSpeed(TARGET_SPEED_KMH)

    print("[INIT] KEYBOARD: F=follow L=left K=right S=straight SPACE=brake Q=quit")
    print("[INIT] >>> READY. Conductor autónomo CIL activo.")

    # --- Loop principal (driver.step, no robot.step) ---
    while driver.step() != -1:
        frame_count += 1

        # 1. Cámara (BGR).
        bgr = get_bgr_image(camera)
        if bgr is None:
            continue

        # 2. Teclado.
        now = time.time()
        manual_brake, quit_requested = handle_keyboard(keyboard, st, now)
        if quit_requested:
            print(f"[QUIT] Sesión terminada en frame {frame_count}.")
            break

        # 3. Sensores: yaw, lidar frontal, radar, recognition.
        yaw_z += gyro.getValues()[2] * dt_s
        lidar_front_m = read_forward_distance(lidar, lidar_central)
        radar_near_m = read_radar_nearest(radar)
        ped_ahead, bus_ahead, ped_dist_m = scan_recognition(camera, img_width)

        # 4. Predicción CIL (siempre se calcula; se aplica sólo en CIL_CRUISE).
        # Índice de comando = command - 2 (0..3) para la rama del CommandSwitch.
        cmd_idx = np.array([st["command"] - 2], dtype=np.int32)
        img_in = preprocess_for_cil(bgr)
        pred = model({"image": img_in, "command": cmd_idx}, training=False)
        model_steer = float(np.reshape(np.asarray(pred), -1)[0])

        # 4b. PID de línea amarilla (modo híbrido). Se calcula cada tick para el HUD.
        if HYBRID_MODE:
            pid_steer, lane_seen = compute_pid_steer(bgr, img_width)
        else:
            pid_steer, lane_seen = 0.0, False

        # 5. Decisión de estado (prioridad documentada).
        new_state = decide_state(state, ev is not None, manual_brake,
                                 ped_ahead, bus_ahead, lidar_front_m, radar_near_m,
                                 ped_dist_m)

        # Hooks de entrada/salida de la evasión.
        if new_state == STATE_EVASION and ev is None:
            # Iniciar evasión: capturar el rumbo actual (snapshot del yaw).
            ev = {"phase": EVADE_SWERVE, "frames": 0, "clear_streak": 0,
                  "saved_yaw": yaw_z}
            print(f"[STATE] {state} -> EVASION  (yaw snapshot={yaw_z:+.3f} rad, "
                  f"lidar={lidar_front_m:.1f} m)")
        elif new_state != STATE_EVASION and ev is not None:
            # Un override (peatón/freno manual) interrumpió la evasión.
            print(f"[STATE] EVASION interrumpida por {new_state}")
            ev = None
        elif new_state != state:
            print(f"[STATE] {state} -> {new_state}  "
                  f"(lidar={lidar_front_m:.1f} m, radar={radar_near_m:.1f} m, "
                  f"ped={ped_ahead}, bus={bus_ahead})")
        state = new_state

        # 6. Actuación por estado.
        if state == STATE_CIL_CRUISE:
            # Híbrido: 30% CIL + 70% PID de línea amarilla (si se ve la línea).
            # Sin línea visible, se usa sólo la predicción del modelo.
            if HYBRID_MODE and lane_seen:
                steering = max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE,
                               HYBRID_W_CIL * model_steer + HYBRID_W_PID * pid_steer))
            else:
                steering = max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, model_steer))
            target_speed = TARGET_SPEED_KMH

        elif state == STATE_EMERGENCY_BRAKE:
            # Mantener la última dirección razonable (recta) y frenar.
            steering = 0.0
            target_speed = 0.0
            ev = None  # cualquier evasión queda cancelada mientras frenamos

        elif state == STATE_EVASION:
            steering, target_speed, done = step_evasion(ev, yaw_z, lidar_front_m)
            if done:
                ev = None
                state = STATE_CIL_CRUISE

        elif state == STATE_CAR_FOLLOW_STOP:
            steering = max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, model_steer))
            target_speed = 0.0

        elif state == STATE_CAR_FOLLOW_SLOW:
            # Mantenemos la dirección del modelo, sólo escalamos la velocidad.
            steering = max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, model_steer))
            target_speed = car_follow_slow_speed(radar_near_m)

        else:
            steering = 0.0
            target_speed = TARGET_SPEED_KMH

        # 7. Slew-rate limit del volante (anti-jerk).
        delta = steering - prev_angle
        if delta > MAX_ANGLE_CHANGE_PER_FRAME:
            steering = prev_angle + MAX_ANGLE_CHANGE_PER_FRAME
        elif delta < -MAX_ANGLE_CHANGE_PER_FRAME:
            steering = prev_angle - MAX_ANGLE_CHANGE_PER_FRAME
        prev_angle = steering

        # 8. Aplicar al vehículo.
        driver.setSteeringAngle(steering)
        driver.setCruisingSpeed(target_speed)

        # 9. HUD.
        paste_camera_to_display(display, bgr, dw, dh)
        draw_hud(display, dw, dh, st["command"], state, model_steer,
                 target_speed, steering, radar_near_m, lidar_front_m, pid_steer)

        # 10. Log periódico (cada 50 frames).
        if frame_count % 50 == 0:
            lidar_str = f"{lidar_front_m:5.1f}" if math.isfinite(lidar_front_m) else "  inf"
            radar_str = f"{radar_near_m:5.1f}" if math.isfinite(radar_near_m) else "  inf"
            print(f"[F{frame_count}] state={state:<15s} cmd={CMD_TEXT[st['command']]:<11s} "
                  f"model_steer={model_steer:+.3f} applied={steering:+.3f} "
                  f"spd={target_speed:4.1f} lidar={lidar_str}m radar={radar_str}m "
                  f"ped={int(ped_ahead)} bus={int(bus_ahead)} yaw={yaw_z:+.2f}")


if __name__ == "__main__":
    main()
