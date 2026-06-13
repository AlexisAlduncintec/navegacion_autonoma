"""
================================================================================
Actividad 4.2 - Evasión de obstáculos con seguimiento de pared
MR4010 - Navegación Autónoma | Tecnológico de Monterrey | MNA
Equipo 18:
  - Alexis Alduncin Barragán          - A01017478
  - David Rodrigo Alvarado Domínguez  - A01797606
  - Abraham Avila Garcia              - A01795305
  - Jorge Luis Ancheyta Segovia       - A01796354
Profesor: Dr. David Antonio Torres
================================================================================

Este controlador externo combina cuatro subsistemas clásicos (sin ML) sobre el
BMW X5 del mundo MR4010_Actividad_4_2/worlds/city_2025a.wbt:

    1. Seguimiento de carril (HEREDADO de Actividad 2.1): HSV-amarillo -> Canny
       -> ROI -> HoughLinesP -> PID -> Driver.setSteeringAngle()
    2. Reconocimiento de cámara (Recognition node de Webots): identifica los 4
       autobuses estáticos por su recognitionColors único (azul, rojo, magenta,
       verde) para confirmar que el obstáculo frontal es un autobús.
    3. LiDAR Sick LMS 291 frontal: distancia central (~25°) para disparar el
       cambio de estado a evasión cuando hay un autobús cerca.
    4. Sensores de distancia laterales (3, lado derecho del BMW) + Gyro para
       seguimiento de pared derecha y restauración del rumbo previo a la evasión.

Máquina de estados (4 estados, string-based):
    LANE  -- LiDAR<TRIGGER + recognition -->  EVADE_START  (guarda yaw_z + bus id)
    EVADE_START  -- bus en x>65% del frame (Recognition) o dsF ve flanco -->  EVADE_WALL
    EVADE_WALL  -- ds_right_rear libre N frames -->  RESTORE_HEADING
    RESTORE_HEADING  -- |yaw - saved| < TOL  -->  LANE

Stage actual: A (sólo seguimiento de carril; resto de estados se añaden en D).

Decisiones de diseño respecto a Actividad 2.1 (cada cambio justificado inline):
    - Loop usa driver.step() en vez de robot.step(). En 3.1 documentamos que
      Driver mantiene un buffer interno de comandos que sólo se vacía cuando
      llamas driver.step(); con robot.step() los comandos se acumulan y el
      coche reacciona con un frame de retraso. Lo aprendimos por las malas.
    - display.imageDelete() es OBLIGATORIO después de imagePaste(). Webots
      reserva un slot interno por cada imageNew() y si no se libera, después
      de ~350 iteraciones (≈11s a timestep 32ms) wbu_driver_step() se queda
      sin slots y deadlockea sin error visible. Bug encontrado en 3.1.
    - prev_angle como variable local en vez de main._prev_angle (atributo de
      función): más limpio y el profesor penaliza código innecesariamente
      complejo.

Mundo: MR4010_Actividad_4_2/worlds/city_2025a.wbt (controller "<extern>")

Tuning inicial (todas las constantes documentadas inline, listas para ajustar
tras correr la sim - ver bloque "CONSTANTES DE LA MÁQUINA DE ESTADOS"):
    LIDAR_BUS_TRIGGER_M     = 8.0   # disparo de evasión
    RIGHT_SETPOINT_M        = 1.2   # holgura objetivo del flanco al bus
    SIDE_KP                 = 0.6   # ganancia P de seguimiento de pared
    RIGHT_FREE_M            = 4.5   # umbral de "rear libre"
    RIGHT_FREE_HOLD_FRAMES  = 6     # debounce del rear (=192 ms a 32 ms ts)
    HEADING_KP              = 1.5   # ganancia P de restauración de rumbo
    HEADING_TOLERANCE_RAD   = 0.05  # ≈ 3°, tolerancia para llamarlo restaurado
    EVADE_INIT_TURN_RAD     = 0.5   # swerve inicial a la izquierda (iter 6)
    EVADE_SPEED_KMH         = 15.0  # velocidad durante EVADE_* y RESTORE (iter 6)

Síntomas y ajustes esperados (afinación iterativa contra simulación):
    - El coche raspa el lado derecho del autobús durante EVADE_WALL ->
      subir RIGHT_SETPOINT_M (1.2 -> 1.5 o 1.8).
    - El coche se aleja demasiado del autobús -> bajar SIDE_KP (0.6 -> 0.4) o
      bajar RIGHT_SETPOINT_M.
    - Oscila izquierda/derecha durante EVADE_WALL -> bajar SIDE_KP, considerar
      añadir término D (no incluido en v1).
    - Sale de EVADE_WALL antes de rebasar el autobús -> subir RIGHT_FREE_M
      (4.5 -> 4.8) o RIGHT_FREE_HOLD_FRAMES (6 -> 10).
    - Sobrepasa el rumbo objetivo en RESTORE -> bajar HEADING_KP (1.5 -> 1.0).
    - Tarda mucho en restaurar -> subir HEADING_KP (1.5 -> 2.0).

Uso (Mac / VS Code terminal):
    export WEBOTS_HOME=/Applications/Webots.app
    export PYTHONPATH=/Applications/Webots.app/Contents/lib/controller/python
    export WEBOTS_PYTHON_EXECUTABLE=$(which python)
    conda activate webots
    cd /Users/alexisalduncin/ITESM/navegacion_autonoma/SDC_webots
    python autonomous_evasion_controller.py
"""

from controller import Display, Keyboard, Robot, Camera, Lidar, Gyro
from vehicle import Car, Driver
import math
import numpy as np
import cv2

# =============================================================================
# CONSTANTES DE CONFIGURACIÓN
# =============================================================================

# --- Velocidad ---
# La actividad exige una velocidad mínima de 50 km/h durante el seguimiento de
# carril. Bajamos a EVADE_SPEED_KMH durante evasión y restauración: girar el
# volante a 50 km/h pierde tracción en el modelo Ackermann del BMW X5.
TARGET_SPEED = 50.0          # km/h - estado LANE
# iter 6: 20 -> 15. Más lento = más rotación de yaw por metro recorrido durante
# el swerve inicial, para que el flanco derecho alcance a "mirar" al autobús.
EVADE_SPEED_KMH = 15.0       # km/h - estados EVADE_* y RESTORE_HEADING
MAX_STEERING_ANGLE = 0.5     # rad - límite mecánico del BMW X5

# --- Parámetros de Canny --- (sin cambios respecto a Actividad 2.1)
CANNY_LOW = 50
CANNY_HIGH = 150

# --- Parámetros de HoughLinesP --- (sin cambios respecto a Actividad 2.1)
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 20          # 20 votos
HOUGH_MIN_LINE_LEN = 15       # 15 px
HOUGH_MAX_LINE_GAP = 20       # 20 px

# --- Filtro de líneas casi horizontales --- (sin cambios respecto a 2.1)
HORIZONTAL_SLOPE_THRESHOLD = 0.05

# --- Ganancias del PID de carril --- (heredado de 2.1, no de 3.1)
# Volvemos a las ganancias de 2.1 porque este mundo no tiene curvas tan cerradas
# como el de 3.1 y el comportamiento original era estable para 50 km/h.
PID_KP = 0.008
PID_KI = 0.0
PID_KD = 0.004

# --- Slew rate limit del volante --- (sin cambios)
MAX_ANGLE_CHANGE_PER_FRAME = 0.05

# --- Comportamiento ante pérdida de línea --- (sin cambios)
DEFAULT_STEERING_ANGLE = 0.0
NO_DETECTION_GRACE_FRAMES = 20

# =============================================================================
# CONSTANTES DE SENSORES (Stage B)
# =============================================================================

# --- LiDAR Sick LMS 291 frontal ---
# La PROTO instalada en sensorsSlotFront tiene el nombre por defecto del PROTO,
# que es "Sick LMS 291" (con espacios y mayúsculas). En 3.1 lo confirmamos.
LIDAR_DEVICE_NAME = "Sick LMS 291"
# El Sick LMS 291 reporta 180 rayos sobre 180°. Sólo nos interesa el sector
# central (~25°) para detectar obstáculos directamente frente al BMW.
LIDAR_FOV_TOTAL_DEG    = 180.0
LIDAR_FOV_CENTRAL_DEG  = 25.0
# Limitamos el rango efectivo: el LiDAR ve hasta 80 m, pero más allá de 20 m
# las detecciones son ruido o edificios del fondo.
LIDAR_MAX_DETECT_M     = 20.0

# --- Cámara con Recognition ---
# El Recognition node está habilitado en el .wbt (max 10 objetos, occlusion=1,
# rango 30 m). En el controlador sólo hay que llamar recognitionEnable().
# Los 4 autobuses tienen recognitionColors = color visual (auto-derivado por el
# template del Bus.proto), así que los podemos identificar por su color.
BUS_COLORS_RGB = {
    "vehicle(1)_blue":    (0.0313726, 0.121569, 0.419608),  # azul oscuro
    "vehicle(2)_red":     (1.0,       0.0,       0.0),       # rojo
    "vehicle(3)_magenta": (0.862745,  0.541176,  0.866667),  # magenta
    "vehicle(4)_green":   (0.180392,  0.760784,  0.494118),  # verde
}
# Tolerancia para emparejar el color reportado por Recognition con el catálogo.
# Distancia euclidiana en el cubo RGB (valores 0..1).
BUS_COLOR_MATCH_TOL = 0.10

# --- Gyro ---
# El Gyro tiene name "gyro" (lo añadimos en el .wbt). Devuelve velocidad
# angular en rad/s en el frame local del cuerpo. Para yaw integramos eje Z.
GYRO_DEVICE_NAME = "gyro"

# --- Sensores de distancia laterales (lado derecho del BMW) ---
# Tres DistanceSensor genéricos de un rayo, montados en sensorsSlotCenter del
# BMW con offsets (x, -0.95, 0) y rotación -π/2 alrededor de Z para apuntar el
# rayo hacia la derecha del vehículo (-Y en el frame del cuerpo).
#   - ds_right_front: x=+1.8 m (esquina delantera derecha)
#   - ds_right_mid:   x= 0.0 m (centro del flanco derecho)
#   - ds_right_rear:  x=-1.8 m (esquina trasera derecha)
# LookupTable [0 0 0, 5 5 0] -> rango 0..5 m, salida = distancia en metros
# (sin ruido para v1; se puede añadir ruido gaussiano más tarde si el reporte
# lo pide). Apertura 0.05 rad ≈ 3°, rayo único.
DS_RIGHT_FRONT_NAME = "ds_right_front"
DS_RIGHT_MID_NAME   = "ds_right_mid"
DS_RIGHT_REAR_NAME  = "ds_right_rear"
DS_MAX_RANGE_M      = 5.0   # según el lookupTable; "no obstáculo" ≈ este valor

# =============================================================================
# CONSTANTES DE LA MÁQUINA DE ESTADOS DE EVASIÓN (Stage D)
# =============================================================================

# --- LANE -> EVADE_START ---
# Disparamos evasión cuando (a) el LiDAR ve algo a menos de LIDAR_BUS_TRIGGER_M
# en su sector central, Y (b) el Recognition ha identificado al menos uno de
# los 4 autobuses por su color. La segunda condición protege contra falsos
# positivos del LiDAR (semáforos, edificios cerca de intersecciones).
LIDAR_BUS_TRIGGER_M     = 8.0   # m - distancia frontal para disparar evasión

# --- EVADE_START: swerve inicial a la izquierda ---
# Saca al BMW del carril para que el flanco derecho quede frente al autobús.
# iter 6: 0.35 -> 0.5 y 25 -> 60 frames. En iters 1-5 el swerve era demasiado
# tímido: EVADE_WALL entraba con dsF=5.0 (el sensor nunca adquiría el flanco)
# y salía en ~6 frames por la condición de rear trivialmente libre. Con giro
# al límite mecánico y más tiempo, el BMW rota lo suficiente para que el bus
# quede a su derecha. El timeout queda como fallback, ya no como gate primario.
EVADE_INIT_TURN_RAD     = 0.5   # rad - magnitud del swerve inicial (negativo en code)
EVADE_START_MAX_FRAMES  = 60    # fallback timeout (~1.9 s a 32 ms timestep)

# --- EVADE_START -> EVADE_WALL: gate por posición del bus en la imagen ---
# iter 6: gate primario nuevo. Cuando la x del bus en el frame de cámara cruza
# el 65% del ancho, el bus pasó de centro-frame a derecha-frame => el BMW ya
# rotó lo suficiente a la izquierda para que los sensores laterales derechos
# lo vean. Seguimos al MISMO bus entre frames vía el id del Recognition object
# (capturado en LANE -> EVADE_START); si se ocluye o sale del frame, caen los
# fallbacks (ds_right_front o timeout).
EVADE_GATE_IMG_FRAC     = 0.65  # fracción del ancho de imagen

# --- EVADE_WALL: seguimiento de pared derecha (P-only) ---
RIGHT_SETPOINT_M        = 1.2   # m - distancia objetivo del flanco al autobús
SIDE_KP                 = 0.6   # rad/m - ganancia P del seguimiento de pared

# --- EVADE_WALL -> RESTORE_HEADING: salida debounceada ---
# El sensor trasero (ds_right_rear) supera RIGHT_FREE_M cuando ya rebasamos el
# autobús. Pedimos RIGHT_FREE_HOLD_FRAMES frames consecutivos para no salir
# por un spike (e.g. parpadeo entre placas del Bus.proto).
RIGHT_FREE_M            = 4.5   # m - umbral para llamar "rear libre"
RIGHT_FREE_HOLD_FRAMES  = 6     # frames consecutivos requeridos

# --- RESTORE_HEADING: restaurar el rumbo previo a la evasión ---
HEADING_KP              = 1.5   # rad/rad - ganancia P sobre error de yaw
HEADING_TOLERANCE_RAD   = 0.05  # rad - tolerancia para llamarlo restaurado (~3°)

# --- EVADE_START: ventana inicial para detectar que ya estamos al lado del bus ---
# El front-right baja de DS_MAX_RANGE_M (~5) a ~SETPOINT cuando el flanco del
# autobús entra en el rayo. Usamos un umbral más generoso para no perder la
# transición si el sensor parpadea durante el swerve.
EVADE_WALL_ENTRY_M      = RIGHT_SETPOINT_M + 0.8  # ≈ 2.0 m

# =============================================================================
# MÁQUINA DE ESTADOS
# =============================================================================
# Stage A: sólo LANE existe; los otros se introducen en Stage D. Los nombres
# se reservan ya para que cualquier referencia en logs sea estable entre stages.
STATE_LANE              = "LANE"
STATE_EVADE_START       = "EVADE_START"
STATE_EVADE_WALL        = "EVADE_WALL"
STATE_RESTORE_HEADING   = "RESTORE_HEADING"


# =============================================================================
# CONTROLADOR PID (copiado verbatim de Actividad 2.1)
# =============================================================================
# Forma paralela estándar: output = Kp*e + Ki*∫e dt + Kd*de/dt
# Características: anti-windup en el término integral, salida acotada,
# reset() limpio para reusar al transitar de evasión a carril.
# =============================================================================
class PIDController:
    def __init__(self, kp, ki, kd, output_limit=MAX_STEERING_ANGLE,
                 integral_limit=100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit  # anti-windup

        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def reset(self):
        """Reinicia el estado del controlador. Lo llamamos al volver a LANE
        después de evasión para no arrastrar el error acumulado del último
        frame de carril válido (que era ya viejo)."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def compute(self, error, current_time):
        # Primer ciclo - sin dt válido todavía: sólo término P
        if self.previous_time is None:
            self.previous_time = current_time
            self.previous_error = error
            return self.kp * error

        dt = current_time - self.previous_time
        if dt <= 0:
            return self.kp * error  # protección contra dt=0

        # Integral con anti-windup
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
# FUNCIONES DE PROCESAMIENTO DE IMAGEN (copiadas verbatim de Actividad 2.1)
# =============================================================================

def get_image(camera):
    """BGRA (Webots) -> BGR (OpenCV)."""
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image[:, :, :3]


def to_grayscale(image_bgr):
    """Máscara HSV-amarillo antes de pasar a gris. Filtra edificios, sombras y
    banquetas que competirían por votos en Hough. Cumple el paso 3 de la
    actividad 2.1 (la imagen resultante sigue siendo escala de grises)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    yellow_lower = np.array([15, 80, 80])
    yellow_upper = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=yellow_mask)
    return cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)


def detect_edges(image_gray):
    """Blur Gaussiano + Canny."""
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    return cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)


def apply_roi(edges):
    """Trapecio sobre la mitad inferior, definido en porcentajes para ser
    independiente de la resolución exacta de la cámara."""
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
    """HoughLinesP probabilística - devuelve segmentos (x1,y1,x2,y2)."""
    return cv2.HoughLinesP(
        edges_roi,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )


def compute_error(lines, image_width):
    """Error PID = midpoint_x - image_width/2. Devuelve el error de menor
    magnitud entre todas las líneas válidas (no horizontales)."""
    if lines is None:
        return None, 0

    setpoint = image_width / 2.0
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
# HELPERS DE SENSORES (Stage B)
# =============================================================================

def init_lidar(robot, timestep):
    """Inicializa el LiDAR Sick LMS 291 y precalcula el slice del sector
    central. Devuelve (lidar_device, central_slice_object). El slice es un
    objeto slice nativo de Python para indexar getRangeImage() sin recalcular
    índices cada frame."""
    lidar = robot.getDevice(LIDAR_DEVICE_NAME)
    lidar.enable(timestep)
    lidar.enablePointCloud()
    horiz_res = lidar.getHorizontalResolution()
    central_n = max(1, int(horiz_res * LIDAR_FOV_CENTRAL_DEG / LIDAR_FOV_TOTAL_DEG))
    mid = horiz_res // 2
    half = central_n // 2
    central_slice = slice(mid - half, mid + half)
    print(f"[INIT] LiDAR: horiz_res={horiz_res}, sector central={central_n} "
          f"rayos (slice [{mid-half}:{mid+half}])")
    return lidar, central_slice


def read_forward_distance(lidar, central_slice):
    """Devuelve la distancia mínima finita en el sector central, acotada a
    LIDAR_MAX_DETECT_M (más allá es ruido). Si no hay rayos finitos devuelve
    inf (no hay obstáculo)."""
    ranges = lidar.getRangeImage()[central_slice]
    finite = [r for r in ranges if math.isfinite(r) and r <= LIDAR_MAX_DETECT_M]
    return min(finite) if finite else float("inf")


def init_gyro(robot, timestep):
    """Inicializa el Gyro. Devuelve el device."""
    gyro = robot.getDevice(GYRO_DEVICE_NAME)
    gyro.enable(timestep)
    print(f"[INIT] Gyro habilitado")
    return gyro


def wall_follow_right_steering(ds_front_m):
    """Controlador P para seguimiento de pared derecha. Mantiene el flanco
    derecho del BMW a RIGHT_SETPOINT_M del autobús.

    Convención de signos:
      - err > 0  -> demasiado cerca -> girar a la IZQUIERDA (steering negativo)
      - err < 0  -> demasiado lejos -> girar a la DERECHA (steering positivo)

    Devuelve el comando de steering (rad), acotado al límite mecánico.
    """
    err = RIGHT_SETPOINT_M - ds_front_m
    cmd = -SIDE_KP * err
    return max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, cmd))


def restore_heading_steering(saved_yaw_z, yaw_z):
    """Controlador P sobre el error de yaw para volver al rumbo previo a la
    evasión.

    Convención de Webots-Driver: setSteeringAngle(positive) = giro a la
    derecha = yaw DECRECE; setSteeringAngle(negative) = giro a la izquierda =
    yaw CRECE.

    Lógica corregida (bug del iter 4):
      - Si yaw_z > saved (sobrepasamos giroizando), queremos disminuir yaw,
        es decir, girar a la derecha -> steering POSITIVO.
      - Por tanto cmd = HEADING_KP * (yaw - saved), NO (saved - yaw).

    También: wrap del error a [-π, π] con atan2(sin, cos). Sin esto, si el
    BMW da una vuelta completa durante evasión, el error se hace 2π en vez
    de 0 y el controlador sigue girando indefinidamente. Bug observado en
    iter 4: yaw crecía monotónicamente +1.66 -> +10.65.
    """
    raw_err = yaw_z - saved_yaw_z
    err = math.atan2(math.sin(raw_err), math.cos(raw_err))
    cmd = HEADING_KP * err
    return max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, cmd))


def compute_transition(state, lidar_front_m, buses_seen, ds_front_m,
                       ds_rear_m, yaw_z, saved_yaw_z,
                       frames_in_state, rear_free_streak,
                       evade_bus_img_x, img_width):
    """Devuelve (new_state, new_rear_free_streak). Función pura: las
    transiciones dependen solo de los argumentos.

    evade_bus_img_x: coordenada x (px) del bus que estamos evadiendo en el
    frame de cámara, o None si Recognition lo perdió este frame (ocluido o
    fuera de frame) — en ese caso aplican los fallbacks de EVADE_START.

    El streak del rear se mantiene a través de las transiciones porque sólo
    importa dentro de EVADE_WALL (se reinicia al entrar).
    """
    if state == STATE_LANE:
        if lidar_front_m < LIDAR_BUS_TRIGGER_M and len(buses_seen) > 0:
            return STATE_EVADE_START, 0
        return STATE_LANE, rear_free_streak

    if state == STATE_EVADE_START:
        # Gate primario (iter 6): el bus cruzó al 65% derecho del frame =>
        # el BMW ya rotó lo suficiente para que el flanco derecho lo adquiera.
        # Fallbacks: el sensor derecho-frontal ya lo ve, o timeout.
        img_gate = (evade_bus_img_x is not None
                    and evade_bus_img_x > img_width * EVADE_GATE_IMG_FRAC)
        if (img_gate or ds_front_m < EVADE_WALL_ENTRY_M
                or frames_in_state >= EVADE_START_MAX_FRAMES):
            return STATE_EVADE_WALL, 0
        return STATE_EVADE_START, rear_free_streak

    if state == STATE_EVADE_WALL:
        # Streak debounceado sobre ds_rear.
        if ds_rear_m >= RIGHT_FREE_M:
            new_streak = rear_free_streak + 1
        else:
            new_streak = 0
        if new_streak >= RIGHT_FREE_HOLD_FRAMES:
            return STATE_RESTORE_HEADING, 0
        return STATE_EVADE_WALL, new_streak

    if state == STATE_RESTORE_HEADING:
        # Error wrapeado a [-π, π] (igual que en restore_heading_steering)
        raw_err = yaw_z - saved_yaw_z
        wrapped_err = math.atan2(math.sin(raw_err), math.cos(raw_err))
        if abs(wrapped_err) < HEADING_TOLERANCE_RAD:
            return STATE_LANE, 0
        return STATE_RESTORE_HEADING, rear_free_streak

    # Fallback - estado desconocido vuelve a LANE
    return STATE_LANE, 0


def init_right_distance_sensors(robot, timestep):
    """Inicializa los 3 DistanceSensor del flanco derecho. Devuelve un dict
    con claves 'front' / 'mid' / 'rear' apuntando al device correspondiente."""
    ds = {
        "front": robot.getDevice(DS_RIGHT_FRONT_NAME),
        "mid":   robot.getDevice(DS_RIGHT_MID_NAME),
        "rear":  robot.getDevice(DS_RIGHT_REAR_NAME),
    }
    for name, dev in ds.items():
        dev.enable(timestep)
    print(f"[INIT] DistanceSensors derechos habilitados: "
          f"{DS_RIGHT_FRONT_NAME}, {DS_RIGHT_MID_NAME}, {DS_RIGHT_REAR_NAME}")
    return ds


def identify_bus_by_color(recognition_color):
    """Empareja un color RGB devuelto por Recognition con uno del catálogo
    BUS_COLORS_RGB. Devuelve el nombre del bus o None si no hay match.

    Webots redondea/aproxima los floats de recognitionColors al pasarlos por
    el template, así que comparamos con tolerancia euclidiana."""
    r, g, b = recognition_color
    best_name = None
    best_dist = BUS_COLOR_MATCH_TOL
    for name, (cr, cg, cb) in BUS_COLORS_RGB.items():
        d = math.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def _safe_index_pair(maybe_ptr):
    """Convierte un resultado de getPositionOnImage/getSizeOnImage (puede ser
    lista Python o ctypes pointer) en una tupla (x, y). Defensivo porque la
    API de Webots para CameraRecognitionObject devuelve ctypes pointers
    para muchos atributos."""
    try:
        return (maybe_ptr[0], maybe_ptr[1])
    except (TypeError, IndexError):
        return (0, 0)


def summarize_recognition_objects(rec_objs):
    """Convierte la lista de CameraRecognitionObject en una lista de dicts con
    los campos relevantes para esta actividad: bus_name, position (m, frame
    de cámara), size_on_image (px). Sólo conserva objetos que se identifican
    como uno de los 4 autobuses.

    Importante: obj.getColors() devuelve un ctypes LP_c_double (pointer), no
    una lista de Python. Tenemos que leer obj.getNumberOfColors() y luego
    indexar el pointer (ncolors * 3 floats consecutivos, RGB intercalado).
    Lo mismo pasa con getPosition() (pointer a 3 doubles) — getPosition() ya
    es subscriptable, así que [0], [1], [2] funciona."""
    summary = []
    for obj in rec_objs:
        try:
            ncolors = obj.getNumberOfColors()
        except Exception:
            ncolors = 0
        if ncolors < 1:
            continue
        colors_ptr = obj.getColors()
        try:
            primary = (colors_ptr[0], colors_ptr[1], colors_ptr[2])
        except (TypeError, IndexError):
            continue
        bus_name = identify_bus_by_color(primary)
        if bus_name is None:
            continue
        pos_ptr = obj.getPosition()
        try:
            pos = (pos_ptr[0], pos_ptr[1], pos_ptr[2])
        except (TypeError, IndexError):
            pos = (float("nan"), float("nan"), float("nan"))
        try:
            obj_id = obj.getId()
        except Exception:
            obj_id = -1
        summary.append({
            "id": obj_id,  # id del Recognition: estable entre frames, permite
                           # seguir al MISMO bus durante toda la evasión
            "bus": bus_name,
            "color_rgb": primary,
            "position_m": pos,
            "pos_on_image": _safe_index_pair(obj.getPositionOnImage()),
            "size_on_image": _safe_index_pair(obj.getSizeOnImage()),
        })
    return summary


# =============================================================================
# VISUALIZACIÓN (para depuración y video demo)
# =============================================================================

def draw_lines_on_image(image_bgr, lines, color=(0, 255, 0), thickness=2):
    output = image_bgr.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), color, thickness)
    return output


def draw_setpoint_line(image_bgr, color=(0, 0, 255)):
    output = image_bgr.copy()
    x = output.shape[1] // 2
    cv2.line(output, (x, 0), (x, output.shape[0]), color, 1)
    return output


def display_image_on_webots(display, image_bgr):
    """Pinta una imagen BGR en el Display interno de Webots.
    OBLIGATORIO llamar imageDelete después de imagePaste: ver docstring."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)
    display.imageDelete(image_ref)  # evita la fuga de slots de 3.1


def draw_state_overlay(display, state, lidar_front_m, ds_front_m,
                       ds_rear_m, yaw_z, saved_yaw_z, frames_in_state):
    """Dibuja sobre el Display el estado actual y las métricas clave para
    poder seguir la máquina de estados en vivo durante la sim.

    Usa los primitivos de dibujo de Webots (drawText), no OpenCV, para no
    sobrescribir la imagen de cámara ya pegada con imagePaste."""
    # Color del texto según estado: ayuda a leerlo en el demo.
    color_by_state = {
        STATE_LANE:            0xFFFFFF,  # blanco
        STATE_EVADE_START:     0xFFFF00,  # amarillo
        STATE_EVADE_WALL:      0xFF8000,  # naranja
        STATE_RESTORE_HEADING: 0x00FFFF,  # cian
    }
    display.setColor(color_by_state.get(state, 0xFF0000))
    display.drawText(f"STATE: {state}", 4, 4)
    display.setColor(0xFFFFFF)
    lidar_str = f"{lidar_front_m:5.2f}" if math.isfinite(lidar_front_m) else "  inf"
    display.drawText(f"LiDAR_F:{lidar_str}m  dsF:{ds_front_m:.2f}  dsR:{ds_rear_m:.2f}",
                     4, 18)
    display.drawText(f"yaw:{yaw_z:+.2f}  saved:{saved_yaw_z:+.2f}  f:{frames_in_state}",
                     4, 32)


# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

def main():
    # --- Inicialización de Webots ---
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    # Teleport del BMW al spawn validado de Actividad 2.1 ANTES de inicializar
    # nada más. En la 4.2 el .wbt deja al BMW en (-1.22, 45.06) — justo al lado
    # de una intersección donde la línea amarilla desaparece y el lane follower
    # falla. El spawn original de 2.1 (-55.67, 45.88, 0.315) está en mitad de
    # un tramo recto largo con línea amarilla continua y es donde la
    # sintonización del PID está validada. supervisor TRUE en el BMW (.wbt)
    # habilita esto.
    # CRÍTICO: setear translation Y rotation A LA VEZ y llamar resetPhysics()
    # para limpiar velocidad lineal/angular acumulada — sin resetPhysics el
    # BMW arrastra la velocidad del momento de spawn y se estrella contra
    # cualquier muro cercano.
    SPAWN_X, SPAWN_Y, SPAWN_Z = -55.67054714756429, 45.88004292921422, 0.31513215760005664
    SPAWN_AXIS = [0.012795947566112598, -4.99889827643198e-07, 0.9999181285113474]
    SPAWN_ANGLE = 3.141589273847238
    try:
        self_node = robot.getSelf()
        if self_node is not None:
            self_node.getField("translation").setSFVec3f([SPAWN_X, SPAWN_Y, SPAWN_Z])
            self_node.getField("rotation").setSFRotation(SPAWN_AXIS + [SPAWN_ANGLE])
            self_node.resetPhysics()
            print(f"[INIT] Teleport BMW -> ({SPAWN_X:.2f}, {SPAWN_Y:.2f}, {SPAWN_Z:.2f}) "
                  f"yaw=π   (spawn validado de Actividad 2.1, physics reset)")
        else:
            print("[INIT] WARN: robot.getSelf() devolvió None - supervisor no habilitado?")
    except Exception as e:
        print(f"[INIT] WARN: no se pudo teleport BMW: {e}")

    # Cámara + Recognition habilitado por el .wbt (Recognition node)
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    camera.recognitionEnable(timestep)
    img_width = camera.getWidth()
    img_height = camera.getHeight()
    print(f"[INIT] Cámara: {img_width}x{img_height}, setpoint={img_width/2}, "
          f"recognition habilitado")

    # Display para visualización en simulador
    display_img = Display("display_image")

    # Teclado: lo dejamos disponible para pausa/intervención manual durante
    # debug, pero NO leemos teclas en el loop (controlador 100% autónomo).
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # LiDAR + sector central precalculado
    lidar, lidar_central = init_lidar(robot, timestep)

    # Gyro (para integrar el yaw_z)
    gyro = init_gyro(robot, timestep)
    dt_s = timestep / 1000.0  # paso de integración en segundos

    # Sensores de distancia laterales (lado derecho del BMW)
    right_ds = init_right_distance_sensors(robot, timestep)

    # Velocidad inicial
    driver.setCruisingSpeed(TARGET_SPEED)

    # PID de carril
    pid = PIDController(kp=PID_KP, ki=PID_KI, kd=PID_KD)
    print(f"[INIT] PID gains: Kp={PID_KP}, Ki={PID_KI}, Kd={PID_KD}")

    # Estado inicial
    state = STATE_LANE
    print(f"[INIT] Estado inicial: {state}")

    # Estado del controlador
    frame_count = 0
    no_line_count = 0
    last_valid_angle = 0.0      # último ángulo aplicado con detección válida
    frames_since_detection = 0  # frames consecutivos sin línea detectada
    prev_angle = 0.0            # para el slew rate limit
    yaw_z = 0.0                 # yaw acumulado (rad) por integración del gyro

    # Estado de la máquina de estados de evasión (Stage D)
    saved_yaw_z = 0.0           # snapshot del yaw al entrar en EVADE_START
    frames_in_state = 0         # cuántos frames lleva el estado actual
    rear_free_streak = 0        # debounce del ds_right_rear (sólo en EVADE_WALL)
    evading_bus_id = None       # id Recognition del bus que estamos evadiendo
                                # (capturado en LANE -> EVADE_START, iter 6)

    # --- Loop principal ---
    # Usamos driver.step() (no robot.step()) para vaciar el buffer interno del
    # Driver en cada paso. Ver docstring del archivo.
    while driver.step() != -1:
        # 1. Captura de imagen
        image_bgr = get_image(camera)

        # 2. Pipeline de detección de carriles
        gray = to_grayscale(image_bgr)
        edges = detect_edges(gray)
        edges_roi = apply_roi(edges)
        lines = detect_lines(edges_roi)

        # 3. Cálculo del error
        error, num_valid = compute_error(lines, img_width)

        # 3b. Lectura de sensores adicionales (Stage B + C).
        # Integramos yaw aquí cada frame; nunca se reinicia (se snapshotará al
        # entrar a EVADE_START en Stage D). Drift admisible para la ventana de
        # evasión (~pocos segundos).
        yaw_z += gyro.getValues()[2] * dt_s
        lidar_front_m = read_forward_distance(lidar, lidar_central)
        rec_objs = camera.getRecognitionObjects()
        # DEBUG: imprime el conteo crudo (sin filtrar por color) cada 50 frames
        # para diagnosticar si Recognition no encuentra nada vs filtro de color
        # demasiado estricto.
        if frame_count % 50 == 0:
            print(f"[DEBUG REC] total_raw_objs={len(rec_objs)} "
                  f"has_recog={camera.hasRecognition()}")
        if frame_count % 50 == 0 and len(rec_objs) > 0:
            for i, _o in enumerate(rec_objs[:5]):
                try:
                    nc = _o.getNumberOfColors()
                    cptr = _o.getColors()
                    c0 = (cptr[0], cptr[1], cptr[2]) if nc > 0 else None
                    pos = _o.getPosition()
                    print(f"[DEBUG REC] obj{i}: ncolors={nc} color0={c0} "
                          f"pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) "
                          f"model={_o.getModel()}")
                except Exception as e:
                    print(f"[DEBUG REC] obj{i}: error reading attrs: {e}")
        buses_seen = summarize_recognition_objects(rec_objs)
        ds_front_m = right_ds["front"].getValue()
        ds_mid_m   = right_ds["mid"].getValue()
        ds_rear_m  = right_ds["rear"].getValue()

        # 3c. Posición en imagen del bus que estamos evadiendo (iter 6).
        # Buscamos por id de Recognition para seguir al MISMO bus aunque la
        # cámara vea otros. None si está ocluido / fuera de frame este frame.
        evade_bus_img_x = None
        if evading_bus_id is not None:
            for b in buses_seen:
                if b["id"] == evading_bus_id:
                    evade_bus_img_x = b["pos_on_image"][0]
                    break

        # 4. Transición de estados (Stage D). Una sola fuente de verdad: la
        # función pura compute_transition() decide en base a sensores.
        new_state, rear_free_streak = compute_transition(
            state, lidar_front_m, buses_seen, ds_front_m, ds_rear_m,
            yaw_z, saved_yaw_z, frames_in_state, rear_free_streak,
            evade_bus_img_x, img_width,
        )
        if new_state != state:
            # Hooks de entrada al nuevo estado
            if new_state == STATE_EVADE_START:
                # Snapshot del rumbo ANTES del swerve: lo usaremos en RESTORE.
                saved_yaw_z = yaw_z
                # Capturamos el id del bus más cercano (frame de cámara) para
                # seguirlo durante EVADE_START aunque aparezcan otros buses.
                def _bus_dist(b):
                    x, y, z = b["position_m"]
                    if all(math.isfinite(v) for v in (x, y, z)):
                        return math.sqrt(x * x + y * y + z * z)
                    return float("inf")
                nearest = min(buses_seen, key=_bus_dist)
                evading_bus_id = nearest["id"]
                print(f"[STATE] {state} -> {new_state}  yaw_z snapshot={yaw_z:+.3f} rad "
                      f"(lidar={lidar_front_m:.2f}m, buses={[b['bus'] for b in buses_seen]}, "
                      f"evading={nearest['bus']} id={evading_bus_id})")
            elif new_state == STATE_LANE:
                # Volvemos a carril: PID limpio para no arrastrar error viejo.
                pid.reset()
                last_valid_angle = 0.0
                frames_since_detection = 0
                evading_bus_id = None  # evasión terminada: soltamos el bus
                print(f"[STATE] {state} -> {new_state}  yaw_z={yaw_z:+.3f} (target={saved_yaw_z:+.3f})")
            else:
                img_x_str = (f"{evade_bus_img_x:.0f}px"
                             if evade_bus_img_x is not None else "lost")
                print(f"[STATE] {state} -> {new_state}  "
                      f"(lidar={lidar_front_m:.2f}m, ds_F={ds_front_m:.2f} "
                      f"ds_R={ds_rear_m:.2f}, yaw_z={yaw_z:+.3f}, "
                      f"bus_img_x={img_x_str}/{img_width}px)")
            state = new_state
            frames_in_state = 0
        else:
            frames_in_state += 1

        # 5. Actuación por estado
        current_time = robot.getTime()
        if state == STATE_LANE:
            # 2.1 PID lane-follower verbatim
            if error is None:
                frames_since_detection += 1
                no_line_count += 1
                if frames_since_detection <= NO_DETECTION_GRACE_FRAMES:
                    steering_angle = last_valid_angle
                else:
                    steering_angle = DEFAULT_STEERING_ANGLE
                    pid.reset()
            else:
                steering_angle = pid.compute(error, current_time)
                last_valid_angle = steering_angle
                frames_since_detection = 0
            target_speed = TARGET_SPEED

        elif state == STATE_EVADE_START:
            # Swerve inicial a la izquierda (negativo en convención del Driver).
            steering_angle = -EVADE_INIT_TURN_RAD
            target_speed = EVADE_SPEED_KMH

        elif state == STATE_EVADE_WALL:
            # Seguimiento de pared derecha con P-only sobre ds_front_m.
            steering_angle = wall_follow_right_steering(ds_front_m)
            target_speed = EVADE_SPEED_KMH

        elif state == STATE_RESTORE_HEADING:
            # Volver al yaw_z guardado antes de empezar la evasión.
            steering_angle = restore_heading_steering(saved_yaw_z, yaw_z)
            target_speed = EVADE_SPEED_KMH

        else:
            # Estado desconocido: comportamiento seguro = recto y lento.
            steering_angle = DEFAULT_STEERING_ANGLE
            target_speed = EVADE_SPEED_KMH

        # 6. Slew rate limit (anti-jerk en steering)
        delta = steering_angle - prev_angle
        if delta > MAX_ANGLE_CHANGE_PER_FRAME:
            steering_angle = prev_angle + MAX_ANGLE_CHANGE_PER_FRAME
        elif delta < -MAX_ANGLE_CHANGE_PER_FRAME:
            steering_angle = prev_angle - MAX_ANGLE_CHANGE_PER_FRAME
        prev_angle = steering_angle

        # 7. Aplicar al vehículo
        driver.setSteeringAngle(steering_angle)
        driver.setCruisingSpeed(target_speed)

        # 8. Visualización en el display
        debug_image = draw_lines_on_image(image_bgr, lines)
        debug_image = draw_setpoint_line(debug_image)
        display_image_on_webots(display_img, debug_image)
        # Overlay de estado + métricas clave ENCIMA de la imagen pegada.
        draw_state_overlay(display_img, state, lidar_front_m, ds_front_m,
                           ds_rear_m, yaw_z, saved_yaw_z, frames_in_state)

        # 9. Logging periódico
        frame_count += 1
        # Log [F#] cada 50 frames (~1.6 s a 32 ms timestep) - estado general
        if frame_count % 50 == 0:
            err_str = f"{error:+.2f}" if error is not None else "  N/A"
            n_lines = len(lines) if lines is not None else 0
            lidar_str = (f"{lidar_front_m:5.2f} m" if math.isfinite(lidar_front_m)
                         else "   inf")
            print(f"[F{frame_count}] state={state} lines={n_lines} "
                  f"valid={num_valid} err={err_str} "
                  f"angle={steering_angle:+.4f} no_line_total={no_line_count} "
                  f"lidar={lidar_str} yaw_z={yaw_z:+.3f} rad "
                  f"buses_seen={len(buses_seen)} "
                  f"ds_R=(F:{ds_front_m:.2f} M:{ds_mid_m:.2f} R:{ds_rear_m:.2f}) m")
            for b in buses_seen:
                pos = b["position_m"]
                print(f"         bus={b['bus']:<22s} "
                      f"pos_cam=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) m "
                      f"size_img={b['size_on_image']}")

        # Log [METRICS] cada 5 frames cuando NO estamos en LANE: durante la
        # evasión queremos resolución alta para diagnosticar la sintonización
        # del controlador P de pared y de RESTORE_HEADING.
        if state != STATE_LANE and frame_count % 5 == 0:
            if state == STATE_EVADE_WALL:
                err_wall = RIGHT_SETPOINT_M - ds_front_m
                print(f"[METRICS] {state} f={frames_in_state} "
                      f"dsF={ds_front_m:5.2f} dsR={ds_rear_m:5.2f} "
                      f"err_wall={err_wall:+5.2f} steer={steering_angle:+.3f} "
                      f"rear_streak={rear_free_streak}")
            elif state == STATE_RESTORE_HEADING:
                err_yaw = saved_yaw_z - yaw_z
                print(f"[METRICS] {state} f={frames_in_state} "
                      f"yaw={yaw_z:+.3f} target={saved_yaw_z:+.3f} "
                      f"err_yaw={err_yaw:+.3f} steer={steering_angle:+.3f}")
            else:  # EVADE_START
                img_x_str = (f"{evade_bus_img_x:.0f}" if evade_bus_img_x is not None
                             else "lost")
                print(f"[METRICS] {state} f={frames_in_state} "
                      f"dsF={ds_front_m:5.2f} lidar={lidar_front_m:.2f} "
                      f"steer={steering_angle:+.3f} "
                      f"bus_img_x={img_x_str}/{img_width} "
                      f"gate_at={img_width * EVADE_GATE_IMG_FRAC:.0f}")


if __name__ == "__main__":
    main()
