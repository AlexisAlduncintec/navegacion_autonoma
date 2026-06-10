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
    LANE  -- LiDAR<TRIGGER + recognition -->  EVADE_START  (guarda yaw_z)
    EVADE_START  -- ds_right_front ve el flanco -->  EVADE_WALL
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

Uso (Mac / VS Code terminal):
    export WEBOTS_HOME=/Applications/Webots.app
    export PYTHONPATH=/Applications/Webots.app/Contents/lib/controller/python
    export WEBOTS_PYTHON_EXECUTABLE=$(which python)
    conda activate webots
    cd /Users/alexisalduncin/ITESM/navegacion_autonoma/SDC_webots
    python autonomous_evasion_controller.py
"""

from controller import Display, Keyboard, Robot, Camera
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
EVADE_SPEED_KMH = 20.0       # km/h - estados EVADE_* y RESTORE_HEADING
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


# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

def main():
    # --- Inicialización de Webots ---
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    # Cámara
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    img_width = camera.getWidth()
    img_height = camera.getHeight()
    print(f"[INIT] Cámara: {img_width}x{img_height}, setpoint={img_width/2}")

    # Display para visualización en simulador
    display_img = Display("display_image")

    # Teclado: lo dejamos disponible para pausa/intervención manual durante
    # debug, pero NO leemos teclas en el loop (controlador 100% autónomo).
    keyboard = Keyboard()
    keyboard.enable(timestep)

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

        # 4. Control de carril (estado LANE)
        # En Stage A sólo existe LANE. La rama if/elif está armada para los
        # estados de Stage D para minimizar diff entre commits.
        current_time = robot.getTime()
        if state == STATE_LANE:
            if error is None:
                frames_since_detection += 1
                no_line_count += 1
                if frames_since_detection <= NO_DETECTION_GRACE_FRAMES:
                    # Pérdida breve: mantener último ángulo válido
                    steering_angle = last_valid_angle
                else:
                    # Pérdida prolongada (intersección): conducir recto
                    steering_angle = DEFAULT_STEERING_ANGLE
                    pid.reset()
            else:
                steering_angle = pid.compute(error, current_time)
                last_valid_angle = steering_angle
                frames_since_detection = 0
            target_speed = TARGET_SPEED
        else:
            # Stage A: no debería llegar aquí; los estados EVADE_*/RESTORE
            # se implementan en Stage D.
            steering_angle = DEFAULT_STEERING_ANGLE
            target_speed = EVADE_SPEED_KMH

        # 5. Slew rate limit (anti-jerk en steering)
        delta = steering_angle - prev_angle
        if delta > MAX_ANGLE_CHANGE_PER_FRAME:
            steering_angle = prev_angle + MAX_ANGLE_CHANGE_PER_FRAME
        elif delta < -MAX_ANGLE_CHANGE_PER_FRAME:
            steering_angle = prev_angle - MAX_ANGLE_CHANGE_PER_FRAME
        prev_angle = steering_angle

        # 6. Aplicar al vehículo
        driver.setSteeringAngle(steering_angle)
        driver.setCruisingSpeed(target_speed)

        # 7. Visualización en el display
        debug_image = draw_lines_on_image(image_bgr, lines)
        debug_image = draw_setpoint_line(debug_image)
        display_image_on_webots(display_img, debug_image)

        # 8. Log cada 50 frames (~0.5 s a 10 ms timestep)
        frame_count += 1
        if frame_count % 50 == 0:
            err_str = f"{error:+.2f}" if error is not None else "  N/A"
            n_lines = len(lines) if lines is not None else 0
            print(f"[F{frame_count}] state={state} lines={n_lines} "
                  f"valid={num_valid} err={err_str} "
                  f"angle={steering_angle:+.4f} no_line_total={no_line_count}")


if __name__ == "__main__":
    main()
