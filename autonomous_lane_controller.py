"""
================================================================================
Actividad 2.1 - Detección de carriles con OpenCV y control PID
MR4010 - Navegación Autónoma | Tecnológico de Monterrey | MNA
================================================================================

Pipeline:
    Cámara (BGRA 128x64) -> Gris -> Canny -> ROI (fillPoly) -> HoughLinesP
    -> Cálculo del error (distancia mínima entre midpoint de línea y centro)
    -> PID -> ángulo de dirección -> Driver.setSteeringAngle()

Mundo: city_2025a.wbt (BMW X5, líneas amarillas centrales, intersecciones)
Validado a velocidad mínima de 50 km/h.

Uso (Mac / VS Code terminal):
    export WEBOTS_HOME=/Applications/Webots.app
    export PYTHONPATH=/Applications/Webots.app/Contents/lib/controller/python
    export WEBOTS_PYTHON_EXECUTABLE=$(which python)
    conda activate <env>
    python autonomous_lane_controller.py
"""

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2

# =============================================================================
# CONSTANTES DE CONFIGURACIÓN
# =============================================================================

# --- Velocidad ---
# La actividad exige una velocidad mínima de 50 km/h. Mantenemos constante
# para que el comportamiento del controlador PID sea independiente de la
# velocidad (la respuesta del PID ya es función del error, no de la velocidad).
TARGET_SPEED = 50.0          # km/h - cumple el mínimo de la actividad
MAX_STEERING_ANGLE = 0.5     # radianes - límite mecánico del BMW X5

# --- Parámetros de Canny ---
# Valores conservadores. Threshold1 (más bajo) para detectar bordes débiles
# conectados a fuertes; Threshold2 (más alto) para detectar bordes fuertes.
# Si el sistema no ve la línea amarilla en sombras, bajar threshold1.
CANNY_LOW = 50
CANNY_HIGH = 150

# --- Parámetros de HoughLinesP ---
# rho: resolución de distancia en pixeles (1 px es estándar)
# theta: resolución angular en radianes (pi/180 = 1 grado, estándar)
# threshold: votos mínimos en el accumulator para considerar una línea
# minLineLength: descartar líneas más cortas (ruido)
# maxLineGap: unir segmentos separados por hasta este número de pixeles
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 20          # 20 votos. Reducido para detectar arcos cortos tras máscara amarilla.
HOUGH_MIN_LINE_LEN = 15       # 15 px. Líneas más cortas son ruido.
HOUGH_MAX_LINE_GAP = 20       # 20 px. Permite reconstruir líneas punteadas.

# --- Filtro de líneas horizontales ---
# Las líneas casi horizontales (techo de edificios, sombras de banquetas)
# alteran el cálculo del error. Las descartamos basándonos en su pendiente.
# slope = (y2 - y1) / (x2 - x1). Si |slope| < HORIZONTAL_SLOPE_THRESHOLD,
# la línea es "demasiado horizontal" y se ignora.
HORIZONTAL_SLOPE_THRESHOLD = 0.05

# --- Ganancias del controlador PID ---
# Punto de partida tras tuning empírico para 50 km/h, cámara 128x96.
# Si el coche oscila izquierda-derecha: bajar Kp o subir Kd.
# Si el coche se desvía y no regresa: subir Kp.
# Si el coche tiene drift constante a un lado: subir Ki ligeramente.
# La actividad recomienda partir de 0 en las tres ganancias y subir Kp
# primero hasta tener seguimiento básico, luego añadir Kd para amortiguar.
PID_KP = 0.008                # Proporcional - principal corrección
PID_KI = 0.0                  # Integral - generalmente 0 para lane keeping
PID_KD = 0.004                # Derivativo - amortigua oscilaciones

# --- Slew rate limit ---
# Máximo cambio de ángulo permitido entre frames consecutivos.
# A 10ms timestep, 0.05 rad/frame = 5 rad/s, equivalente a un giro de
# volante razonablemente rápido para un humano.
MAX_ANGLE_CHANGE_PER_FRAME = 0.05

# --- Comportamiento por defecto cuando no hay líneas detectadas ---
# En las intersecciones del mundo no hay línea amarilla. Conducir recto
# (angle=0) es la conducta segura por omisión, según indica la actividad.
DEFAULT_STEERING_ANGLE = 0.0

# --- Comportamiento ante pérdida de detección ---
# En vez de aplicar ángulo=0 inmediatamente al perder la línea, mantenemos
# el último ángulo válido durante NO_DETECTION_GRACE_FRAMES frames. Esto
# evita que el coche se desvíe durante huecos cortos (sombras, oclusiones
# momentáneas) y reduce el salto del término derivativo cuando reaparece.
NO_DETECTION_GRACE_FRAMES = 20


# =============================================================================
# CONTROLADOR PID
# =============================================================================
# Implementación orientada a objetos. Forma paralela estándar:
#     output(t) = Kp*e(t) + Ki*∫e(t)dt + Kd*de(t)/dt
# Referencias:
#   - https://medium.com/@aleksej.gudkov/python-pid-controller-example-...
#   - MathWorks PID Control - Part 1
#
# Características importantes:
#   - Anti-windup: el término integral se acota para que un error sostenido
#     no acumule indefinidamente y sature la salida.
#   - Salida acotada: el ángulo final se limita a ±MAX_STEERING_ANGLE
#     (límite mecánico del BMW X5).
# =============================================================================
class PIDController:
    def __init__(self, kp, ki, kd, output_limit=MAX_STEERING_ANGLE,
                 integral_limit=100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit  # anti-windup

        # Estado interno
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def reset(self):
        """Reinicia el estado del controlador. Útil al perder la línea."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def compute(self, error, current_time):
        """
        Calcula la salida del PID.

        Args:
            error: error actual (setpoint - measurement) en pixeles.
            current_time: tiempo actual en segundos. Usar robot.getTime()
                          (tiempo de simulación) en lugar de time.time()
                          (tiempo de reloj) para que el PID se comporte
                          correctamente si Webots corre a velocidad distinta
                          del tiempo real.

        Returns:
            Salida del controlador, acotada a ±output_limit.
        """
        # Primer ciclo - sin dt válido todavía
        if self.previous_time is None:
            self.previous_time = current_time
            self.previous_error = error
            return self.kp * error  # solo término P en el primer ciclo

        dt = current_time - self.previous_time
        if dt <= 0:
            # Protección contra dt=0 (evita división por cero)
            return self.kp * error

        # Término integral con anti-windup
        self.integral += error * dt
        self.integral = max(-self.integral_limit,
                            min(self.integral_limit, self.integral))

        # Término derivativo - tasa de cambio del error
        derivative = (error - self.previous_error) / dt

        # Salida combinada
        output = (self.kp * error
                  + self.ki * self.integral
                  + self.kd * derivative)

        # Acotamiento a límites mecánicos
        output = max(-self.output_limit, min(self.output_limit, output))

        # Actualizar estado
        self.previous_error = error
        self.previous_time = current_time

        return output


# =============================================================================
# FUNCIONES DE PROCESAMIENTO DE IMAGEN
# =============================================================================

def get_image(camera):
    """
    Obtiene la imagen de la cámara de Webots.
    Webots devuelve BGRA (4 canales). Convertimos a BGR (3 canales) para
    que OpenCV se comporte de forma estándar.
    """
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image[:, :, :3]  # descarta alpha, devuelve BGR


def to_grayscale(image_bgr):
    """
    Conversión a escala de grises - paso 3 de la actividad.

    Antes de pasar a grises, aplicamos una máscara de color amarillo en
    espacio HSV. El mundo de Webots usa líneas amarillas centrales que son
    el objetivo de detección. Filtrar primero por color elimina edificios,
    sombras, banquetas y otros bordes que compiten por votos en Hough.

    El resultado sigue siendo una imagen en escala de grises (cumple el
    requisito del paso 3 de la actividad), pero solo conserva luminancia
    en pixeles que originalmente eran amarillos.
    """
    # HSV es más estable que BGR para filtrado de color porque separa
    # tono (H), saturación (S) y luminosidad (V).
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Rango amarillo: H ~ 20-35 (en escala 0-179 de OpenCV), saturación
    # alta para evitar grises desaturados, valor alto para descartar
    # sombras oscuras.
    yellow_lower = np.array([15, 80, 80])
    yellow_upper = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Aplicamos la máscara a la imagen original y luego convertimos a gris
    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=yellow_mask)
    return cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)


def detect_edges(image_gray):
    """
    Detección de bordes con Canny - paso 4 de la actividad.
    Aplicamos un blur Gaussiano antes para reducir ruido (recomendado en
    el video #20 de ProgrammingKnowledge y en el algoritmo original de Canny).
    """
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    return cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)


def apply_roi(edges):
    """
    Define y aplica una Región de Interés - paso 5 de la actividad.

    La ROI es un trapecio que cubre la mitad inferior de la imagen, donde
    está la carretera. Excluimos:
      - La parte superior (cielo, edificios, semáforos)
      - Los extremos laterales lejanos (otras calles, banquetas)

    Forma del trapecio (para imagen 128x96):
        Vertice superior izq:  (32, 48)  - 25% del ancho, 50% de la altura
        Vertice superior der:  (96, 48)  - 75% del ancho, 50% de la altura
        Vertice inferior der:  (128, 96) - esquina inferior derecha
        Vertice inferior izq:  (0, 96)   - esquina inferior izquierda

    Estos porcentajes son parametrizables y funcionan para cualquier
    resolución, no solo 128x96.
    """
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # Trapecio definido por porcentajes para ser independiente de la resolución
    polygon = np.array([[
        (int(width * 0.25), int(height * 0.50)),  # sup izq
        (int(width * 0.75), int(height * 0.50)),  # sup der
        (width, height),                           # inf der
        (0, height),                               # inf izq
    ]], dtype=np.int32)

    # cv2.fillPoly llena el polígono con 255 (blanco) en la máscara
    cv2.fillPoly(mask, polygon, 255)

    # Bitwise AND: conservamos solo los bordes dentro del polígono
    return cv2.bitwise_and(edges, mask)


def detect_lines(edges_roi):
    """
    Detección de líneas con HoughLinesP - paso 6 de la actividad.

    HoughLinesP (probabilística) es preferida sobre HoughLines (estándar)
    porque devuelve segmentos con endpoints (x1,y1,x2,y2), lo cual permite
    calcular el midpoint de cada línea para el cálculo del error.

    Returns:
        Array de líneas con shape (N, 1, 4) o None si no hay detecciones.
    """
    return cv2.HoughLinesP(
        edges_roi,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )


def compute_error(lines, image_width):
    """
    Calcula el error del controlador a partir de las líneas detectadas.

    Lógica (siguiendo las indicaciones de la actividad):
      1. Setpoint = ancho de imagen / 2 (centro horizontal)
      2. Para cada línea, calcular midpoint horizontal: (x1 + x2) / 2
      3. Filtrar líneas mayormente horizontales (alteran el cálculo)
      4. Distancia = midpoint - setpoint (con signo)
      5. Devolver el error de menor magnitud (|distancia| más pequeño)

    El error tiene signo:
      - error > 0: la línea está a la derecha del centro -> girar a la derecha
      - error < 0: la línea está a la izquierda del centro -> girar a la izquierda

    Returns:
        (error, num_valid_lines): el error y cuántas líneas pasaron el filtro.
        Si no hay líneas válidas, devuelve (None, 0).
    """
    if lines is None:
        return None, 0

    setpoint = image_width / 2.0
    smallest_error = None
    valid_lines = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Filtro de líneas mayormente horizontales
        # slope = dy/dx. Líneas casi horizontales tienen |slope| pequeño.
        # Casos especiales:
        #   - dx=0: línea vertical perfecta -> slope infinito -> conservar
        dx = x2 - x1
        if dx == 0:
            slope = float('inf')
        else:
            slope = (y2 - y1) / dx

        if abs(slope) < HORIZONTAL_SLOPE_THRESHOLD:
            continue  # descartar línea horizontal

        valid_lines += 1
        midpoint_x = (x1 + x2) / 2.0
        error = midpoint_x - setpoint

        # Conservar el error de menor magnitud (más cercano al setpoint)
        if smallest_error is None or abs(error) < abs(smallest_error):
            smallest_error = error

    return smallest_error, valid_lines


# =============================================================================
# FUNCIONES DE VISUALIZACIÓN (para depuración y video demo)
# =============================================================================

def draw_lines_on_image(image_bgr, lines, color=(0, 255, 0), thickness=2):
    """Dibuja las líneas detectadas sobre la imagen original (para el display)."""
    output = image_bgr.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), color, thickness)
    return output


def draw_setpoint_line(image_bgr, color=(0, 0, 255)):
    """Dibuja una línea vertical en el setpoint para visualizar el centro."""
    output = image_bgr.copy()
    x = output.shape[1] // 2
    cv2.line(output, (x, 0), (x, output.shape[0]), color, 1)
    return output


def display_image_on_webots(display, image_bgr):
    """
    Muestra una imagen BGR en el display interno de Webots.
    Webots Display espera RGB, así que convertimos.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)


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

    # Teclado (lo conservamos para poder pausar/intervenir manualmente)
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # Velocidad inicial constante (cumple requisito de >=50 km/h)
    driver.setCruisingSpeed(TARGET_SPEED)

    # --- Inicialización del PID ---
    pid = PIDController(kp=PID_KP, ki=PID_KI, kd=PID_KD)
    print(f"[INIT] PID gains: Kp={PID_KP}, Ki={PID_KI}, Kd={PID_KD}")

    # Contadores para logging periódico
    frame_count = 0
    no_line_count = 0
    last_valid_angle = 0.0      # último ángulo aplicado con detección válida
    frames_since_detection = 0  # frames consecutivos sin detección

    # --- Loop principal ---
    while robot.step() != -1:
        # 1. Captura de imagen
        image_bgr = get_image(camera)

        # 2. Pipeline de detección de carriles
        gray = to_grayscale(image_bgr)
        edges = detect_edges(gray)
        edges_roi = apply_roi(edges)
        lines = detect_lines(edges_roi)

        # 3. Cálculo del error
        error, num_valid = compute_error(lines, img_width)

        # 4. PID -> ángulo de dirección
        # Usamos robot.getTime() (tiempo de simulación) en lugar de time.time()
        # para que el PID funcione correctamente si Webots corre a velocidad
        # distinta del tiempo real (por ejemplo en fast-forward).
        current_time = robot.getTime()
        if error is None:
            # No hay líneas detectadas
            frames_since_detection += 1
            no_line_count += 1
            if frames_since_detection <= NO_DETECTION_GRACE_FRAMES:
                # Pérdida breve - mantener último ángulo válido para no
                # desviarse durante sombras/oclusiones momentáneas
                steering_angle = last_valid_angle
            else:
                # Pérdida prolongada (probablemente intersección) -
                # conducir recto y reiniciar el PID
                steering_angle = DEFAULT_STEERING_ANGLE
                pid.reset()
        else:
            # Detección válida - calcular con PID y guardar como última válida
            steering_angle = pid.compute(error, current_time)
            last_valid_angle = steering_angle
            frames_since_detection = 0

        # 5. Aplicar slew-rate limit y aplicar al vehículo
        # Limitamos cuánto puede cambiar el ángulo entre frames consecutivos
        # para evitar movimientos bruscos por ruido en la detección.
        previous_applied_angle = getattr(main, '_prev_angle', 0.0)
        max_change = MAX_ANGLE_CHANGE_PER_FRAME
        if steering_angle - previous_applied_angle > max_change:
            steering_angle = previous_applied_angle + max_change
        elif steering_angle - previous_applied_angle < -max_change:
            steering_angle = previous_applied_angle - max_change
        main._prev_angle = steering_angle

        driver.setSteeringAngle(steering_angle)
        driver.setCruisingSpeed(TARGET_SPEED)

        # 6. Visualización en el display de Webots
        # Dibujamos las líneas detectadas y el setpoint sobre la imagen original
        debug_image = draw_lines_on_image(image_bgr, lines)
        debug_image = draw_setpoint_line(debug_image)
        display_image_on_webots(display_img, debug_image)

        # Log cada 50 frames (~0.5 segundos a 10ms timestep)
        frame_count += 1
        if frame_count % 50 == 0:
            err_str = f"{error:+.2f}" if error is not None else "  N/A"
            n_lines = len(lines) if lines is not None else 0
            print(f"[F{frame_count}] lines={n_lines} valid={num_valid} "
                  f"err={err_str} angle={steering_angle:+.4f} "
                  f"no_line_total={no_line_count}")


if __name__ == "__main__":
    main()
