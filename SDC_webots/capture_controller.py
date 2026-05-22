"""
================================================================================
Actividad 3.1 - Controlador de CAPTURA de frames para reentrenamiento del SVM
MR4010 - Navegación Autónoma | Tec de Monterrey | MNA | Equipo 18
================================================================================

Propósito:
    Recoger frames REALES de la cámara del BMW dentro del mundo de Webots
    (mannequins renderizados, no fotos del Penn-Fudan dataset) para luego
    reentrenar el SVM con datos del dominio correcto.

Diferencias respecto a autonomous_obstacle_controller.py:
    - NO carga el SVM y NO clasifica (estaríamos contaminando las etiquetas
      con la salida del clasificador viejo, que es justamente lo que queremos
      reemplazar).
    - NO frena por obstáculos (queremos pasar al lado de peatones y barriles
      sin detenernos para que la cámara los registre desde distintas distancias).
    - Velocidad reducida a 25 km/h para que cada frame capturado contenga
      contexto distinto sin saltarse pedazos del recorrido.
    - Sigue usando el lane keeper con Fix 1 (mantener last_valid_angle en
      intersecciones, bajar a LANE_LOST_SPEED_KMH cuando se pierde la línea).

Salida:
    SDC_webots/captured_frames/frame_NNNNNN.png   — cámara BGR 256x128
    SDC_webots/captured_frames/metadata.csv       — sim_time, dist LiDAR

Uso (Mac, R2023b):
    1. Abrir Webots con worlds/city_2025a_activity_3_1.wbt
    2. En otra terminal:
         conda activate webots
         export WEBOTS_HOME=/Applications/Webots-R2023b.app
         export PYTHONPATH=/Applications/Webots-R2023b.app/Contents/lib/controller/python
         export PYTHONUNBUFFERED=1
         /Applications/Webots-R2023b.app/Contents/MacOS/webots-controller \\
             --robot-name=vehicle \\
             $(which python) -u SDC_webots/capture_controller.py

    El controlador se detiene solo después de CAPTURE_MAX_FRAMES.
================================================================================
"""

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import csv
import math
import os
import sys

import cv2
import numpy as np

# =============================================================================
# CONFIGURACIÓN DE CAPTURA
# =============================================================================

CAPTURE_DIR        = "captured_frames"   # relativo al directorio del .py
CAPTURE_EVERY_N    = 16                  # frames de sim entre capturas (16*32ms = 0.5 s)
CAPTURE_MAX_FRAMES = 600                 # ~5 min de sim a 0.5 s/frame
CAPTURE_SPEED_KMH  = 25.0                # más lento para no saltarse contexto
CAPTURE_LIDAR_NAME = "Sick LMS 291"
CAPTURE_LIDAR_FOV_CENTRAL_DEG = 25.0
CAPTURE_LIDAR_FOV_TOTAL_DEG   = 180.0

# Lane keeping (mismo que controlador principal, Fix 1 incluido)
PID_KP = 0.010
PID_KI = 0.0
PID_KD = 0.002
MAX_STEERING_ANGLE = 0.5
MAX_ANGLE_CHANGE_PER_FRAME = 0.05
NO_DETECTION_GRACE_FRAMES = 20
LANE_LOST_SPEED_KMH = 15.0   # aún más lento en cruces durante captura
CANNY_LOW, CANNY_HIGH = 50, 150
HOUGH_RHO, HOUGH_THETA = 1, np.pi / 180
HOUGH_THRESHOLD, HOUGH_MIN_LINE_LEN, HOUGH_MAX_LINE_GAP = 20, 15, 20
HORIZONTAL_SLOPE_THRESHOLD = 0.05


# =============================================================================
# PID y pipeline de carril (copia mínima, sin SVM ni state machine)
# =============================================================================
class PIDController:
    def __init__(self, kp, ki, kd, output_limit=MAX_STEERING_ANGLE, integral_limit=100.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def reset(self):
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
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(-self.output_limit, min(self.output_limit, output))
        self.previous_error, self.previous_time = error, current_time
        return output


def get_image(camera):
    raw = camera.getImage()
    image = np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image[:, :, :3]


def process_lane(image_bgr, img_width):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([40, 255, 255]))
    gray = cv2.cvtColor(cv2.bitwise_and(image_bgr, image_bgr, mask=mask),
                        cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    h, w = edges.shape
    roi_mask = np.zeros_like(edges)
    poly = np.array([[
        (int(w * 0.25), int(h * 0.50)),
        (int(w * 0.75), int(h * 0.50)),
        (w, h), (0, h)]], dtype=np.int32)
    cv2.fillPoly(roi_mask, poly, 255)
    edges_roi = cv2.bitwise_and(edges, roi_mask)
    lines = cv2.HoughLinesP(edges_roi, HOUGH_RHO, HOUGH_THETA,
                            threshold=HOUGH_THRESHOLD,
                            minLineLength=HOUGH_MIN_LINE_LEN,
                            maxLineGap=HOUGH_MAX_LINE_GAP)
    if lines is None:
        return None
    setpoint = img_width / 2.0
    smallest = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        slope = float('inf') if dx == 0 else (y2 - y1) / dx
        if abs(slope) < HORIZONTAL_SLOPE_THRESHOLD:
            continue
        mid = (x1 + x2) / 2.0
        err = mid - setpoint
        if smallest is None or abs(err) < abs(smallest):
            smallest = err
    return smallest


def init_lidar(robot, timestep):
    lidar = robot.getDevice(CAPTURE_LIDAR_NAME)
    lidar.enable(timestep)
    lidar.enablePointCloud()
    horiz = lidar.getHorizontalResolution()
    central_n = max(1, int(horiz * CAPTURE_LIDAR_FOV_CENTRAL_DEG /
                           CAPTURE_LIDAR_FOV_TOTAL_DEG))
    mid = horiz // 2
    half = central_n // 2
    return lidar, slice(mid - half, mid + half)


def read_forward_distance(lidar, central):
    ranges = lidar.getRangeImage()[central]
    finite = [r for r in ranges if math.isfinite(r)]
    return min(finite) if finite else float("inf")


# =============================================================================
# LOOP PRINCIPAL — captura sin clasificación
# =============================================================================
def main():
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)
    img_width = camera.getWidth()
    img_height = camera.getHeight()
    print(f"[CAPTURE] Cámara: {img_width}x{img_height}")

    lidar, lidar_central = init_lidar(robot, timestep)
    print(f"[CAPTURE] LiDAR habilitado")

    pid = PIDController(PID_KP, PID_KI, PID_KD)

    # Directorio absoluto basado en la ubicación de este .py
    controller_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(controller_dir, CAPTURE_DIR)
    os.makedirs(out_dir, exist_ok=True)

    metadata_path = os.path.join(out_dir, "metadata.csv")
    metadata_file = open(metadata_path, "w", newline="")
    metadata_csv = csv.writer(metadata_file)
    metadata_csv.writerow(["frame_id", "sim_frame", "sim_time_s",
                            "forward_dist_m", "speed_kmh"])
    print(f"[CAPTURE] Guardando en {out_dir}/")

    driver.setCruisingSpeed(CAPTURE_SPEED_KMH)

    frame_count = 0
    capture_id = 0
    last_valid_angle = 0.0
    frames_since_detection = 0
    prev_angle = 0.0
    last_cruising_speed = CAPTURE_SPEED_KMH

    while driver.step() != -1:
        frame_count += 1
        image_bgr = get_image(camera)
        error = process_lane(image_bgr, img_width)

        # Steering: Fix 1 incluido
        lane_lost_extended = False
        current_time = robot.getTime()
        if error is None:
            frames_since_detection += 1
            steering_angle = last_valid_angle
            if frames_since_detection > NO_DETECTION_GRACE_FRAMES:
                lane_lost_extended = True
        else:
            steering_angle = pid.compute(error, current_time)
            last_valid_angle = steering_angle
            frames_since_detection = 0

        max_change = MAX_ANGLE_CHANGE_PER_FRAME
        if steering_angle - prev_angle > max_change:
            steering_angle = prev_angle + max_change
        elif steering_angle - prev_angle < -max_change:
            steering_angle = prev_angle - max_change
        prev_angle = steering_angle

        # Velocidad con debounce
        target_speed = LANE_LOST_SPEED_KMH if lane_lost_extended else CAPTURE_SPEED_KMH
        if target_speed != last_cruising_speed:
            driver.setCruisingSpeed(target_speed)
            last_cruising_speed = target_speed
        driver.setSteeringAngle(steering_angle)

        # === Captura ===
        if frame_count % CAPTURE_EVERY_N == 0:
            forward_m = read_forward_distance(lidar, lidar_central)
            filename = f"frame_{capture_id:06d}.png"
            cv2.imwrite(os.path.join(out_dir, filename), image_bgr)
            metadata_csv.writerow([
                capture_id,
                frame_count,
                f"{current_time:.3f}",
                f"{forward_m:.2f}" if math.isfinite(forward_m) else "inf",
                f"{target_speed:.1f}",
            ])
            metadata_file.flush()
            if capture_id % 20 == 0:
                print(f"[CAPTURE] frame {capture_id:04d}  sim_t={current_time:.1f}s  "
                      f"dist={forward_m:.1f}m  speed={target_speed:.0f}kmh")
            capture_id += 1
            if capture_id >= CAPTURE_MAX_FRAMES:
                print(f"[CAPTURE] Llegamos a CAPTURE_MAX_FRAMES={CAPTURE_MAX_FRAMES}, parando.")
                break

    metadata_file.close()
    print(f"[CAPTURE] Done. {capture_id} frames guardados en {out_dir}/")


if __name__ == "__main__":
    main()
