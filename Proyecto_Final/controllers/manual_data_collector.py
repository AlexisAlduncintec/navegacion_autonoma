"""
================================================================================
Proyecto Final - Recolector de datos para Conditional Imitation Learning (CIL)
MR4010 - Navegación Autónoma | Tecnológico de Monterrey | MNA
Equipo 18:
  - Alexis Alduncin Barragán          - A01017478
  - David Rodrigo Alvarado Domínguez  - A01797606
  - Abraham Avila Garcia              - A01795305
  - Jorge Luis Ancheyta Segovia       - A01796354
Profesor: Dr. David Antonio Torres
================================================================================

Controlador externo de CONDUCCIÓN MANUAL para recolectar el dataset de CIL
(Codevilla et al., 2018, "End-to-end Driving via Conditional Imitation Learning")
sobre el BMW X5 del mundo de entrenamiento:

    Proyecto_Final/worlds/city_traffic_2025_01.wbt   (controller "<extern>")

El humano conduce con el teclado mientras el controlador captura
AUTOMÁTICAMENTE, cada CAPTURE_EVERY_N_STEPS pasos (~10 Hz), un frame de cámara
(200x88, tamaño de entrada de Codevilla) junto con:
    - el ángulo de dirección comandado por el conductor (etiqueta de regresión)
    - el COMANDO DE NAVEGACIÓN activo (sticky): seguir carril / izquierda /
      derecha / recto (la variable condicional "c" del paper, sección V)
    - throttle / brake / velocidad / timestamp

Todo se registra en Proyecto_Final/dataset/driving_log.csv (modo append, nunca
se sobrescribe) con las imágenes en Proyecto_Final/dataset/IMG/{session_id}/.

Extiende el controlador manual de Actividad 2.1 (simple_controller_act_2_1.py):
mismo esqueleto Car/Driver/Keyboard/Camera/Display, pero se ELIMINA el guardado
manual con la tecla 'A' y se sustituye por captura periódica automática, más el
estado de comando de navegación, el HUD en vivo, la inyección opcional de ruido
de dirección y el registro a CSV.

--------------------------------------------------------------------------------
DECISIONES DE DISEÑO (cada cambio justificado)
--------------------------------------------------------------------------------
  - Loop usa driver.step() en vez de robot.step(). En 3.1 documentamos que
    Driver mantiene un buffer interno de comandos que sólo se vacía cuando
    llamas driver.step(); con robot.step() los comandos se acumulan y el coche
    reacciona con un frame de retraso. Lo aprendimos por las malas.
  - display.imageDelete() es OBLIGATORIO después de imagePaste(). Webots reserva
    un slot interno por cada imageNew() y si no se libera, tras ~350 iteraciones
    wbu_driver_step() se queda sin slots y deadlockea sin error visible. Bug
    encontrado en 3.1.
  - El Display de ESTE mundo no tiene campo 'name' -> su nombre de dispositivo
    es el default de Webots: "display" (NO "display_image" como en 2.1/3.1/4.2).
    resolve_display() lo busca de forma robusta: prueba "display" y luego
    "display_image"; aborta sólo si ninguno conecta. Así no tocamos el .wbt.
  - basicTimeStep del mundo = 16 ms (no 32). Con CAPTURE_EVERY_N_STEPS=6 da
    ~10.4 Hz, la tasa que usa Codevilla. La duración de la ráfaga de ruido y los
    logs periódicos se derivan del timestep en runtime para ser correctos sea
    cual sea el mundo.
  - La cámara de World #1 NO tiene nodo Recognition, así que NO llamamos
    camera.recognitionEnable() (no se necesita para recolectar). El BMW tampoco
    es Supervisor, así que NO hay teleport (a diferencia de 4.2): arranca en su
    spawn (-238.13, 45.88).
  - El ángulo etiquetado en el CSV es el del CONDUCTOR, no el aplicado con ruido
    (Codevilla §IV.C: se usa la respuesta correctiva del conductor al ruido
    inyectado, no el ruido en sí).

--------------------------------------------------------------------------------
MAPA DE TECLADO
--------------------------------------------------------------------------------
  Movimiento (debug; la recolección debe mantenerse en TARGET_SPEED_KMH=25):
    UP / DOWN     - velocidad +/- 5 km/h (tope debug MAX_SPEED_KMH=60)
    LEFT / RIGHT  - dirección -/+ ANGLE_INCR
    SPACE         - freno de emergencia (velocidad=0 mientras se mantiene)

  Comandos de navegación CIL (sticky, persisten hasta cambiarlos):
    F             - Seguir carril   (command=2, default)
    L             - Izquierda en la siguiente intersección (command=3)
    K             - Derecha en la siguiente intersección   (command=4)  [NO 'R']
    S             - Recto en la siguiente intersección      (command=5)

  Controles de grabación:
    P             - Pausar/reanudar grabación (no se guardan frames en pausa)
    N             - Inyectar ráfaga de ruido de dirección (~6 s, recuperación)
    Q             - Salir limpio (cierra el CSV)

  NOTA: la tecla 'A' (captura manual) de 2.1 se ELIMINA. La captura es
  automática cada CAPTURE_EVERY_N_STEPS pasos.

--------------------------------------------------------------------------------
USO (Mac / R2025a)
--------------------------------------------------------------------------------
  # Terminal 1 - Webots
  pkill -KILL -x webots 2>/dev/null; pkill -KILL -f "webots-controller" 2>/dev/null; sleep 2
  cd ~/ITESM/navegacion_autonoma
  /Applications/Webots.app/Contents/MacOS/webots --mode=realtime --port=1234 \
    Proyecto_Final/worlds/city_traffic_2025_01.wbt

  # Terminal 2 - Controlador
  conda activate webots
  export WEBOTS_HOME=/Applications/Webots.app
  export PYTHONPATH=/Applications/Webots.app/Contents/lib/controller/python
  export PYTHONUNBUFFERED=1
  /Applications/Webots.app/Contents/MacOS/webots-controller \
    --port=1234 --robot-name=vehicle \
    $(which python) -u \
    /Users/alexisalduncin/ITESM/navegacion_autonoma/Proyecto_Final/controllers/manual_data_collector.py
================================================================================
"""

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
from datetime import datetime
import numpy as np
import cv2
import math
import os
import sys
import csv
import time

# =============================================================================
# CONSTANTES DE CONFIGURACIÓN
# =============================================================================

# --- Conducción ---
TARGET_SPEED_KMH      = 25.0    # crucero constante por especificación (<30 km/h)
MAX_SPEED_KMH         = 60.0    # Tope SÓLO para debug (tecla UP). La recolección
                                # debe mantenerse en TARGET_SPEED_KMH=25.0.
SPEED_INCR            = 5.0     # km/h por pulsación de UP/DOWN
MAX_ANGLE             = 0.5     # rad
ANGLE_INCR            = 0.05    # rad por pulsación de LEFT/RIGHT
DEBOUNCE_TIME         = 0.1     # s, anti-rebote del teclado

# --- Captura ---
CAPTURE_EVERY_N_STEPS = 6       # ~10.4 Hz a basicTimeStep=16 ms
IMAGE_OUT_W           = 200     # ancho de entrada CIL (Codevilla)
IMAGE_OUT_H           = 88      # alto de entrada CIL (Codevilla)
JPEG_QUALITY          = 90

# --- Ruido de dirección (Codevilla §IV.C, apagado por defecto) ---
NOISE_MAX             = 0.15    # rad, amplitud del triángulo
NOISE_BURST_S         = 6.0     # duración total de la ráfaga (sube 3 s, baja 3 s)

# --- Códigos de comando (paper sección V / columnas del dataset HDF5) ---
CMD_FOLLOW_LANE       = 2
CMD_TURN_LEFT         = 3
CMD_TURN_RIGHT        = 4
CMD_GO_STRAIGHT       = 5

# Texto y color (0xRRGGBB) por comando, para el HUD y los logs.
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

# --- Rutas (relativas a este archivo: Proyecto_Final/controllers/) ---
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "dataset"))
CSV_PATH    = os.path.join(DATASET_DIR, "driving_log.csv")
CSV_HEADER  = ["image_path", "steering", "throttle", "brake",
               "command", "speed", "session_id", "timestamp_ms"]


# =============================================================================
# HELPERS
# =============================================================================

def get_camera_image(camera):
    """Devuelve la imagen de la cámara como ndarray BGRA (H, W, 4), o None si
    aún no hay frame disponible (puede pasar en el primer paso)."""
    raw = camera.getImage()
    if raw is None:
        return None
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )


def resolve_display(robot):
    """Localiza el Display de forma robusta. En este mundo el nodo Display no
    tiene 'name', así que su nombre es el default "display"; los mundos de
    2.1/4.2 lo llamaban "display_image". Probamos ambos y devolvemos el primero
    que conecte, o None si ninguno existe (ver docstring)."""
    for name in ("display", "display_image"):
        dev = robot.getDevice(name)
        if dev is not None:
            return dev
    return None


def paste_camera_to_display(display, image_bgr, dw, dh):
    """Pinta la imagen BGR de la cámara en el Display (redimensionada al tamaño
    del Display, NO a 200x88). OBLIGATORIO imageDelete tras imagePaste: ver
    docstring (fuga de slots de 3.1)."""
    resized = cv2.resize(image_bgr, (dw, dh))
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image_ref = display.imageNew(
        image_rgb.tobytes(), Display.RGB, width=dw, height=dh
    )
    display.imagePaste(image_ref, 0, 0, False)
    display.imageDelete(image_ref)  # evita la fuga de slots de 3.1


def draw_hud(display, dw, dh, command, recording_enabled, speed_kmh,
             driver_steering, noise_active, session_id, frames_saved):
    """Dibuja el overlay sobre la imagen ya pegada. Usa drawText de Webots (no
    OpenCV) para no sobrescribir el frame. Layout para Display 200x150.

    El HUD es esencial: si el conductor no ve con qué comando está grabando,
    puede etiquetar mal y envenenar el dataset."""
    rec_x = dw - 56                 # esquina superior derecha
    row_bottom_1 = dh - 32
    row_bottom_2 = dh - 16

    # Arriba-izquierda: comando activo (color-coded para escaneo visual rápido).
    display.setColor(CMD_COLOR.get(command, 0xFFFFFF))
    display.drawText(f"CMD: {CMD_TEXT.get(command, '?')}", 4, 4)

    # Arriba-derecha: estado de grabación.
    if recording_enabled:
        display.setColor(0x00FF00)   # verde
        display.drawText("REC ON", rec_x, 4)
    else:
        display.setColor(0xFF0000)   # rojo
        display.drawText("REC OFF", rec_x, 4)

    # Centro: ruido activo (sólo durante la ráfaga).
    if noise_active:
        display.setColor(0xFF0000)
        display.drawText("NOISE: ACTIVE", 4, dh // 2 - 8)

    # Abajo-izquierda: velocidad + dirección (compacto para caber en 200 px).
    display.setColor(0xFFFFFF)
    display.drawText(f"spd:{speed_kmh:4.1f} st:{driver_steering:+.2f}",
                     4, row_bottom_1)
    # Abajo: frames + sesión.
    display.drawText(f"f:{frames_saved} {session_id}", 4, row_bottom_2)


def compute_triangular_noise(phase_steps, burst_steps, noise_max):
    """Onda triangular determinista (sin RNG). Rampa -noise_max -> +noise_max en
    la primera mitad de la ráfaga y +noise_max -> -noise_max en la segunda.
    Fuera de [0, burst_steps) devuelve 0."""
    if phase_steps < 0 or phase_steps >= burst_steps:
        return 0.0
    half = burst_steps / 2.0
    if phase_steps < half:
        return -noise_max + 2.0 * noise_max * (phase_steps / half)
    return noise_max - 2.0 * noise_max * ((phase_steps - half) / half)


def handle_keyboard(keyboard, st, now, step_count):
    """Lee TODAS las teclas presionadas este paso y actualiza el estado 'st'.
    Devuelve (brake_active, quit_requested).

    Las teclas incrementales/toggle usan anti-rebote (flanco de subida cada
    DEBOUNCE_TIME); SPACE se trata como nivel (freno mientras se mantenga)."""
    keys = set()
    k = keyboard.getKey()
    while k != -1:
        keys.add(k & 0xFFFF)        # quita bits de modificadores (SHIFT/CTRL...)
        k = keyboard.getKey()

    brake_active = ord(' ') in keys
    quit_requested = False

    def pressed(code):
        """True sólo en el flanco de subida tras DEBOUNCE_TIME (anti-rebote)."""
        if code in keys and (now - st["last_press"].get(code, 0.0) >= DEBOUNCE_TIME):
            st["last_press"][code] = now
            return True
        return False

    # --- Movimiento (debug) ---
    if pressed(Keyboard.UP):
        st["speed"] = min(st["speed"] + SPEED_INCR, MAX_SPEED_KMH)
    if pressed(Keyboard.DOWN):
        st["speed"] = max(st["speed"] - SPEED_INCR, 0.0)
    if pressed(Keyboard.RIGHT):
        st["angle"] = min(st["angle"] + ANGLE_INCR, MAX_ANGLE)
    if pressed(Keyboard.LEFT):
        st["angle"] = max(st["angle"] - ANGLE_INCR, -MAX_ANGLE)

    # --- Comandos de navegación CIL (sticky) ---
    if pressed(ord('F')):
        st["command"] = CMD_FOLLOW_LANE
    if pressed(ord('L')):
        st["command"] = CMD_TURN_LEFT
    if pressed(ord('K')):
        st["command"] = CMD_TURN_RIGHT
    if pressed(ord('S')):
        st["command"] = CMD_GO_STRAIGHT

    # --- Controles de grabación ---
    if pressed(ord('P')):
        st["recording_enabled"] = not st["recording_enabled"]
        print(f"[CTRL] Grabación {'ON' if st['recording_enabled'] else 'OFF'}")
    if pressed(ord('N')):
        if not st["noise_enabled"]:
            st["noise_enabled"] = True
            st["noise_start_step"] = step_count
            print("[CTRL] Ruido de dirección: ráfaga iniciada (~6 s)")
    if pressed(ord('Q')):
        quit_requested = True

    return brake_active, quit_requested


def print_keyboard_map():
    print("[INIT] KEYBOARD:")
    print("[INIT]   UP/DOWN=speed+-5  LEFT/RIGHT=steer-+  SPACE=brake")
    print("[INIT]   F=follow  L=left  K=right  S=straight   (sticky nav command)")
    print("[INIT]   P=pause/resume  N=noise burst  Q=quit")


# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

def main():
    # --- Inicialización de Webots ---
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    session_id = "s" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Safety check 1: cámara ---
    camera = robot.getDevice("camera")
    if camera is None:
        print("[ERROR] No se encontró la cámara 'camera'. Abortando.")
        sys.exit(1)
    camera.enable(timestep)
    # NOTA: NO recognitionEnable (la cámara de World #1 no tiene nodo Recognition).

    # --- Safety check 2: display ---
    display = resolve_display(robot)
    if display is None:
        print("[ERROR] Display widget not found - dataset will be unauditable. "
              "Aborting.")
        sys.exit(1)
    dw, dh = display.getWidth(), display.getHeight()

    # --- Safety check 3: directorio del dataset ---
    # exist_ok=False es INTENCIONAL: fallar ruidosamente en vez de sobrescribir.
    # Sobrescribir envenenaría el dataset al mezclar frames de corridas distintas
    # bajo un mismo session_id (hallazgo del smoke-test: dos arranques en el mismo
    # minuto colisionaban). Con segundos en el session_id, una colisión sólo
    # ocurriría si se reinicia dentro del mismo segundo -> abortamos y avisamos.
    img_dir = os.path.join(DATASET_DIR, "IMG", session_id)
    try:
        os.makedirs(img_dir, exist_ok=False)
    except FileExistsError:
        print(f"[ERROR] Session directory already exists: {img_dir}")
        print("[ERROR] This indicates a duplicate session_id collision. "
              "Restart the controller.")
        sys.exit(1)
    except OSError as e:
        print(f"[ERROR] No se pudo crear el directorio del dataset: {e}. Abortando.")
        sys.exit(1)

    # --- Safety check 4: CSV escribible (modo append; crea si no existe) ---
    try:
        is_new = not os.path.exists(CSV_PATH)
        csv_file = open(CSV_PATH, "a", newline="")
        writer = csv.writer(csv_file)
        if is_new:
            writer.writerow(CSV_HEADER)   # cabecera SÓLO si el archivo es nuevo
            csv_file.flush()
    except OSError as e:
        print(f"[ERROR] CSV no escribible ({CSV_PATH}): {e}. Abortando.")
        sys.exit(1)

    # --- Teclado ---
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # --- Banner de inicio ---
    print(f"[INIT] Session ID: {session_id}")
    print(f"[INIT] Output: {img_dir}/")
    print(f"[INIT] CSV: {CSV_PATH}")
    print(f"[INIT] Speed: {TARGET_SPEED_KMH:.0f} km/h constant "
          f"(debug cap {MAX_SPEED_KMH:.0f})")
    print(f"[INIT] Capture: every {CAPTURE_EVERY_N_STEPS} steps "
          f"(~{1000.0/(timestep*CAPTURE_EVERY_N_STEPS):.1f} Hz at {timestep} ms)")
    print(f"[INIT] Image size: {IMAGE_OUT_W}x{IMAGE_OUT_H} RGB")
    print(f"[INIT] Display: {dw}x{dh}")
    print(f"[INIT] Initial command: FOLLOW_LANE ({CMD_FOLLOW_LANE})")
    print_keyboard_map()
    print("[INIT] >>> READY. Drive carefully. Press F before normal driving.")

    # --- Estado del controlador (locals, pasados explícitamente) ---
    st = {
        "speed": TARGET_SPEED_KMH,      # km/h objetivo (ajustable por debug)
        "angle": 0.0,                   # rad, dirección del conductor
        "command": CMD_FOLLOW_LANE,     # comando de navegación sticky
        "recording_enabled": True,
        "noise_enabled": False,
        "noise_start_step": 0,
        "last_press": {},               # anti-rebote por tecla
    }
    burst_steps = int(NOISE_BURST_S * 1000.0 / timestep)
    step_count = 0
    frames_saved = 0
    cmd_counts = {CMD_FOLLOW_LANE: 0, CMD_TURN_LEFT: 0,
                  CMD_TURN_RIGHT: 0, CMD_GO_STRAIGHT: 0}

    driver.setCruisingSpeed(TARGET_SPEED_KMH)

    # --- Loop principal ---
    # driver.step() (no robot.step()) para vaciar el buffer del Driver. Ver
    # docstring del archivo.
    try:
        while driver.step() != -1:
            step_count += 1

            # 1. Captura de cámara (siempre).
            bgra = get_camera_image(camera)
            if bgra is None:
                continue
            bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

            # 2. Teclado (anti-rebote); actualiza ángulo/velocidad/comando/flags.
            now = time.time()
            brake_active, quit_requested = handle_keyboard(
                keyboard, st, now, step_count)

            # 3. Ruido de dirección (si la ráfaga está activa).
            noise_offset = 0.0
            if st["noise_enabled"]:
                phase = step_count - st["noise_start_step"]
                if phase >= burst_steps:
                    st["noise_enabled"] = False
                    print("[CTRL] Ruido de dirección: ráfaga finalizada")
                else:
                    noise_offset = compute_triangular_noise(
                        phase, burst_steps, NOISE_MAX)
            noise_active = st["noise_enabled"]

            # 4. Aplicar control al Driver. OJO: se aplica ángulo+ruido, pero al
            #    CSV se registra el ángulo del CONDUCTOR (Codevilla §IV.C).
            applied_speed = 0.0 if brake_active else st["speed"]
            driver.setSteeringAngle(st["angle"] + noise_offset)
            driver.setCruisingSpeed(applied_speed)

            # 5. HUD (cada frame): pega cámara + dibuja overlay.
            paste_camera_to_display(display, bgr, dw, dh)
            draw_hud(display, dw, dh, st["command"], st["recording_enabled"],
                     st["speed"], st["angle"], noise_active,
                     session_id, frames_saved)

            # 6. Captura automática cada N pasos si la grabación está activa.
            if st["recording_enabled"] and step_count % CAPTURE_EVERY_N_STEPS == 0:
                cur = driver.getCurrentSpeed()
                if cur is None or math.isnan(cur):
                    cur = 0.0
                cur = max(cur, 0.0)
                throttle = 0.0 if brake_active else 1.0   # proxy (usamos crucero)
                brake = 1.0 if brake_active else 0.0
                ts_ms = int(robot.getTime() * 1000)

                fname = f"center_{frames_saved:06d}.jpg"
                resized = cv2.resize(bgr, (IMAGE_OUT_W, IMAGE_OUT_H))
                cv2.imwrite(os.path.join(img_dir, fname), resized,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                rel_path = f"IMG/{session_id}/{fname}"
                writer.writerow([
                    rel_path, f"{st['angle']:.6f}", f"{throttle:.1f}",
                    f"{brake:.1f}", st["command"], f"{cur:.3f}",
                    session_id, ts_ms,
                ])
                csv_file.flush()

                cmd_counts[st["command"]] += 1
                frames_saved += 1

                # 7. Logs por consola con throttling.
                if frames_saved % 50 == 0:
                    print(f"[DATA] frame={frames_saved} "
                          f"cmd={CMD_TEXT[st['command']]} "
                          f"steer={st['angle']:+.3f} speed={cur:.1f} "
                          f"noise={'ON' if noise_active else 'OFF'}")
                if frames_saved % 250 == 0:
                    print(f"[SESSION] {session_id}: {frames_saved} frames saved "
                          f"(cmd dist: FOLLOW={cmd_counts[CMD_FOLLOW_LANE]} "
                          f"LEFT={cmd_counts[CMD_TURN_LEFT]} "
                          f"RIGHT={cmd_counts[CMD_TURN_RIGHT]} "
                          f"STRAIGHT={cmd_counts[CMD_GO_STRAIGHT]})")

            # 8. Salida limpia.
            if quit_requested:
                print(f"[QUIT] Session {session_id} complete. "
                      f"{frames_saved} frames saved.")
                break
    finally:
        csv_file.close()


if __name__ == "__main__":
    main()
