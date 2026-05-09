# Actividad 2.1 – Detección de Carriles con OpenCV

**MR4010 – Navegación Autónoma**  
Tecnológico de Monterrey | Maestría en Robótica y Sistemas Digitales  
Equipo 18

---

## Descripción

Controlador autónomo de seguimiento de carriles para un vehículo BMW X5 simulado en **Webots R2025a**. Utiliza visión por computadora con OpenCV para detectar las líneas amarillas centrales de la calzada y un controlador PID para mantener el vehículo centrado en su carril a una velocidad mínima de 50 km/h.

### Pipeline de visión

```
Cámara (BGRA 128×64) → Escala de grises → Máscara amarilla (HSV)
→ Canny → ROI (fillPoly) → HoughLinesP → Cálculo de error
→ PID → setSteeringAngle()
```

---

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `autonomous_lane_controller.py` | Controlador autónomo con detección de carriles y PID |
| `simple_controller_act_2_1.py` | Controlador manual con teclado (base de la actividad) |
| `city_2025a.wbt` | Mundo de Webots con ciudad y BMW X5 |
| `probe_camera.py` | Diagnóstico de cámara y valores HSV |
| `probe_save_frames.py` | Guardado de frames para análisis de detección |

---

## Requisitos

- **Webots R2025a**
- **Python 3.10** (conda environment recomendado)
- `opencv-python`
- `numpy`

---

## Configuración del entorno

```bash
# Variables de entorno (agregar a ~/.zshrc o ~/.bashrc)
export WEBOTS_HOME=/Applications/Webots.app
export PYTHONPATH=/Applications/Webots.app/Contents/lib/controller/python
export WEBOTS_PYTHON_EXECUTABLE=$(which python)

# Activar entorno conda
conda activate webots
```

---

## Uso

1. Abrir `city_2025a.wbt` en Webots R2025a
2. En la configuración del robot BMW X5, establecer el controlador como `<extern>`
3. Ejecutar el controlador desde terminal:

```bash
python autonomous_lane_controller.py
```

---

## Parámetros principales

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `TARGET_SPEED` | 50 km/h | Velocidad de crucero |
| `MAX_STEERING_ANGLE` | 0.5 rad | Límite mecánico del vehículo |
| `CANNY_LOW / HIGH` | 50 / 150 | Umbrales del detector de bordes |
| `HOUGH_THRESHOLD` | 20 | Votos mínimos para detectar una línea |
| `HOUGH_MIN_LINE_LEN` | 15 px | Longitud mínima de segmento |
