# Actividad 3.1 — Detección de Peatones y Barriles con SVM + LiDAR

**MR4010 Navegación Autónoma · MNA · Tec de Monterrey · Equipo 18**
**Fecha de entrega: domingo 24 de mayo de 2026, 23:59**

---

## Qué hace esta actividad (resumen de 30 segundos)

El vehículo autónomo (BMW X5) recorre el mundo de Webots haciendo cuatro cosas a la vez:

1. **Sigue la línea amarilla** del carril con un controlador PID heredado de la
   Actividad 2.1, con un fix de "setpoint" para que el coche se mantenga en su
   carril derecho en lugar de pisar la línea central.
2. **Detecta obstáculos al frente** con un LiDAR Sick (sector central de 25°,
   hasta 20 metros).
3. **Clasifica el obstáculo** con un modelo SVM entrenado con HOG sobre la imagen
   de la cámara: ¿es peatón o es barril? El SVM fue reentrenado con frames del
   propio simulador (no fotos reales) para reducir la brecha de dominio.
4. **Frena de emergencia** según el tipo de obstáculo:
   - **Barril** → frena + enciende las **luces intermitentes** + espera a que el
     barril desaparezca.
   - **Peatón** → frena **sin** luces intermitentes + espera a que el peatón se
     quite del frente.

---

## Requisitos (lean antes de empezar)

- **macOS** (las instrucciones asumen Mac; en Windows cambian las rutas).
- **Webots R2023b** instalado en `/Applications/Webots-R2023b.app`.
  - *Importante:* el mundo de la actividad usa EXTERNPROTOs apuntados a R2023b
    y se cuelga determinísticamente bajo R2025a. **Usen R2023b, no R2025a.**
    Si Webots R2023b no está instalado, lo bajan del repo oficial de
    Cyberbotics (`webots-R2023b.dmg`, ~106 MB) y lo arrastran a `/Applications/`
    renombrándolo a `Webots-R2023b.app` para no chocar con cualquier R2025a
    que ya tengan.
- **miniconda** con un environment llamado `webots` (Python 3.10).
- **Git** para clonar el repo.

---

## Paso 1 — Clonar el repositorio

```bash
cd ~/ITESM
git clone <URL_DEL_REPO> navegacion_autonoma
cd navegacion_autonoma/SDC_webots
```

Si Alexis ya les compartió la carpeta directamente (por Drive o USB), solo
cópienla a `~/ITESM/navegacion_autonoma/` y sáltense este paso.

---

## Paso 2 — Preparar el environment de Python (una sola vez)

El error más común del equipo es que Python no encuentra las librerías de
Webots o las dependencias del clasificador. Copia y pega TODO este bloque:

```bash
conda activate webots
pip install scikit-learn scikit-image joblib numpy opencv-python
```

Verifica que todo importa sin errores:

```bash
conda run -n webots python -c "import numpy, cv2, sklearn, skimage, joblib; print('TODO OK')"
```

Debe imprimir `TODO OK`. Si marca `ImportError`, avísale a Alexis antes de
seguir.

---

## Paso 3 — Confirmar los archivos del modelo SVM

El controlador necesita tres archivos `.joblib`. Verifica que existen:

```bash
ls -la ~/ITESM/navegacion_autonoma/SDC_webots/*.joblib
```

Deben aparecer exactamente estos tres:

- `svm_peatones.joblib` — el clasificador `LinearSVC` entrenado
- `scaler_peatones.joblib` — el `StandardScaler` ajustado al set de entrenamiento
- `hog_params.joblib` — los parámetros HOG (orientations=9, ppc=8x8, cpb=2x2,
  feature_dim=3780)

Si falta alguno, **NO** continúen — avísenle a Alexis.

---

## Paso 4 — Lanzar la simulación (secuencia exacta, IMPORTANTE)

> ⚠️ **GOTCHA crítico:** si dejan una instancia vieja de Webots colgada en el
> sistema, la nueva agarra el puerto **1235** en vez de **1234**, y el
> controlador no se puede conectar. Antes de lanzar, **siempre** maten cualquier
> Webots colgado:
>
> ```bash
> pkill -KILL -x webots 2>/dev/null
> pkill -KILL -f "autonomous_obstacle_controller" 2>/dev/null
> pkill -KILL -f "webots-controller" 2>/dev/null
> sleep 2
> pgrep -af webots                # debe imprimir VACÍO
> lsof -iTCP:1234                 # debe imprimir VACÍO
> ```

Necesitas **dos terminales** abiertas.

### Terminal 1 — abrir Webots R2023b con el mundo de la actividad

```bash
cd ~/ITESM/navegacion_autonoma/SDC_webots
/Applications/Webots-R2023b.app/Contents/MacOS/webots \
  --mode=realtime \
  --port=1234 \
  worlds/city_2025a_activity_3_1.wbt
```

Espera ~10–15 segundos. Verás la ventana de Webots con la ciudad. En la
consola interna de Webots (parte inferior) verás:

```
INFO: 'vehicle' extern controller: Waiting for local or remote connection
on port 1234 targeting robot named 'vehicle'.
ipc://1234/vehicle
```

Cuando aparezca eso, ya está listo para recibir al controlador. **NO presionen
play** — el extern controller arranca la simulación al conectarse.

### Terminal 2 — lanzar el controlador del vehículo

```bash
conda activate webots
export WEBOTS_HOME=/Applications/Webots-R2023b.app
export PYTHONPATH=/Applications/Webots-R2023b.app/Contents/lib/controller/python
export PYTHONUNBUFFERED=1
WPY=$(which python)

/Applications/Webots-R2023b.app/Contents/MacOS/webots-controller \
  --port=1234 \
  --robot-name=vehicle \
  "$WPY" -u ~/ITESM/navegacion_autonoma/SDC_webots/autonomous_obstacle_controller.py
```

**Detalles que importan y nos costaron tiempo**:

- `WEBOTS_HOME` y `PYTHONPATH` apuntan a la app **R2023b**, no a la R2025a si la
  tienen instalada.
- `PYTHONUNBUFFERED=1` y la bandera `-u` de Python son necesarias para que los
  `print()` del controlador aparezcan en consola en tiempo real (sin esto,
  el log parece "congelado" pero el coche sí está corriendo — confusión típica).
- `--port=1234` explícito en AMBOS comandos para evitar que `webots-controller`
  intente puertos alternos si encuentra otra instancia.
- `--robot-name=vehicle` (en minúsculas — es el `name` por defecto del PROTO
  `BmwX5`, no `BMW_X5`).
- La RUTA al script tiene que ser **absoluta** — `webots-controller` cambia su
  `cwd` al directorio del binario de Python, así que un path relativo
  (`autonomous_obstacle_controller.py`) no se encuentra.

En cuanto el controlador se conecte, el BMW empieza a moverse. En la Terminal 2
verás logs como:

```
[INIT] Cámara: 256x128, setpoint=110px (offset -18 px desde el centro)
[INIT] PID gains: Kp=0.01, Ki=0.0, Kd=0.002
[INIT] LiDAR 'Sick LMS 291' habilitado, FOV central = 25.0° de 180.0°, umbral = 20.0 m
[INIT] SVM cargado desde .../SDC_webots (feature_dim=3780, window=(128, 64))
[STATE] -> DRIVING (dist=33.92 m)
[STATE] -> BRAKE_BARREL (dist=5.80 m)
[STATE] -> BRAKE_PEDESTRIAN (dist=18.17 m)
[F50] lines=6 valid=4 err=+0.50 angle=+0.0050 dist=33.68 m state=DRIVING no_line_total=17
...
```

(También verán muchísimas líneas `RuntimeWarning: ... in matmul` de sklearn —
son inocuas, vienen de la rutina interna que usa `decision_function`. Ignorar.)

---

## Paso 5 — Qué grabar (el video, menos de 5 minutos)

El video debe durar **menos de 5 minutos** y **todas las personas del equipo
deben participar** (pueden turnarse narrando). Tiene que mostrar TRES cosas, en
este orden:

### 1. La matriz de confusión del entrenamiento del SVM

Abre el notebook `pedestrian_detection (1).ipynb` (el original, entrenado con
Penn-Fudan) **Y/O** muestra los resultados del modelo final reentrenado con
frames de Webots (`SDC_webots/retrain_svm_hardneg.py`).

**Resultados del modelo final que se entrega** (entrenado con 600 frames del
propio simulador hand-etiquetados, hard-negative mining incluido, set de
prueba 360 muestras estratificadas 80/20):

| | Predicho: No peatón | Predicho: Peatón |
|---|:---:|:---:|
| **Real: No peatón** | 345 | 5 |
| **Real: Peatón** | 7 | 3 |

- **Accuracy: 96.67%**
- **Recall peatón: 30%** (3 de 10 peatones detectados)
- **Precisión peatón: 37.5%** (3 aciertos de 8 predicciones positivas)
- **Recall no-peatón: 98.6%**

Expliquen honestamente: el dataset es pequeño (sólo 48 positivos labeled), por
lo que el clasificador tiene un trade-off entre recall y selectividad runtime.
El modelo final detecta correctamente ~1 de cada 3 peatones que aparecen en el
sector frontal del LiDAR; el resto los clasifica como barriles y el coche
frena con luces intermitentes.

### 2. La operación del LiDAR y su colaboración con el SVM

Muestren el código del controlador (`SDC_webots/autonomous_obstacle_controller.py`)
y expliquen:

- Cómo el LiDAR Sick LMS 291 lee únicamente el sector central de 25° (filtrado
  en software, no en el `.wbt`) y limita la detección a 20 m.
- Cómo, cuando el LiDAR detecta un obstáculo cercano, se dispara el clasificador
  SVM sobre la imagen de la cámara (256x128) con ventana deslizante 64x128 a
  paso 16 px.
- Por qué el SVM solo corre cuando el LiDAR confirma un obstáculo (ahorra ~400
  inferencias HOG por segundo cuando la vía está despejada).

### 3. La simulación en vivo

Graben la ventana de Webots mostrando:

- El vehículo siguiendo la línea amarilla a 50 km/h.
- Un **barril** apareciendo al frente → el coche **frena y enciende
  intermitentes** → el barril desaparece → el coche **reanuda**.
- Un **peatón** cruzando al frente → idealmente el coche **frena sin
  intermitentes** → el peatón pasa → el coche **reanuda**.

**Asegúrense de que se VEAN las luces intermitentes encendiéndose y apagándose
en el BMW durante el evento del barril** — es un punto explícito de la rúbrica.

---

## Paso 6 — Subir el video y entregar

1. Suban el video a **YouTube como "No listado" (Unlisted)** — **NO** privado.
   - "Privado" requiere que el profesor inicie sesión con una cuenta autorizada.
   - "No listado" permite acceso por link sin login.
2. Verifiquen el link abriéndolo en una ventana de incógnito sin sesión. Si
   reproduce, está bien.
3. Peguen el link en el documento `.docx` del reporte (Alexis arma el `.docx`).
4. Entreguen el `.docx` por Canvas con el botón "Entregar Tarea" antes del
   **24 de mayo, 23:59**.

---

## Limitaciones conocidas (sean honestos en el reporte)

1. **Recall del SVM ≈ 30%.** El clasificador detecta ~1 de cada 3 peatones del
   simulador. Con sólo 48 frames de entrenamiento positivos, el espacio de
   features no se cubre lo suficiente para tener mejor recall sin sacrificar
   selectividad runtime (el modelo "anterior" tenía 99% recall pero clasificaba
   TODO como peatón en runtime — inutilizable). En el video pueden mencionar
   que mejorar esto requeriría capturar ~5-10× más frames positivos.

2. **El coche puede salirse del camino en curvas muy cerradas** después de
   varios minutos de simulación, especialmente cuando hay obstáculos
   simultáneos de barril+peatón que disparan transiciones rápidas BARREL ↔
   PEDESTRIAN. El lane-keeper original de la Actividad 2.1 no anticipa curvas;
   en runs largos eventualmente acaba contra una banqueta. El sistema está
   diseñado para grabar segmentos de 1-3 minutos sin problema.

3. **El "engine_speaker" del BMW imprime warnings continuos** en la consola
   de Webots (`Speaker engine_speaker: Impossible to play ...`). Es un bug
   conocido del PROTO `BmwX5` cuando se carga bajo R2023b; no afecta nada,
   simplemente ignórenlo.

4. **El error `Error: only one Robot instance can be created per controller
   process`** aparece en la primera línea del log del controlador. Es una
   advertencia no fatal de Webots por instanciar `Car()` y `Driver()` al mismo
   tiempo (patrón heredado de la Actividad 2.1). El controlador continúa
   funcionando normalmente después de imprimirla.

---

## Problemas conocidos y soluciones rápidas

**El coche no se mueve después de lanzar el controlador**
→ Probablemente Webots no terminó de cargar el mundo, o agarró un puerto
distinto a 1234. Maten todo (`pkill -KILL -x webots`) y reinicien siguiendo el
Paso 4 al pie de la letra, esperando los 10-15 s entre comandos.

**`getDevice` regresa None / el controlador truena al iniciar**
→ Las variables `WEBOTS_HOME` y `PYTHONPATH` no apuntan a R2023b. Verifiquen
con `echo $WEBOTS_HOME` y `echo $PYTHONPATH` — ambas deben tener
`Webots-R2023b.app`, no `Webots.app` (que es R2025a si la tienen instalada).

**Python no encuentra el módulo `controller`**
→ Misma causa anterior: `PYTHONPATH` mal. Repitan los `export`.

**El log parece congelado pero el coche sí se mueve**
→ Falta `PYTHONUNBUFFERED=1` o la bandera `-u`. Maten el controlador, agreguen
los dos, vuelvan a lanzar.

**Webots agarra el puerto 1235 en vez de 1234**
→ Hay una instancia anterior colgada. Maten todo con `pkill -KILL -x webots`,
verifiquen con `lsof -iTCP:1234` (debe estar vacío), y reinicien.

**El coche se cuelga en el frame ~350 (~11 s de simulación)**
→ Están usando R2025a. Usen R2023b. Es un bloqueo determinístico en
`wbu_driver_step()` por incompatibilidad de PROTOs entre versiones.

**El SVM clasifica como peatón TODO lo que ve (incluso barriles y muros)**
→ Tienen el modelo "broken retrain" en disco. Verifiquen el hash o pídanle a
Alexis los joblibs originales. El modelo correcto da una tasa runtime de ~7-28%
PEATÓN, no >90%.

**El coche se sale del camino y se queda contra una banqueta**
→ Limitación conocida (ver arriba). Reinicien el simulador y vuelvan a lanzar;
el comportamiento es determinístico, así que el coche probablemente vuelva a
fallar en el mismo lugar después de unos minutos. Graben segmentos cortos en
los primeros 2-3 minutos cuando todo funciona bien.

---

## Reglas importantes (no romper)

- **NO modifiquen** el controlador del Supervisor
  (`SDC_webots/controllers/supervisor_controller/supervisor_controller.py`) —
  controla los barriles.
- **NO modifiquen** los controladores de los peatones — tienen trayectorias
  definidas.
- **NO modifiquen** el archivo del mundo (`.wbt`) del profesor. Webots R2023b
  intenta auto-guardar viewport state al cerrarlo; si Git les marca el `.wbt`
  como modificado después de cerrar Webots, revírtanlo con
  `git checkout -- SDC_webots/worlds/`.
- **NO ajusten** parámetros del controlador (PID, LiDAR, SVM, setpoint) sin
  avisarle a Alexis — todos están calibrados.
- Si algo truena, manden screenshot del error a Alexis ANTES de cambiar nada.

---

*Cualquier duda, escríbanle a Alexis.*
