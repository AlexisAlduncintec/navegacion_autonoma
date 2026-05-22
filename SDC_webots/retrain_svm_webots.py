"""
================================================================================
Actividad 3.1 - Reentrenamiento del SVM con frames del dominio Webots
MR4010 - Navegación Autónoma | Tec de Monterrey | MNA | Equipo 18
================================================================================

Lee captured_frames/labels.csv y los PNGs correspondientes, extrae parches HOG
con el mismo esquema que usa el controlador en inferencia, reentrena el
LinearSVC + StandardScaler, y sobrescribe svm_peatones.joblib y
scaler_peatones.joblib. hog_params.joblib NO se toca (la dim del feature
vector debe seguir siendo 3780 para que el assert del controlador pase).

Esquema de extracción de parches (acordado con Alexis antes de etiquetar):
    peatón  -> 1 parche centrado gray[0:128, 96:160]          (label=1, positivo)
    barril  -> 1 parche centrado gray[0:128, 96:160]          (label=0, negativo)
    ninguno -> 3 parches random  gray[0:128, x:x+64], x∈[0,192] (label=0, negativo)

Balance: el total de negativos se topa en 4× los positivos para no sesgar
la frontera del SVM hacia "no peatón". Con 48 positivos y 74 barriles fijos
+ random_n_patches, se quedan ~118 patches random adicionales.

Uso:
    conda activate webots
    cd SDC_webots
    python retrain_svm_webots.py
================================================================================
"""

import csv
import os
import random
import sys

import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# --- Reproducibilidad (mismo seed que el notebook original) ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR  = os.path.join(HERE, "captured_frames")
LABELS_CSV  = os.path.join(FRAMES_DIR, "labels.csv")
HOG_PARAMS  = os.path.join(HERE, "hog_params.joblib")
SVM_OUT     = os.path.join(HERE, "svm_peatones.joblib")
SCALER_OUT  = os.path.join(HERE, "scaler_peatones.joblib")

# Dimensiones de la cámara y del parche (fijas, deben coincidir con el controlador)
IMG_W, IMG_H        = 256, 128
PATCH_W, PATCH_H    = 64, 128
PATCH_X_CENTER_LO   = (IMG_W - PATCH_W) // 2   # = 96
PATCH_X_CENTER_HI   = PATCH_X_CENTER_LO + PATCH_W  # = 160
N_RANDOM_NEG_PER_N  = 3
NEG_TO_POS_CAP      = 4   # negatives totales <= 4 * positivos


# =============================================================================
# 1) Cargar etiquetas + parámetros HOG
# =============================================================================
print("=== 1) Cargando etiquetas y parámetros HOG ===")
if not os.path.exists(LABELS_CSV):
    sys.exit(f"FALTA {LABELS_CSV} -- ¿ya etiquetaste?")
labels = {}
with open(LABELS_CSV) as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        if not row:
            continue
        labels[int(row[0])] = row[1]

counts = {"peaton": 0, "barril": 0, "ninguno": 0}
for v in labels.values():
    counts[v] = counts.get(v, 0) + 1
print(f"  peatón:   {counts['peaton']}")
print(f"  barril:   {counts['barril']}")
print(f"  ninguno:  {counts['ninguno']}")
print(f"  TOTAL:    {sum(counts.values())} frames etiquetados")

hog_p = joblib.load(HOG_PARAMS)
print(f"  HOG params cargados: feature_dim esperado = {hog_p['feature_dim']}")


def hog_features(patch_gray):
    """HOG sobre un parche 128x64 (alto x ancho) con los params exactos del entrenamiento original."""
    return hog(
        patch_gray,
        orientations    = hog_p["orientations"],
        pixels_per_cell = hog_p["pixels_per_cell"],
        cells_per_block = hog_p["cells_per_block"],
        block_norm      = hog_p["block_norm"],
        transform_sqrt  = hog_p["transform_sqrt"],
        feature_vector  = True,
    )


# =============================================================================
# 2) Extraer parches según etiqueta
# =============================================================================
print()
print("=== 2) Extracción de parches ===")
X_pos = []   # peatones (positivos)
X_neg_barrel = []
X_neg_random = []

missing = 0
for frame_id, label in sorted(labels.items()):
    png_path = os.path.join(FRAMES_DIR, f"frame_{frame_id:06d}.png")
    img = cv2.imread(png_path)
    if img is None:
        missing += 1
        continue
    # cv2 lee BGR; pasamos a gris una sola vez por imagen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if (h, w) != (IMG_H, IMG_W):
        # Defensa: la cámara debe ser 256x128 — si llega algo distinto, salta.
        print(f"  ! frame_{frame_id:06d} tiene shape ({h},{w}); esperado ({IMG_H},{IMG_W}). Saltando.")
        continue

    if label == "peaton":
        patch = gray[0:PATCH_H, PATCH_X_CENTER_LO:PATCH_X_CENTER_HI]
        X_pos.append(hog_features(patch))
    elif label == "barril":
        patch = gray[0:PATCH_H, PATCH_X_CENTER_LO:PATCH_X_CENTER_HI]
        X_neg_barrel.append(hog_features(patch))
    elif label == "ninguno":
        for _ in range(N_RANDOM_NEG_PER_N):
            x = random.randint(0, IMG_W - PATCH_W)   # inclusive ambos extremos
            patch = gray[0:PATCH_H, x:x + PATCH_W]
            X_neg_random.append(hog_features(patch))
    # 'skip' u otros: ignorados.

if missing:
    print(f"  AVISO: faltaron {missing} PNGs (no se pudieron leer)")
print(f"  positivos (peatón centrado):              {len(X_pos)}")
print(f"  negativos por barril centrado:            {len(X_neg_barrel)}")
print(f"  negativos random de frames 'ninguno':     {len(X_neg_random)}")


# =============================================================================
# 3) Balance: capar negativos a 4× positivos
# =============================================================================
print()
print("=== 3) Balance de clases ===")
n_pos = len(X_pos)
neg_cap = NEG_TO_POS_CAP * n_pos
print(f"  cap de negativos = 4 × positivos = 4 × {n_pos} = {neg_cap}")
print(f"  negativos brutos = {len(X_neg_barrel)} (barril) + {len(X_neg_random)} (random) = "
      f"{len(X_neg_barrel) + len(X_neg_random)}")

# Mantener TODOS los barriles (son la señal más útil contra falsos positivos),
# y submuestrear los random para ajustar el total al cap.
n_random_keep = max(0, neg_cap - len(X_neg_barrel))
if n_random_keep < len(X_neg_random):
    random.shuffle(X_neg_random)  # seed ya fijado
    X_neg_random = X_neg_random[:n_random_keep]
    print(f"  submuestreados random negativos a {n_random_keep}")
else:
    print(f"  no se subsamplea (random_keep={n_random_keep} >= len(X_neg_random))")

X_neg = X_neg_barrel + X_neg_random
print(f"  negativos finales = {len(X_neg)}  (positivos={n_pos}, "
      f"ratio neg:pos = {len(X_neg)/n_pos:.2f}x)")


# =============================================================================
# 4) Armar X, y; verificar dimensión HOG
# =============================================================================
print()
print("=== 4) Construyendo X, y ===")
X = np.array(X_pos + X_neg, dtype=np.float32)
y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))]).astype(np.int64)
print(f"  X.shape = {X.shape}  y.shape = {y.shape}")
assert X.shape[1] == hog_p["feature_dim"], (
    f"Dim HOG ({X.shape[1]}) no coincide con hog_params.joblib ({hog_p['feature_dim']}). "
    "Esto rompería el assert del controlador. Aborting.")
print(f"  ✓ feature_dim coincide con hog_params.joblib ({hog_p['feature_dim']})")


# =============================================================================
# 5) Train/test split + entrenamiento
# =============================================================================
print()
print("=== 5) Train/test split + entrenamiento ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y,
)
print(f"  train: {X_train.shape[0]} samples  ({int(y_train.sum())} pos / {int((y_train==0).sum())} neg)")
print(f"  test:  {X_test.shape[0]} samples  ({int(y_test.sum())} pos / {int((y_test==0).sum())} neg)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

clf = LinearSVC(C=1.0, max_iter=10000, random_state=RANDOM_SEED)
clf.fit(X_train_scaled, y_train)
print("  ✓ LinearSVC entrenado")


# =============================================================================
# 6) Evaluación
# =============================================================================
print()
print("=== 6) Evaluación en el set de prueba ===")
y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred, labels=[0, 1])  # filas reales, columnas predichas

print(f"  Accuracy: {acc*100:.2f}%")
print()
print("  Matriz de confusión:")
print(f"                     pred no-peatón   pred peatón")
print(f"  real no-peatón     {cm[0,0]:>10}      {cm[0,1]:>10}")
print(f"  real peatón        {cm[1,0]:>10}      {cm[1,1]:>10}")
print()
# Per-class metrics manualmente para llamar atención al RECALL de peatón.
TN, FP = cm[0, 0], cm[0, 1]
FN, TP = cm[1, 0], cm[1, 1]
recall_peaton  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
precision_peaton = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall_noped   = TN / (TN + FP) if (TN + FP) > 0 else 0.0
print(f"  >>> RECALL PEATÓN  = {recall_peaton*100:.1f}%  "
      f"({TP} de {TP+FN} peatones detectados)")
print(f"  >>> PRECISIÓN PEATÓN = {precision_peaton*100:.1f}%  "
      f"({TP} aciertos de {TP+FP} predicciones positivas)")
print(f"  >>> RECALL NO-PEATÓN = {recall_noped*100:.1f}%")
print()
print("  classification_report:")
print(classification_report(y_test, y_pred,
                            target_names=["no-peaton", "peaton"],
                            digits=3))


# =============================================================================
# 7) Sobrescribir artefactos
# =============================================================================
print()
print("=== 7) Guardando modelos ===")
joblib.dump(clf,    SVM_OUT)
joblib.dump(scaler, SCALER_OUT)
print(f"  ✓ sobrescrito {SVM_OUT}")
print(f"  ✓ sobrescrito {SCALER_OUT}")
print(f"  (hog_params.joblib NO se tocó — feature_dim={hog_p['feature_dim']} preservado)")
print()
print("Listo. El controlador NO necesita cambios — recarga estos joblib al arrancar.")
