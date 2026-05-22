"""
================================================================================
Actividad 3.1 - Reentrenamiento del SVM con HARD-NEGATIVE MINING
MR4010 - Navegación Autónoma | Tec de Monterrey | MNA | Equipo 18
================================================================================

Estrategia Dalal-Triggs estándar:
    1. Usar el SVM actual (sobreentrenado, 99.94% PEATÓN en runtime) para
       escanear los frames que SÉ que NO contienen peatón (ninguno + barril).
    2. Cada ventana que clasifica como PEATÓN en esos frames es un FALSO
       POSITIVO — es decir, una "hard negative" por definición.
    3. Ordenar por decision_function descendente (las MÁS difíciles primero)
       y tomar el top-K como negativos adicionales.
    4. Reentrenar LinearSVC con (positivos + barril center + random n + hard
       negs). El nuevo conjunto de negativos ahora cubre el espacio donde
       el modelo anterior fallaba.

NO se tocan hyperparámetros del SVM, HOG, ni la lógica del controlador.
Sólo se mejora el conjunto de datos de entrenamiento.

Uso:
    conda activate webots
    cd SDC_webots
    python retrain_svm_hardneg.py
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

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

HERE        = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR  = os.path.join(HERE, "captured_frames")
LABELS_CSV  = os.path.join(FRAMES_DIR, "labels.csv")
HOG_PARAMS  = os.path.join(HERE, "hog_params.joblib")
SVM_OUT     = os.path.join(HERE, "svm_peatones.joblib")
SCALER_OUT  = os.path.join(HERE, "scaler_peatones.joblib")
CURRENT_SVM_PATH    = os.path.join(HERE, "svm_peatones.joblib")
CURRENT_SCALER_PATH = os.path.join(HERE, "scaler_peatones.joblib")

# Parámetros idénticos al controlador en inferencia
IMG_W, IMG_H        = 256, 128
PATCH_W, PATCH_H    = 64, 128
PATCH_X_CENTER_LO   = (IMG_W - PATCH_W) // 2       # = 96
PATCH_X_CENTER_HI   = PATCH_X_CENTER_LO + PATCH_W  # = 160
SVM_STRIDE_X        = 16
N_RANDOM_NEG_PER_N  = 3

# Mining params
HARDNEG_TOP_K_PER_FRAME = 3      # máximo 3 hard-negs por frame antes del cap global
HARDNEG_GLOBAL_CAP      = 240    # 5× positivos -> total neg ~ 9× positivos

# Balance del pool de random negatives.
# Tras experimentar con varios ratios:
#   - 118 random (4× positivos)       => 99.6% PEATÓN runtime, recall 90%  (overfit)
#   - 1434 random (todos los disponibles) => 28% PEATÓN runtime, recall 30%  (sano)
# La regla empírica con sólo 48 positivos: necesitamos un pool de
# negativos amplio para evitar overfit; los hard-negs por sí solos no
# alcanzan a contener el hyperplano. Final: usamos TODOS los 1434.
RANDOM_NEG_CAP          = 1434   # mantener todo el pool random


# =============================================================================
# 1) Cargar etiquetas + HOG params + SVM ACTUAL (para mining)
# =============================================================================
print("=== 1) Cargando inputs ===")
with open(LABELS_CSV) as f:
    reader = csv.reader(f)
    next(reader, None)
    labels = {int(r[0]): r[1] for r in reader if r}
print(f"  labels: {len(labels)} frames")

hog_p = joblib.load(HOG_PARAMS)
print(f"  HOG params: feature_dim esperado = {hog_p['feature_dim']}")

current_clf    = joblib.load(CURRENT_SVM_PATH)
current_scaler = joblib.load(CURRENT_SCALER_PATH)
print(f"  SVM actual (overfit): {type(current_clf).__name__} con "
      f"{current_clf.coef_.shape[1]} features")


def hog_features(patch_gray):
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
# 2) Extraer positivos + negativos base (mismo esquema que antes)
# =============================================================================
print()
print("=== 2) Extrayendo parches base (misma lógica que retrain inicial) ===")
X_pos = []
X_neg_barrel = []
X_neg_random = []

# Importante: cargar los frames UNA SOLA VEZ y mantenerlos en RAM para que
# las dos pasadas (extracción base + hard-neg mining) sean rápidas.
frames_in_memory = {}  # frame_id -> gray (128x256)

for frame_id, label in sorted(labels.items()):
    if label not in ("peaton", "barril", "ninguno"):
        continue
    png_path = os.path.join(FRAMES_DIR, f"frame_{frame_id:06d}.png")
    img = cv2.imread(png_path)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.shape != (IMG_H, IMG_W):
        continue
    frames_in_memory[frame_id] = gray

    if label == "peaton":
        X_pos.append(hog_features(gray[0:PATCH_H, PATCH_X_CENTER_LO:PATCH_X_CENTER_HI]))
    elif label == "barril":
        X_neg_barrel.append(hog_features(gray[0:PATCH_H, PATCH_X_CENTER_LO:PATCH_X_CENTER_HI]))
    elif label == "ninguno":
        for _ in range(N_RANDOM_NEG_PER_N):
            x = random.randint(0, IMG_W - PATCH_W)
            X_neg_random.append(hog_features(gray[0:PATCH_H, x:x + PATCH_W]))

print(f"  positivos:              {len(X_pos)}")
print(f"  barril center neg:      {len(X_neg_barrel)}")
print(f"  random neg pool:        {len(X_neg_random)}  (antes de subsampling)")

# Subsamplear random negatives al cap 4×positivos-barril, ANTES de añadir
# hard-negs. Esto evita que 1434 random fáciles dominen el hyperplano.
if len(X_neg_random) > RANDOM_NEG_CAP:
    random.shuffle(X_neg_random)   # seed=42 fija ya seteado
    X_neg_random = X_neg_random[:RANDOM_NEG_CAP]
print(f"  random neg (subsampled): {len(X_neg_random)}  (cap = {RANDOM_NEG_CAP})")


# =============================================================================
# 3) HARD-NEGATIVE MINING:
#    Aplicar SVM actual a todos los frames de ninguno+barril, recolectar
#    los parches que clasifica como PEATÓN (esos son falsos positivos por
#    definición porque sé que esos frames no tienen peatón centrado).
# =============================================================================
print()
print("=== 3) Hard-negative mining ===")
print("  Escaneando ventanas deslizantes (stride=16) en cada frame "
      "ninguno/barril...")

hard_candidates = []  # lista de (score, hog_vector)
n_frames_scanned = 0
n_windows_total  = 0
n_windows_positive = 0

# Para cada frame sin peatón, scoreamos las 13 ventanas. Mantenemos top-K
# por frame ANTES de competir globalmente para no sobre-representar un
# solo frame con muchas ventanas idénticamente confundidas.
for frame_id, label in sorted(labels.items()):
    if label not in ("ninguno", "barril"):
        continue
    if frame_id not in frames_in_memory:
        continue
    gray = frames_in_memory[frame_id]
    n_frames_scanned += 1

    frame_candidates = []   # (score, hog_vec) — sólo de este frame
    for x in range(0, IMG_W - PATCH_W + 1, SVM_STRIDE_X):
        patch = gray[0:PATCH_H, x:x + PATCH_W]
        feats = hog_features(patch)
        feats_scaled = current_scaler.transform(feats.reshape(1, -1))
        score = float(current_clf.decision_function(feats_scaled)[0])
        n_windows_total += 1
        if score > 0:  # SVM actual dice PEATÓN -> falso positivo
            n_windows_positive += 1
            frame_candidates.append((score, feats))
    # Quedarnos con los K más confundidos de este frame
    frame_candidates.sort(key=lambda t: t[0], reverse=True)
    hard_candidates.extend(frame_candidates[:HARDNEG_TOP_K_PER_FRAME])

print(f"  frames escaneados:         {n_frames_scanned}")
print(f"  ventanas totales:          {n_windows_total}")
print(f"  ventanas positivas (FP):   {n_windows_positive}  "
      f"({100.0*n_windows_positive/max(1,n_windows_total):.1f}%)")
print(f"  hard candidates retenidos: {len(hard_candidates)}  "
      f"(top-{HARDNEG_TOP_K_PER_FRAME} por frame)")

# Cap global: ordenar por score y tomar los top-N
hard_candidates.sort(key=lambda t: t[0], reverse=True)
hard_candidates = hard_candidates[:HARDNEG_GLOBAL_CAP]
X_neg_hard = [feats for _, feats in hard_candidates]
print(f"  cap global aplicado:       {len(X_neg_hard)} hard-negs")
if hard_candidates:
    print(f"  rango de scores:           [{hard_candidates[-1][0]:.3f}, "
          f"{hard_candidates[0][0]:.3f}]  (mayor = más difícil)")


# =============================================================================
# 4) Construir X, y. Total negativos = barril + random + hard.
# =============================================================================
print()
print("=== 4) Conjunto final ===")
all_negs = X_neg_barrel + X_neg_random + X_neg_hard
X = np.array(X_pos + all_negs, dtype=np.float32)
y = np.concatenate([np.ones(len(X_pos)),
                    np.zeros(len(all_negs))]).astype(np.int64)
print(f"  positivos:             {len(X_pos)}")
print(f"  negativos totales:     {len(all_negs)}  "
      f"= {len(X_neg_barrel)} barril + {len(X_neg_random)} random + "
      f"{len(X_neg_hard)} hard")
print(f"  ratio neg:pos:         {len(all_negs)/max(1,len(X_pos)):.2f}x")
print(f"  X.shape={X.shape}, y.shape={y.shape}")
assert X.shape[1] == hog_p["feature_dim"], "Dim mismatch"
print(f"  ✓ feature_dim={hog_p['feature_dim']} preservado")


# =============================================================================
# 5) Train/test split + entrenamiento
# =============================================================================
print()
print("=== 5) Train/test split + entrenamiento ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
print(f"  train: {X_train.shape[0]} ({int(y_train.sum())} pos / {int((y_train==0).sum())} neg)")
print(f"  test:  {X_test.shape[0]} ({int(y_test.sum())} pos / {int((y_test==0).sum())} neg)")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

clf = LinearSVC(C=1.0, max_iter=10000, random_state=RANDOM_SEED)
clf.fit(X_train_s, y_train)
print("  ✓ LinearSVC entrenado")


# =============================================================================
# 6) Evaluación
# =============================================================================
print()
print("=== 6) Evaluación (set de prueba) ===")
y_pred = clf.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred, labels=[0, 1])

print(f"  Accuracy: {acc*100:.2f}%")
print()
print(f"                     pred no-peatón   pred peatón")
print(f"  real no-peatón     {cm[0,0]:>10}      {cm[0,1]:>10}")
print(f"  real peatón        {cm[1,0]:>10}      {cm[1,1]:>10}")
print()
TN, FP = cm[0,0], cm[0,1]
FN, TP = cm[1,0], cm[1,1]
recall_p = TP / max(1, TP + FN)
prec_p   = TP / max(1, TP + FP)
print(f"  >>> Recall peatón:    {recall_p*100:.1f}%  ({TP}/{TP+FN})")
print(f"  >>> Precisión peatón: {prec_p*100:.1f}%  ({TP}/{TP+FP})")
print(f"  >>> Recall no-peatón: {TN/max(1,TN+FP)*100:.1f}%  ({TN}/{TN+FP})")
print()
print(classification_report(y_test, y_pred,
                            target_names=["no-peaton", "peaton"], digits=3))


# =============================================================================
# 7) Guardar (sobrescribe el SVM/scaler; hog_params no se toca)
# =============================================================================
print()
print("=== 7) Guardando modelo ===")
joblib.dump(clf,    SVM_OUT)
joblib.dump(scaler, SCALER_OUT)
print(f"  ✓ {SVM_OUT}")
print(f"  ✓ {SCALER_OUT}")
print()
print("Listo. Próximo paso: correr el controlador brevemente y medir la "
      "tasa runtime PEATÓN vs no-peatón.")
