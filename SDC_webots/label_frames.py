"""
================================================================================
Actividad 3.1 - Helper de etiquetado para reentrenamiento del SVM
MR4010 - Navegación Autónoma | Tec de Monterrey | MNA | Equipo 18
================================================================================

Propósito:
    Etiquetar manualmente los frames capturados por capture_controller.py
    para reentrenar el SVM con datos del dominio correcto (Webots renderizado,
    no Penn-Fudan real).

Uso:
    cd SDC_webots
    python label_frames.py

    Para cada frame muestra una ventana de OpenCV con:
        - la imagen 256x128 ampliada 3x
        - el número de frame, sim_time, distancia LiDAR del momento de captura

    Teclas:
        p   = peatón (hay al menos un peatón claramente visible delante)
        b   = barril (hay un barril visible delante; pueden coexistir peatones,
              pero el OBSTÁCULO PRINCIPAL es un barril)
        n   = ninguno (vía despejada o solo edificios/árboles al fondo,
              ningún peatón ni barril relevante)
        s   = saltar (no estoy seguro / mala calidad / esperar para repensar)
        u   = deshacer última etiqueta
        q   = salir (guarda progreso)

    El progreso se guarda en captured_frames/labels.csv después de CADA tecla.
    Se puede salir y retomar después; sólo se mostrarán los frames sin etiqueta.

Salida:
    captured_frames/labels.csv  con columnas: frame_id, label

Después de etiquetar suficientes frames:
    - ~200 con peatón, ~200 con barril, ~200 ninguno es un buen objetivo.
    - Avísame a mí (Alexis/Claude) y yo escribo el script de reentrenamiento.

================================================================================
"""
import csv
import os
import sys

import cv2
import numpy as np

CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "captured_frames")
METADATA_CSV = os.path.join(CAPTURE_DIR, "metadata.csv")
LABELS_CSV   = os.path.join(CAPTURE_DIR, "labels.csv")
SCALE = 3            # ampliación visual

VALID_KEYS = {
    ord('p'): "peaton",
    ord('b'): "barril",
    ord('n'): "ninguno",
    ord('s'): "skip",
}


def load_existing_labels():
    """Devuelve un dict {frame_id (int): label (str)} con lo ya etiquetado."""
    if not os.path.exists(LABELS_CSV):
        return {}
    out = {}
    with open(LABELS_CSV) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            out[int(row[0])] = row[1]
    return out


def save_labels(labels_dict):
    """Reescribe labels.csv con el dict en memoria. Atómico-ish."""
    tmp = LABELS_CSV + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "label"])
        for fid in sorted(labels_dict.keys()):
            w.writerow([fid, labels_dict[fid]])
    os.replace(tmp, LABELS_CSV)


def load_metadata():
    """Devuelve lista de dicts {frame_id, sim_time_s, forward_dist_m}."""
    if not os.path.exists(METADATA_CSV):
        sys.exit(f"ERROR: no encuentro {METADATA_CSV}. "
                 "¿Corriste capture_controller.py primero?")
    rows = []
    with open(METADATA_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def show_frame(frame_path, info_text):
    """Muestra el frame ampliado con overlay de información."""
    img = cv2.imread(frame_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    big = cv2.resize(img, (w * SCALE, h * SCALE),
                     interpolation=cv2.INTER_NEAREST)
    # Banner negro arriba con texto.
    banner_h = 60
    banner = np.zeros((banner_h, big.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, info_text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(banner,
                "p=peaton  b=barril  n=ninguno  s=skip  u=undo  q=quit",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (200, 200, 200), 1)
    combined = np.vstack([banner, big])
    cv2.imshow("label_frames", combined)
    return combined


def main():
    metadata = load_metadata()
    labels = load_existing_labels()

    print(f"Frames totales:    {len(metadata)}")
    print(f"Ya etiquetados:    {len(labels)}")
    print(f"Pendientes:        {len(metadata) - len(labels)}")
    print()
    print("Abriendo ventana de etiquetado. Foco en la ventana, luego teclas:")
    print("  p=peaton  b=barril  n=ninguno  s=skip  u=undo_last  q=quit")
    print()

    cv2.namedWindow("label_frames", cv2.WINDOW_AUTOSIZE)

    pending = [m for m in metadata if int(m["frame_id"]) not in labels]
    last_labeled = []  # stack para undo

    i = 0
    while i < len(pending):
        m = pending[i]
        fid = int(m["frame_id"])
        frame_path = os.path.join(CAPTURE_DIR, f"frame_{fid:06d}.png")
        info = (f"frame {fid:04d}/{int(metadata[-1]['frame_id']):04d}  "
                f"sim_t={m['sim_time_s']}s  "
                f"LiDAR_dist={m['forward_dist_m']}m  "
                f"pendientes={len(pending) - i}")
        result = show_frame(frame_path, info)
        if result is None:
            print(f"  (no se pudo cargar {frame_path}, saltando)")
            i += 1
            continue

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("\n[etiquetado] saliendo, progreso guardado.")
            break
        if key == ord('u'):
            if last_labeled:
                undo_fid = last_labeled.pop()
                if undo_fid in labels:
                    del labels[undo_fid]
                    save_labels(labels)
                    print(f"  undo: frame_{undo_fid:06d} desetiquetado")
                    # buscar el índice anterior en pending
                    for j in range(i, -1, -1):
                        if int(pending[j]["frame_id"]) == undo_fid:
                            i = j
                            break
            else:
                print("  (nada que deshacer)")
            continue
        if key in VALID_KEYS:
            label = VALID_KEYS[key]
            if label != "skip":
                labels[fid] = label
                last_labeled.append(fid)
                save_labels(labels)
                print(f"  frame_{fid:06d} -> {label}")
            else:
                print(f"  frame_{fid:06d} skipped")
            i += 1
        else:
            print(f"  tecla inválida ({chr(key) if 32 <= key < 127 else key}), "
                  "usa p/b/n/s/u/q")

    cv2.destroyAllWindows()

    # Resumen
    counts = {"peaton": 0, "barril": 0, "ninguno": 0}
    for v in labels.values():
        counts[v] = counts.get(v, 0) + 1
    print()
    print(f"=== Resumen ===")
    print(f"  peatón:   {counts['peaton']}")
    print(f"  barril:   {counts['barril']}")
    print(f"  ninguno:  {counts['ninguno']}")
    print(f"  TOTAL:    {sum(counts.values())} etiquetados de {len(metadata)} capturados")
    print(f"  archivo:  {LABELS_CSV}")


if __name__ == "__main__":
    main()
