import cv2
import numpy as np
import onnxruntime as ort
import os
from utils import preprocess_arcface
from session_options import get_optimized_session

# ==========================
#       MODELOS
# ==========================

YOLO_MODEL = "modelo_det_face/model.onnx"
ARC_MODEL  = "modelo_arcface/arcface_r100.onnx"

session_yolo = get_optimized_session(YOLO_MODEL)
input_name_yolo = session_yolo.get_inputs()[0].name

session_arc = get_optimized_session(ARC_MODEL)
input_name_arc = session_arc.get_inputs()[0].name

# ==========================
#  CARGAR BASE DE USUARIOS
# ==========================

def cargar_base():
    base = {}
    for archivo in os.listdir("base_rostros"):
        if archivo.endswith(".npy"):
            nombre = archivo.replace(".npy", "")
            base[nombre] = np.load(f"base_rostros/{archivo}")
    return base

base_usuarios = cargar_base()
print(f"✔ Usuarios cargados: {list(base_usuarios.keys())}")

def distancia_coseno(a, b):
    dot = np.dot(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    return 1 - (dot / norms)

# ==========================
#   DETECTAR CÁMARA
# ==========================

def abrir_camara():
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if cap.isOpened():
            print(f"✔ Cámara encontrada en índice {i}")
            return cap
    print("❌ No se encontró cámara.")
    exit()

cap = abrir_camara()
print("🎥 Iniciando reconocimiento...")

# ==========================
#  CONTROL DE PARPADEO
# ==========================

last_box = None
last_label = None
last_color = (0, 255, 0)
last_access = None  # PERMITIDO / DENEGADO

frames_sin_det = 0
MAX_FRAMES_SIN_DET = 6

frame_count = 0

# ==========================
#   LOOP PRINCIPAL
# ==========================

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    H, W = frame.shape[:2]

    # ⚡ SOLO INFERIR YOLO CADA 2 FRAMES
    frame_count += 1
    if frame_count % 2 != 0:
        if last_box is not None and last_label is not None:
            x1, y1, x2, y2 = last_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), last_color, 2)
            cv2.putText(frame, last_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, last_color, 2)

            # Mostrar estado de acceso
            if last_access is not None:
                cv2.putText(frame, last_access, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, last_color, 3)

        cv2.imshow("Reconocimiento Facial", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # ==========================
    #       PREPROCESO YOLO
    # ==========================

    resized = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.moveaxis(img, -1, 0)
    img = img[np.newaxis, :, :, :]

    out = session_yolo.run(None, {input_name_yolo: img})[0]
    out = out.squeeze()

    xs, ys, ws, hs, confs = out

    scale_x = W / 640
    scale_y = H / 640

    best_conf = 0
    best_box = None

    for i in range(8400):
        conf = confs[i]
        if conf > best_conf and conf > 0.55:
            cx = xs[i] * scale_x
            cy = ys[i] * scale_y
            w_box = ws[i] * scale_x
            h_box = hs[i] * scale_y

            x1 = int(cx - w_box/2)
            y1 = int(cy - h_box/2)
            x2 = int(cx + w_box/2)
            y2 = int(cy + h_box/2)

            if w_box < 50 or h_box < 50:
                continue
            if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                continue

            best_box = (x1, y1, x2, y2)
            best_conf = conf

    # ==========================
    #      ANTI-PARPADEO
    # ==========================

    if best_box is not None:
        last_box = best_box
        frames_sin_det = 0
    else:
        frames_sin_det += 1
        if frames_sin_det <= MAX_FRAMES_SIN_DET and last_box is not None:
            best_box = last_box
        else:
            best_box = None
            last_box = None
            last_label = None
            last_access = None

    # ==========================
    #         RECONOCER
    # ==========================

    if best_box is not None:
        x1, y1, x2, y2 = best_box

        face = frame[y1:y2, x1:x2]

        face_pre = preprocess_arcface(face)
        embedding_live = session_arc.run(None, {input_name_arc: face_pre})[0][0]

        mejor_usuario = "DESCONOCIDO"
        mejor_distancia = 1e9
        tiene_acceso = False

        for usuario, datos in base_usuarios.items():
            emb_base = datos['embedding']
            dist = distancia_coseno(embedding_live, emb_base)
            if dist < mejor_distancia:
                mejor_distancia = dist
                mejor_usuario = usuario
                tiene_acceso = datos['access']

        # ------ REGLA DE ACCESO ------
        if mejor_distancia < 0.55:
            label = f"{mejor_usuario} ({mejor_distancia:.3f})"
            if tiene_acceso:
                color = (0, 255, 0)
                access = "ACCESO PERMITIDO"
            else:
                color = (0, 165, 255)  # Naranja
                access = "ACCESO DENEGADO"
        else:
            label = f"DESCONOCIDO ({mejor_distancia:.3f})"
            color = (0, 0, 255)
            access = "ACCESO DENEGADO"

        # guardar para smoothing
        last_label = label
        last_color = color
        last_access = access

    # ==========================
    #    DIBUJAR EN PANTALLA
    # ==========================

    if last_box is not None and last_label is not None:
        x1, y1, x2, y2 = last_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), last_color, 2)
        cv2.putText(frame, last_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, last_color, 2)

        # Mensaje de acceso
        cv2.putText(frame, last_access, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, last_color, 3)

    cv2.imshow("Reconocimiento Facial", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
