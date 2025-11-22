import cv2
import numpy as np
import onnxruntime as ort
import os
from utils import preprocess_arcface

# Rutas de los modelos
YOLO_MODEL = "modelo_det_face/model.onnx"
ARC_MODEL  = "modelo_arcface/arcface_r100.onnx"

# Cargar YOLO FACE
session_yolo = ort.InferenceSession(YOLO_MODEL, providers=["CPUExecutionProvider"])
input_name_yolo = session_yolo.get_inputs()[0].name

# Cargar ARC FACE
session_arc = ort.InferenceSession(ARC_MODEL, providers=["CPUExecutionProvider"])
input_name_arc = session_arc.get_inputs()[0].name

# Cargar base de usuarios
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
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Buscar cámara disponible
def abrir_camara():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✔ Cámara encontrada en índice {i}")
            return cap
    print("❌ No hay cámaras disponibles.")
    exit()

cap = abrir_camara()
print("🎥 Iniciando reconocimiento...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    H, W = frame.shape[:2]

    # ----- PREPROCESO YOLO -----
    resized = cv2.resize(frame, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    out = session_yolo.run(None, {input_name_yolo: img})[0]  # (1,5,8400)
    out = out.squeeze()  # (5,8400)

    xs, ys, ws, hs, confs = out

    scale_x = W / 640
    scale_y = H / 640

    best_conf = 0
    best_box = None

    # --- Seleccionar mejor detección ---
    for i in range(8400):
        conf = confs[i]
        if conf > best_conf and conf > 0.55:
            cx = xs[i] * scale_x
            cy = ys[i] * scale_y
            w_box = ws[i] * scale_x
            h_box = hs[i] * scale_y

            x1 = int(cx - w_box / 2)
            y1 = int(cy - h_box / 2)
            x2 = int(cx + w_box / 2)
            y2 = int(cy + h_box / 2)

            if w_box < 50 or h_box < 50:
                continue
            if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                continue

            best_conf = conf
            best_box = (x1, y1, x2, y2)

    if best_box is not None:
        x1, y1, x2, y2 = best_box

        face = frame[y1:y2, x1:x2]

        # ------- EMBEDDING ARC FACE -------
        face_pre = preprocess_arcface(face)
        embedding_live = session_arc.run(None, {input_name_arc: face_pre})[0][0]

        # ------- COMPARACIÓN -------
        mejor_usuario = "DESCONOCIDO"
        mejor_distancia = 1e9

        for usuario, emb_base in base_usuarios.items():
            dist = distancia_coseno(embedding_live, emb_base)

            if dist < mejor_distancia:
                mejor_distancia = dist
                mejor_usuario = usuario

        # ------ UMBRAL (ajustado para ArcFace R100) -------
        if mejor_distancia < 0.55:
            label = f"{mejor_usuario} ({mejor_distancia:.3f})"
            color = (0, 255, 0)
        else:
            label = f"DESCONOCIDO ({mejor_distancia:.3f})"
            color = (0, 0, 255)

        # Dibujar cuadro
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Reconocimiento Facial", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
