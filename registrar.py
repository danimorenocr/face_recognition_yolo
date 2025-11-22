import cv2
import numpy as np
import onnxruntime as ort
import os
from utils import preprocess_arcface

# Rutas de los modelos
YOLO_MODEL = "modelo_det_face/model.onnx"
ARC_MODEL  = "modelo_arcface/arcface_r100.onnx"

# Cargar YOLO FACE (tu modelo)
session_yolo = ort.InferenceSession(YOLO_MODEL, providers=["CPUExecutionProvider"])
input_name_yolo = session_yolo.get_inputs()[0].name

# Cargar ARC FACE
session_arc = ort.InferenceSession(ARC_MODEL, providers=["CPUExecutionProvider"])
input_name_arc = session_arc.get_inputs()[0].name

# Crear carpeta de usuarios si no existe
os.makedirs("base_rostros", exist_ok=True)

nombre = input("Ingresa el nombre del usuario: ")

# Buscar cámara disponible
def abrir_camara():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✔ Cámara encontrada en índice {i}")
            return cap
    print("❌ No se encontró una cámara disponible.")
    exit()

cap = abrir_camara()
print("🎥 Cámara iniciada. Acércate bien a la cámara.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    H, W = frame.shape[:2]

    # --- PREPROCESAR A 640x640 ---
    resized = cv2.resize(frame, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    # --- INFERENCIA YOLO ---
    outputs = session_yolo.run(None, {input_name_yolo: img})[0]  # (1,5,8400)
    outputs = outputs.squeeze()  # (5,8400)

    xs = outputs[0]  # centros x EN PIXELES DEL 640x640
    ys = outputs[1]  # centros y
    ws = outputs[2]
    hs = outputs[3]
    confs = outputs[4]

    # ESCALA a tamaño real
    scale_x = W / 640
    scale_y = H / 640

    best_conf = 0
    best_box = None

    # --- TOMAR SOLO LA DETECCIÓN CON MAYOR CONF ---
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

            # Validación de caja
            if w_box < 50 or h_box < 50:
                continue
            if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                continue

            best_conf = conf
            best_box = (x1, y1, x2, y2)

    # --- SI HAY ROSTRO VÁLIDO ---
    if best_box is not None:
        x1, y1, x2, y2 = best_box

        # Dibujar cuadro
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{best_conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Extraer rostro
        face = frame[y1:y2, x1:x2]

        # --- ARC FACE EMBEDDING ---
        face_pre = preprocess_arcface(face)
        embedding = session_arc.run(None, {input_name_arc: face_pre})[0][0]

        # Guardar embedding
        path = f"base_rostros/{nombre}.npy"
        np.save(path, embedding)

        print(f"✔ Rostro registrado correctamente: {path}")

        cap.release()
        cv2.destroyAllWindows()
        exit()

    # Mostrar cámara mientras intenta detectar
    cv2.imshow("Registrar rostro", frame)

    # Salir con ESC
    if cv2.waitKey(1) == 27:
        print("❌ Registro cancelado")
        break

cap.release()
cv2.destroyAllWindows()
