import cv2
import numpy as np
import onnxruntime as ort

YOLO_MODEL = "../modelo_det_face/model.onnx"

session = ort.InferenceSession(YOLO_MODEL, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)
print("Probando detección...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    H, W = frame.shape[:2]

    # ---- Preprocesamiento ----
    resized = cv2.resize(frame, (640, 640))   # <- modelo trabaja en 640x640
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    # ---- Inferencia ----
    out = session.run(None, {input_name: img})[0]  # (1,5,8400)
    out = out.squeeze()  # (5,8400)

    xs = out[0]  # pixel real 0–640
    ys = out[1]
    ws = out[2]
    hs = out[3]
    confs = out[4]

    for i in range(8400):
        conf = confs[i]
        if conf < 0.55:   # tu modelo tiene confidencias muy bajas
            continue

        cx = xs[i]
        cy = ys[i]
        w_box = ws[i]
        h_box = hs[i]

        # convertir de espacio 640x640 a tamaño real W,H
        scale_x = W / 640
        scale_y = H / 640

        cx *= scale_x
        cy *= scale_y
        w_box *= scale_x
        h_box *= scale_y

        x1 = int(cx - w_box / 2)
        y1 = int(cy - h_box / 2)
        x2 = int(cx + w_box / 2)
        y2 = int(cy + h_box / 2)

        # filtrar cajas inválidas
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            continue
        if w_box < 10 or h_box < 10:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("TEST YOLO FACE (Ajustado)", frame)

    if cv2.waitKey(1) == 27:  # Esc
        break

cap.release()
cv2.destroyAllWindows()
