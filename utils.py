import cv2
import numpy as np

def preprocess_arcface(face):
    # 112x112 (input oficial)
    face = cv2.resize(face, (112, 112))

    # BGR -> RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face = face.astype(np.float32)

    # Normalización típica de ArcFace
    face = (face - 127.5) / 128.0

    # NOTA IMPORTANTE:
    # Tu modelo NO usa formato CHW, usa HWC.
    # Entonces el shape final debe ser: (1,112,112,3)

    face = np.expand_dims(face, axis=0)

    return face

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def distance(a, b):
    return np.linalg.norm(a - b)
