import numpy as np
from sqlalchemy.orm import Session
from core.models import Usuario

def guardar_usuario(db: Session, name: str, embedding: np.ndarray):
    usuario = Usuario(
        name=name,
        embedding=embedding.tobytes()  # convertir ndarray → binario
    )
    db.add(usuario)
    db.commit()
    db.refresh(usuario)
    return usuario


def obtener_usuarios(db: Session):
    usuarios = db.query(Usuario).all()

    result = {}
    for u in usuarios:
        emb = np.frombuffer(u.embedding, dtype=np.float32)
        result[u.name] = {
            'embedding': emb,
            'access': u.access
        }

    return result