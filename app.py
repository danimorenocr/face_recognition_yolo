import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, session, redirect, url_for, request
from functools import wraps
from utils import preprocess_arcface
from session_options import get_optimized_session
from core.database import SessionLocal
from services.face_recognizer import obtener_usuarios

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Clave secreta para sesiones

# ==========================
#       MODELOS
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL = os.path.join(BASE_DIR, "models", "model.onnx")
ARC_MODEL = os.path.join(BASE_DIR, "models", "arcface_r100.onnx")

session_yolo = get_optimized_session(YOLO_MODEL)
input_name_yolo = session_yolo.get_inputs()[0].name

session_arc = get_optimized_session(ARC_MODEL)
input_name_arc = session_arc.get_inputs()[0].name

# ==========================
#  CARGAR BASE DE USUARIOS
# ==========================

def cargar_base():
    db = SessionLocal()
    base_usuarios = obtener_usuarios(db)
    db.close()
    return base_usuarios

base_usuarios = cargar_base()
print(f"✔ Usuarios cargados: {list(base_usuarios.keys())}")

def distancia_coseno(a, b):
    dot = np.dot(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    return 1 - (dot / norms)

# ==========================
#   DECORADOR DE AUTENTICACIÓN
# ==========================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ==========================
#   RUTAS
# ==========================

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login')
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/register')
def register():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    username = session.get('user')
    return render_template('dashboard.html', username=username)

@app.route('/admin/users')
@login_required
def admin_users():
    username = session.get('user')
    return render_template('admin_users.html', username=username)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/api/users', methods=['GET'])
@login_required
def get_users():
    """Obtener lista de todos los usuarios"""
    from core.models import Usuario
    db = SessionLocal()
    try:
        usuarios = db.query(Usuario).all()
        users_list = []
        for u in usuarios:
            users_list.append({
                'id': u.id,
                'name': u.name,
                'access': u.access
            })
        return jsonify({'success': True, 'users': users_list})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        db.close()

@app.route('/api/users/<int:user_id>/toggle_access', methods=['POST'])
@login_required
def toggle_user_access(user_id):
    """Aprobar o desaprobar acceso de un usuario"""
    global base_usuarios
    from core.models import Usuario
    db = SessionLocal()
    try:
        usuario = db.query(Usuario).filter(Usuario.id == user_id).first()
        if not usuario:
            return jsonify({'success': False, 'message': 'Usuario no encontrado'}), 404
        
        # Cambiar estado de acceso
        usuario.access = not usuario.access
        db.commit()
        
        # Recargar base de usuarios
        base_usuarios = cargar_base()
        
        return jsonify({
            'success': True,
            'message': f'Acceso {"otorgado" if usuario.access else "revocado"} para {usuario.name}',
            'access': usuario.access
        })
    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        db.close()

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@login_required
def delete_user(user_id):
    """Eliminar un usuario"""
    global base_usuarios
    from core.models import Usuario
    db = SessionLocal()
    try:
        usuario = db.query(Usuario).filter(Usuario.id == user_id).first()
        if not usuario:
            return jsonify({'success': False, 'message': 'Usuario no encontrado'}), 404
        
        username = usuario.name
        db.delete(usuario)
        db.commit()
        
        # Recargar base de usuarios
        base_usuarios = cargar_base()
        
        return jsonify({
            'success': True,
            'message': f'Usuario {username} eliminado correctamente'
        })
    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        db.close()

# ==========================
#   API DE RECONOCIMIENTO
# ==========================

# Variable global para la cámara
camera = None

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if cap.isOpened():
                camera = cap
                return camera
        return None
    return camera

@app.route('/video_feed')
def video_feed():
    def generate():
        cap = get_camera()
        if cap is None:
            return
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Codificar frame como JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/authenticate', methods=['POST'])
def authenticate():
    cap = get_camera()
    if cap is None:
        return jsonify({'success': False, 'message': 'No se pudo acceder a la cámara'}), 500
    
    # Capturar varios frames para obtener el mejor
    best_result = None
    best_distance = 1e9
    
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            continue
        
        H, W = frame.shape[:2]
        
        # Preprocesar para YOLO
        resized = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.moveaxis(img, -1, 0)
        img = img[np.newaxis, :, :, :]
        
        # Detectar rostro con YOLO
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
        
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            face = frame[y1:y2, x1:x2]
            
            # Reconocimiento con ArcFace
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
            
            if mejor_distancia < best_distance:
                best_distance = mejor_distancia
                best_result = {
                    'usuario': mejor_usuario,
                    'distancia': mejor_distancia,
                    'acceso': tiene_acceso
                }
    
    # Evaluar resultado
    if best_result is None:
        return jsonify({
            'success': False,
            'message': 'No se detectó ningún rostro. Por favor, colócate frente a la cámara.'
        })
    
    if best_result['distancia'] < 0.55 and best_result['acceso']:
        session['user'] = best_result['usuario']
        return jsonify({
            'success': True,
            'user': best_result['usuario'],
            'message': f"¡Bienvenido, {best_result['usuario']}!"
        })
    elif best_result['distancia'] < 0.55 and not best_result['acceso']:
        return jsonify({
            'success': False,
            'message': f"Usuario {best_result['usuario']} identificado, pero no tiene acceso autorizado."
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Rostro no reconocido. Acceso denegado.'
        })

@app.route('/check_camera', methods=['GET'])
def check_camera():
    cap = get_camera()
    if cap is None:
        return jsonify({'available': False})
    return jsonify({'available': True})

@app.route('/register_user', methods=['POST'])
def register_user():
    global base_usuarios
    
    data = request.get_json()
    username = data.get('username', '').strip()
    grant_access = True  # Todos los usuarios registrados tienen acceso automáticamente
    
    if not username:
        return jsonify({'success': False, 'message': 'El nombre de usuario es requerido'}), 400
    
    # Verificar si el usuario ya existe
    if username in base_usuarios:
        return jsonify({'success': False, 'message': f'El usuario "{username}" ya está registrado'}), 400
    
    cap = get_camera()
    if cap is None:
        return jsonify({'success': False, 'message': 'No se pudo acceder a la cámara'}), 500
    
    # Capturar varios frames para obtener el mejor rostro
    best_embedding = None
    best_confidence = 0
    attempts = 15
    
    for _ in range(attempts):
        ret, frame = cap.read()
        if not ret:
            continue
        
        H, W = frame.shape[:2]
        
        # Preprocesar para YOLO
        resized = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.moveaxis(img, -1, 0)
        img = img[np.newaxis, :, :, :]
        
        # Detectar rostro con YOLO
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
        
        if best_box is not None and best_conf > best_confidence:
            x1, y1, x2, y2 = best_box
            face = frame[y1:y2, x1:x2]
            
            # Generar embedding con ArcFace
            face_pre = preprocess_arcface(face)
            embedding = session_arc.run(None, {input_name_arc: face_pre})[0][0]
            
            best_embedding = embedding
            best_confidence = best_conf
    
    # Verificar si se capturó un rostro válido
    if best_embedding is None:
        return jsonify({
            'success': False,
            'message': 'No se detectó ningún rostro. Por favor, colócate frente a la cámara con buena iluminación.'
        })
    
    # Guardar usuario en la base de datos
    try:
        from services.face_recognizer import guardar_usuario
        db = SessionLocal()
        
        # Guardar usuario
        new_user = guardar_usuario(db, username, best_embedding)
        
        # Actualizar permisos de acceso si se especificaron
        if grant_access:
            new_user.access = False
            db.commit()
        
        db.close()
        
        # Recargar base de usuarios
        base_usuarios = cargar_base()
        
        return jsonify({
            'success': True,
            'message': f'Usuario "{username}" registrado exitosamente',
            'username': username,
            'access_granted': grant_access
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error al guardar el usuario: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
