# 🎭 Sistema de Reconocimiento Facial con Autenticación Web

Sistema completo de reconocimiento facial con aplicación web que incluye autenticación biométrica, registro de usuarios y panel de administración.

## 📋 Descripción

Este proyecto implementa un sistema integral de reconocimiento facial que:
- 🔐 **Autenticación web** mediante reconocimiento facial en tiempo real
- 👤 **Registro de usuarios** con captura facial desde el navegador
- 👥 **Panel de administración** para gestionar usuarios y permisos
- 🎯 Detecta rostros usando un modelo YOLO optimizado
- 🔍 Extrae embeddings faciales con ArcFace R100
- 💾 Almacena usuarios en base de datos PostgreSQL con SQLAlchemy
- ⚡ Identifica usuarios con alta precisión y baja latencia

## 🛠️ Tecnologías

### Backend
- **Python 3.13+**
- **Flask** - Framework web
- **SQLAlchemy** - ORM para base de datos
- **PostgreSQL** - Base de datos relacional
- **OpenCV** - Procesamiento de video e imágenes
- **ONNX Runtime** - Inferencia optimizada de modelos
- **NumPy** - Operaciones numéricas
- **YOLO Face Detection** - Detección de rostros
 https://huggingface.co/deepghs/yolo-face/blob/1eb85df806aed8a6789c88dcf7194005aaed6fe2/yolov8n-face/model.onnx
- **ArcFace R100** - Extracción de embeddings faciales
wget https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true -O arcface.onnx


## 📁 Estructura del Proyecto

```
.
├── app.py                    # ⭐ Aplicación web Flask principal
├── reconocer.py              # Script de reconocimiento (standalone)
├── registrar.py              # Script de registro (standalone)
├── utils.py                  # Funciones auxiliares (preprocesamiento)
├── session_options.py        # Optimización de sesiones ONNX
├── core/
│   ├── config.py            # Configuración de la aplicación
│   ├── database.py          # Configuración de base de datos
│   └── models.py            # Modelos SQLAlchemy
├── services/
│   └── face_recognizer.py   # Lógica de reconocimiento facial
├── templates/
│   ├── login.html           # 🔐 Página de autenticación facial
│   ├── register.html        # ➕ Página de registro de usuarios
│   ├── dashboard.html       # 🏠 Dashboard principal
│   └── admin_users.html     # 👥 Panel de administración
├── models/
│   ├── model.onnx           # Modelo YOLO para detección facial
│   └── arcface_r100.onnx    # Modelo ArcFace para embeddings
└── tests/
    ├── test-yolo.py         # Prueba del modelo YOLO
    └── verificar_embedding.py # Utilidad para inspeccionar embeddings
```

## 🚀 Instalación

### 1. Requisitos previos

- Python 3.13 o superior
- PostgreSQL instalado y en ejecución
- Cámara web conectada

### 2. Clonar el repositorio

```bash
git clone https://github.com/danimorenocr/face_recognition_yolo.git
cd face_recognition_yolo
```

### 3. Crear entorno virtual

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 4. Instalar dependencias

```bash
pip install flask opencv-python numpy onnxruntime sqlalchemy psycopg2-binary pydantic-settings python-dotenv
```

### 5. Configurar base de datos

Crea un archivo `.env` en la raíz del proyecto:

```env
DATABASE_URL=postgresql://usuario:contraseña@localhost:5432/face_recognition
DATABASE_SCHEMA=public
```

### 6. Inicializar la base de datos

```python
from core.database import init_db
init_db()
```

## 📖 Uso

### 🌐 Aplicación Web (Recomendado)

#### Iniciar el servidor

```bash
python app.py
```

El servidor estará disponible en: **http://localhost:5000**

#### Funcionalidades de la aplicación web:

1. **🔐 Login con Reconocimiento Facial** (`/login`)
   - Accede con tu rostro, sin contraseñas
   - Detección automática de la cámara
   - Validación de permisos de acceso
   - Mensajes claros de éxito/error

2. **➕ Registro de Nuevos Usuarios** (`/register`)
   - Formulario simple con nombre de usuario
   - Captura facial automática (15 frames para mejor calidad)
   - Validación de usuarios duplicados
   - Acceso automático al sistema

3. **🏠 Dashboard Principal** (`/dashboard`)
   - Panel protegido con sesión
   - Información del usuario autenticado
   - Acceso a módulos del sistema
   - Enlace a administración de usuarios

4. **👥 Administración de Usuarios** (`/admin/users`)
   - Lista completa de usuarios registrados
   - Estadísticas en tiempo real
   - Aprobar/Revocar acceso con un clic
   - Eliminar usuarios con confirmación
   - Actualización automática

### 💻 Scripts Standalone

#### Registrar un nuevo usuario (CLI)

```bash
python registrar.py
```

1. Ingresa el nombre del usuario
2. Colócate frente a la cámara
3. El sistema captura automáticamente
4. Presiona `ESC` para cancelar

#### Reconocer rostros (CLI)

```bash
python reconocer.py
```

- Visualización en tiempo real
- Cuadros de colores según estado
- Nombre y distancia de similitud
- Presiona `ESC` para salir

## ⚙️ Configuración Avanzada

### Modelos de IA

**YOLO Face Detection:**
- Entrada: (1, 3, 640, 640) - RGB normalizado
- Salida: (1, 5, 8400) - [x, y, w, h, confidence]
- Umbral de confianza: 0.55

**ArcFace R100:**
- Entrada: (1, 112, 112, 3) - HWC format
- Normalización: (pixel - 127.5) / 128.0
- Salida: Vector de 512 dimensiones

### Parámetros ajustables

En `app.py` y `reconocer.py`:

```python
# Umbral de confianza para detección YOLO
conf > 0.55

# Umbral de similitud para reconocimiento
mejor_distancia < 0.55
```

**Ajustar umbral de similitud:**
- `0.4-0.5`: Más estricto, menos falsos positivos
- `0.6-0.7`: Más permisivo, menos rechazos

### Variables de entorno

```env
# Base de datos
DATABASE_URL=postgresql://user:pass@host:port/dbname
DATABASE_SCHEMA=public

# Flask (opcional)
FLASK_ENV=development
SECRET_KEY=tu-clave-secreta
```

## 🎯 Características Principales

### ✨ Aplicación Web
- ✅ Autenticación sin contraseñas
- ✅ Sistema de sesiones seguras
- ✅ Feed de video en tiempo real
- ✅ Interfaz moderna y responsive
- ✅ Panel de administración completo
- ✅ Control de acceso por usuario
- ✅ Validación de permisos

### 🔒 Seguridad
- ✅ Solo usuarios con `access=True` pueden iniciar sesión
- ✅ Sesiones protegidas con clave secreta
- ✅ Rutas protegidas con decorador `@login_required`
- ✅ Confirmaciones para acciones destructivas

### 🚀 Rendimiento
- ✅ Inferencia cada 2 frames (optimización)
- ✅ Anti-parpadeo de detecciones
- ✅ Sesiones ONNX optimizadas
- ✅ Recarga automática de base de usuarios

## 📊 Funcionamiento del Sistema

### Flujo de Autenticación

1. Usuario accede a `/login`
2. Cámara se activa automáticamente
3. Usuario hace clic en "Autenticar"
4. Sistema captura 10 frames
5. YOLO detecta rostros en cada frame
6. ArcFace genera embeddings
7. Comparación con base de datos (distancia coseno)
8. Si distancia < 0.55 y `access=True` → Login exitoso
9. Sesión creada, redirige a dashboard

### Flujo de Registro

1. Usuario accede a `/register`
2. Ingresa nombre de usuario
3. Sistema valida que no exista
4. Captura 15 frames para mejor calidad
5. Selecciona el mejor rostro detectado
6. Genera embedding con ArcFace
7. Guarda en base de datos con `access=True`
8. Usuario puede iniciar sesión inmediatamente

### Cálculo de Similitud

```python
distancia_coseno = 1 - (embedding_live · embedding_db) / (||embedding_live|| × ||embedding_db||)
```

- Distancia ≈ 0: Alta similitud (mismo usuario)
- Distancia ≈ 1: Baja similitud (usuarios diferentes)

## 🔧 API REST

### Endpoints disponibles

| Método | Ruta | Descripción | Requiere Auth |
|--------|------|-------------|---------------|
| GET | `/` | Redirige según estado de sesión | No |
| GET | `/login` | Página de login | No |
| GET | `/register` | Página de registro | No |
| POST | `/authenticate` | Autenticar con rostro | No |
| POST | `/register_user` | Registrar nuevo usuario | No |
| GET | `/dashboard` | Dashboard principal | Sí |
| GET | `/admin/users` | Panel de administración | Sí |
| GET | `/api/users` | Lista de usuarios | Sí |
| POST | `/api/users/<id>/toggle_access` | Cambiar acceso | Sí |
| DELETE | `/api/users/<id>` | Eliminar usuario | Sí |
| GET | `/logout` | Cerrar sesión | No |
| GET | `/video_feed` | Stream de video | No |
| GET | `/check_camera` | Verificar cámara | No |

## 🛠️ Solución de Problemas

### La cámara no se detecta

- Verifica permisos de cámara en el navegador
- Asegúrate de que ninguna otra app esté usando la cámara
- Prueba con diferentes navegadores
- Reinicia el servidor Flask

### Error de conexión a la base de datos

```
sqlalchemy.exc.OperationalError
```

- Verifica que PostgreSQL esté corriendo
- Comprueba credenciales en `.env`
- Asegúrate de que la base de datos existe

### Detección inexacta

- Mejora la iluminación del entorno
- Acércate más a la cámara
- Mantén el rostro quieto durante la captura
- Ajusta el umbral de confianza

### Falsos positivos/negativos

- Ajusta el umbral de similitud (0.55 por defecto)
- Re-registra usuarios con mejores condiciones
- Verifica la calidad de la cámara

## 📝 Notas Técnicas

- Embeddings almacenados como `BYTEA` en PostgreSQL
- Sesiones Flask con tiempo de expiración
- Feed de video usa `multipart/x-mixed-replace`
- Anti-parpadeo: mantiene detección hasta 6 frames sin detección
- Optimización: inferencia cada 2 frames
- Thread-safe: múltiples usuarios pueden acceder simultáneamente

## 🚧 Roadmap

- [ ] Autenticación multifactor (facial + PIN)
- [ ] Registro de logs de acceso
- [ ] API REST completa
- [ ] Dashboard con estadísticas y gráficos
- [ ] Exportar/Importar usuarios
- [ ] Detección de liveness (anti-spoofing)
- [ ] Soporte para múltiples rostros simultáneos
- [ ] App móvil con React Native

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add: amazing feature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 👤 Autor

Daniela Moreno - Proyecto de Reconocimiento Facial

## 🙏 Agradecimientos

- [YOLO Face Detection](https://huggingface.co/deepghs/yolo-face) - Detección de rostros
- [ArcFace](https://arxiv.org/abs/1801.07698) - Reconocimiento facial de alta precisión
- ONNX Runtime - Optimización de inferencia
- Flask - Framework web ligero y potente
- SQLAlchemy - ORM robusto para Python

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub
