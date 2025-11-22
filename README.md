# 🎭 Sistema de Reconocimiento Facial

Sistema de reconocimiento facial en tiempo real utilizando YOLO para detección de rostros y ArcFace para identificación biométrica.

## 📋 Descripción

Este proyecto implementa un sistema completo de reconocimiento facial que:
- Detecta rostros en tiempo real usando un modelo YOLO optimizado
- Extrae embeddings faciales con ArcFace R100
- Permite registrar nuevos usuarios
- Identifica usuarios registrados con alta precisión

## 🛠️ Tecnologías

- **Python 3.x**
- **OpenCV** - Procesamiento de video e imágenes
- **ONNX Runtime** - Inferencia de modelos
- **NumPy** - Operaciones numéricas
- **YOLO Face Detection** - Detección de rostros
 https://huggingface.co/deepghs/yolo-face/blob/1eb85df806aed8a6789c88dcf7194005aaed6fe2/yolov8n-face/model.onnx
- **ArcFace R100** - Extracción de embeddings faciales
wget https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true -O arcface.onnx


## 📁 Estructura del Proyecto

```
.
├── reconocer.py              # Script principal de reconocimiento
├── registrar.py              # Script para registrar nuevos usuarios
├── utils.py                  # Funciones auxiliares (preprocesamiento)
├── test-yolo.py             # Script de prueba del modelo YOLO
├── verificar_embedding.py    # Utilidad para inspeccionar embeddings
├── modelo_det_face/
│   └── model.onnx           # Modelo YOLO para detección facial
├── modelo_arcface/
│   └── arcface_r100.onnx    # Modelo ArcFace para embeddings
└── base_rostros/            # Base de datos de usuarios (embeddings)
    ├── usuario1.npy
    └── usuario2.npy
```

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd "Reconocimiento facial"
```

### 2. Crear entorno virtual (recomendado)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
pip install opencv-python numpy onnxruntime
```

## 📖 Uso

### Registrar un nuevo usuario

```bash
python registrar.py
```

1. Ingresa el nombre del usuario cuando se solicite
2. Colócate frente a la cámara
3. El sistema detectará tu rostro automáticamente
4. El embedding se guardará en `base_rostros/[nombre].npy`
5. Presiona `ESC` para cancelar

### Reconocer rostros

```bash
python reconocer.py
```

- El sistema mostrará la cámara en tiempo real
- Los rostros detectados se enmarcarán en verde (reconocido) o rojo (desconocido)
- Se mostrará el nombre y la distancia de similitud
- Presiona `ESC` para salir

### Probar detección YOLO

```bash
python test-yolo.py
```

Útil para verificar que el modelo de detección funciona correctamente.

### Verificar embeddings

```bash
python verificar_embedding.py
```

Muestra la forma y valores de los embeddings guardados (para debugging).

## ⚙️ Configuración

### Modelos

- **YOLO Face Detection**: Detecta rostros en imágenes de 640x640
  - Umbral de confianza: 0.55
  - Entrada: (1, 3, 640, 640)
  - Salida: (1, 5, 8400) - [x, y, w, h, conf]

- **ArcFace R100**: Genera embeddings de 512 dimensiones
  - Entrada: (1, 112, 112, 3) - HWC format
  - Normalización: (pixel - 127.5) / 128.0
  - Salida: Vector de 512 dimensiones

### Parámetros ajustables

En `reconocer.py`:

```python
# Umbral de confianza para detección
conf > 0.55  # Línea 49

# Umbral de similitud para identificación
mejor_distancia < 0.55  # Línea 80
```

**Ajustar umbral de similitud**:
- Valores más bajos (0.4-0.5): Más estricto, menos falsos positivos
- Valores más altos (0.6-0.7): Más permisivo, menos rechazos

## 🎯 Características

- ✅ Detección de rostros en tiempo real
- ✅ Reconocimiento multi-usuario
- ✅ Sistema de registro simple
- ✅ Búsqueda automática de cámara
- ✅ Indicadores visuales (cuadros y etiquetas)
- ✅ Cálculo de distancia coseno para similitud
- ✅ Filtrado de detecciones inválidas

## 📊 Funcionamiento

### Proceso de Detección

1. **Captura**: Frame de la cámara
2. **Preprocesamiento**: Redimensionar a 640x640, normalizar
3. **Detección**: YOLO identifica rostros
4. **Extracción**: Se recorta el rostro detectado
5. **Embedding**: ArcFace genera vector característico
6. **Comparación**: Distancia coseno con base de datos
7. **Identificación**: Si distancia < umbral → usuario reconocido

### Cálculo de Similitud

```python
distancia = 1 - (a · b) / (||a|| × ||b||)
```

Donde:
- `a`: Embedding del rostro en vivo
- `b`: Embedding del usuario registrado
- Valores cercanos a 0: Alta similitud
- Valores cercanos a 1: Baja similitud

## 🔧 Solución de Problemas

### La cámara no se detecta

- Verifica que la cámara esté conectada
- Prueba con diferentes índices en `cv2.VideoCapture(i)`
- Asegúrate de que ninguna otra aplicación esté usando la cámara

### Detección inexacta

- Mejora la iluminación
- Acércate más a la cámara
- Ajusta el umbral de confianza en `reconocer.py`

### Falsos positivos/negativos

- Ajusta el umbral de similitud (línea 80 en `reconocer.py`)
- Re-registra usuarios con mejores condiciones de iluminación
- Verifica que los embeddings se hayan guardado correctamente

### Error al cargar modelos

```
Error: [ONNXRuntimeError]
```

- Verifica que los archivos `.onnx` existan en sus carpetas
- Comprueba que `onnxruntime` esté instalado correctamente

## 📝 Notas Técnicas

- Los embeddings se almacenan en formato `.npy` (NumPy)
- El modelo ArcFace usa formato HWC (Height, Width, Channels), no CHW
- La detección YOLO trabaja en espacio 640x640 y se escala al frame original
- Se selecciona solo la detección con mayor confianza por frame

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 👤 Autor

Daniela Moreno - Proyecto de Reconocimiento Facial

## 🙏 Agradecimientos

- Modelo YOLO Face Detection
- ArcFace: Additive Angular Margin Loss for Deep Face Recognition
- ONNX Runtime por la optimización de inferencia
