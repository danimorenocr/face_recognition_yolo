# Web Application - Face Recognition Authentication

## Archivos de la aplicación web

- `app.py` - Servidor Flask principal
- `templates/login.html` - Página de inicio de sesión con reconocimiento facial
- `templates/dashboard.html` - Panel de control protegido

## Cómo ejecutar la aplicación web

1. Asegúrate de tener activado el entorno virtual:
```bash
.venv\Scripts\activate
```

2. Ejecuta el servidor Flask:
```bash
python app.py
```

3. Abre tu navegador y ve a:
```
http://localhost:5000
```

## Características

- ✅ Autenticación mediante reconocimiento facial en tiempo real
- ✅ Sistema de sesiones seguras
- ✅ Interfaz moderna y responsive
- ✅ Feed de video en vivo desde la cámara
- ✅ Página de dashboard protegida
- ✅ Verificación de permisos de acceso por usuario

## Uso

1. La aplicación abrirá automáticamente tu cámara
2. Colócate frente a la cámara con buena iluminación
3. Haz clic en "Autenticar con Rostro"
4. Si tu rostro es reconocido y tienes acceso autorizado, serás redirigido al dashboard
5. Puedes cerrar sesión en cualquier momento desde el dashboard

## Seguridad

- Solo usuarios registrados en la base de datos pueden iniciar sesión
- Solo usuarios con `access=True` pueden acceder al sistema
- Las sesiones están protegidas con clave secreta
- Todas las rutas del dashboard requieren autenticación
