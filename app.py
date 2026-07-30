import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

# Inicializar Flask y habilitar CORS para recibir peticiones de Expo / Celular
app = Flask(__name__)
CORS(app)

# Carpeta donde se guardarán las fotos de nuevos rostros registrados
CARPETA_ROSTROS = 'rostros_registrados'
os.makedirs(CARPETA_ROSTROS, exist_ok=True)

# ----------------------------------------------------
# 1. PING INICIAL (Utilizado por el Loader / Splash)
# ----------------------------------------------------
@app.route('/', methods=['GET'])
def ping():
    # Esta ruta despierta a Render y le confirma a la app que el server ya está listo
    return jsonify({
        "status": "online",
        "message": "Servidor FaceCheck AI activo"
    }), 200

# ----------------------------------------------------
# 2. LOGIN DE USUARIO
# ----------------------------------------------------
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    usuario = data.get('usuario', '').strip().lower()
    pin = data.get('pin', '').strip()

    # 🛑 AQUÍ PUEDES PONER TU LÓGICA / BASE DE DATOS REAL
    # Ejemplo con usuarios de prueba:
    usuarios_validos = {
        "steban": {"pin": "1234", "sheet_name": "Asistencia_Steban"},
        "admin": {"pin": "0000", "sheet_name": "Asistencia_General"}
    }

    if usuario in usuarios_validos and usuarios_validos[usuario]['pin'] == pin:
        return jsonify({
            "success": True,
            "usuario": usuario,
            "sheet_name": usuarios_validos[usuario]['sheet_name']
        }), 200
    else:
        return jsonify({
            "success": False,
            "error": "Usuario o PIN incorrectos."
        }), 401

# ----------------------------------------------------
# 3. CONFIRMAR ASISTENCIA (CÁMARA)
# ----------------------------------------------------
@app.route('/api/facecheck', methods=['POST'])
def facecheck():
    if 'photo' not in request.files:
        return jsonify({"success": False, "mensaje": "No se recibió la foto"}), 400

    foto = request.files['photo']
    sheet_name = request.form.get('sheet_name', 'General')

    # Guardar foto temporalmente si se requiere procesar con OpenCV / Face Recognition
    # ruta_temp = os.path.join('/tmp', foto.filename)
    # foto.save(ruta_temp)

    # 🛑 AQUÍ VA TU LÓGICA DE RECONOCIMIENTO FACIAL Y REGISTRO EN GOOGLE SHEETS
    # Ejemplo de respuesta exitosa simulada:
    reconocido = True  # Cambiar por el resultado real de tu IA
    precision_obtenida = 98.5

    if reconocido:
        # e.g., guardar_en_google_sheets(sheet_name, usuario_detectado)
        return jsonify({
            "success": True,
            "autorizado": True,
            "usuario": "Persona Detectada",
            "precision": precision_obtenida,
            "mensaje": "Asistencia registrada correctamente"
        }), 200
    else:
        return jsonify({
            "success": False,
            "autorizado": False,
            "mensaje": "Rostro no reconocido en el sistema"
        }), 400

# ----------------------------------------------------
# 4. REGISTRAR NUEVO ROSTRO (NOMBRE + FOTO)
# ----------------------------------------------------
@app.route('/api/register_face', methods=['POST'])
def register_face():
    if 'photo' not in request.files:
        return jsonify({"success": False, "mensaje": "No se envió ninguna imagen"}), 400

    nombre_completo = request.form.get('nombre_completo', '').strip()
    if not nombre_completo:
        return jsonify({"success": False, "mensaje": "El nombre completo es obligatorio"}), 400

    foto = request.files['photo']

    # Formatear el nombre para guardar el archivo de la foto
    nombre_archivo = f"{nombre_completo.lower().replace(' ', '_')}.jpg"
    ruta_guardado = os.path.join(CARPETA_ROSTROS, nombre_archivo)
    
    # Guardar la foto del nuevo rostro en la carpeta
    foto.save(ruta_guardado)

    # 🛑 AQUÍ PUEDES AGREGAR ENTRENAMIENTO DE IA O GUARDAR EL NOMBRE EN GOOGLE SHEETS/DB

    return jsonify({
        "success": True,
        "mensaje": f"Rostro de '{nombre_completo}' guardado exitosamente."
    }), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
