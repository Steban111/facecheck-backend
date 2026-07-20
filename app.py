import os
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Importar Cloudinary
import cloudinary
import cloudinary.uploader
import cloudinary.api

app = Flask(__name__)
CORS(app)

ROSTROS_DIR = "rostros"
if not os.path.exists(ROSTROS_DIR):
    os.makedirs(ROSTROS_DIR)

# ==========================================
# ☁️ CONFIGURACIÓN DE CLOUDINARY
# ==========================================
cloudinary.config(
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME", "TU_CLOUD_NAME_AQUÍ"),
    api_key = os.environ.get("CLOUDINARY_API_KEY", "TU_API_KEY_AQUÍ"),
    api_secret = os.environ.get("CLOUDINARY_API_SECRET", "TU_API_SECRET_AQUÍ"),
    secure = True
)

# ==========================================
# 📊 CONFIGURACIÓN DE GOOGLE SHEETS & BASE DE USUARIOS
# ==========================================
SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]

client = None
try:
    print("⚡ Conectando con Google Sheets...")
    creds = ServiceAccountCredentials.from_json_keyfile_name("credenciales.json", SCOPE)
    client = gspread.authorize(creds)
    print("✅ ¡Conexión con Google Sheets establecida con éxito!")
except Exception as e:
    print(f"❌ Error crítico al conectar Google Sheets: {e}")

# Base de datos provisional de usuarios y sus Hojas de Google Sheets
# (Aquí puedes mapear cada usuario con el nombre exacto de su Hoja de Google)
USUARIOS_CONFIG = {
    "Steban": {
        "sheet_name": "Registro de Asistencias", # Tu hoja predeterminada
        "pin": "1999"
    },
    "Liss": {
        "sheet_name": "Asistencia Rama 1", # Hoja de tu amigo
        "pin": "1302"
    },
     "Prueba": {
        "sheet_name": "Asistencia Rama 2", # Tu hoja predeterminada
        "pin": "1234"
    },
}

# ==========================================
# 🔄 FUNCIONES DE APOYO Y SINCRONIZACIÓN
# ==========================================
def sincronizar_desde_cloudinary():
    """
    Si Render se reinicia y borra las fotos locales, esta función
    descarga automáticamente todas las fotos guardadas en Cloudinary.
    """
    try:
        carpetas_locales = [d for d in os.listdir(ROSTROS_DIR) if os.path.isdir(os.path.join(ROSTROS_DIR, d))]
        if len(carpetas_locales) > 0:
            return

        print("🔄 Servidor vacío detectado. Sincronizando rostros desde Cloudinary...")
        resources = cloudinary.api.resources(prefix="rostros/", type="upload")
        
        for resource in resources.get("resources", []):
            public_id = resource["public_id"]
            url = resource["secure_url"]
            
            partes = public_id.split('/')
            if len(partes) >= 3:
                nombre_usuario = partes[1]
                usuario_dir = os.path.join(ROSTROS_DIR, nombre_usuario)
                if not os.path.exists(usuario_dir):
                    os.makedirs(usuario_dir)
                
                response = requests.get(url)
                if response.status_code == 200:
                    with open(os.path.join(usuario_dir, "registro.jpg"), "wb") as f:
                        f.write(response.content)
                    print(f"📥 Rostro de '{nombre_usuario}' recuperado con éxito.")
                    
        print("✅ Sincronización completa.")
    except Exception as e:
        print(f"❌ Error al sincronizar con Cloudinary: {e}")


def registrar_asistencia(usuario_carpeta, target_sheet_name="Registro de Asistencias"):
    if client is None:
        print("❌ No se puede registrar la asistencia: No hay conexión con Google Sheets.")
        return

    try:
        sheet = client.open(target_sheet_name).sheet1
        nombre_bonito = usuario_carpeta.replace("_", " ").title()
        ahora = datetime.now()
        
        dias_espanol = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        dia_semana = dias_espanol[ahora.weekday()]
        
        fecha_actual = ahora.strftime("%Y-%m-%d")
        hora_actual = ahora.strftime("%H:%M:%S")
        
        nueva_fila = [nombre_bonito, dia_semana, fecha_actual, hora_actual]
        sheet.append_row(nueva_fila)
        print(f"🚀 ¡Asistencia guardada en la hoja '{target_sheet_name}' para: {nombre_bonito}!")
        
    except Exception as e:
        print(f"❌ Error al registrar en Google Sheets ('{target_sheet_name}'): {e}")


def comparar_imagenes(ruta_img1, ruta_img2):
    """
    Comparación por bloques matriciales normalizados + Histograma
    para mayor estabilidad contra cambios de luz.
    """
    try:
        img1 = Image.open(ruta_img1).convert('L').resize((128, 128))
        img2 = Image.open(ruta_img2).convert('L').resize((128, 128))

        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)

        # Normalizar variaciones de iluminación globales
        arr1 = (arr1 - np.mean(arr1)) / (np.std(arr1) + 1e-6)
        arr2 = (arr2 - np.mean(arr2)) / (np.std(arr2) + 1e-6)

        # Correlación de Pearson
        num = np.sum(arr1 * arr2)
        den = np.sqrt(np.sum(arr1**2) * np.sum(arr2**2))
        similitud = num / den if den != 0 else 0.0

        # Mapeo a porcentaje 0 - 100%
        precision_pct = round(max(0.0, float(similitud)) * 100, 2)
        
        print(f"DEBUG: Comparando. Similitud calculada: {precision_pct}%")
        
        # Umbral estricto para evitar falsos positivos
        autorizado = precision_pct > 68.0 
        return autorizado, precision_pct
        
    except Exception as e:
        print(f"❌ Error al comparar imágenes: {e}")
        return False, 0.0

# ==========================================
# 🛣️ RUTAS DEL SERVIDOR FLASK
# ==========================================

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json or {}
    username = data.get("username", "").strip().lower()
    pin = data.get("pin", "").strip()

    if username in USUARIOS_CONFIG and USUARIOS_CONFIG[username]["pin"] == pin:
        return jsonify({
            "mensaje": "Login exitoso",
            "sheet_assigned": USUARIOS_CONFIG[username]["sheet_name"]
        }), 200
    
    return jsonify({"error": "Usuario o contraseña incorrectos"}), 401


@app.route("/api/register", methods=["POST"])
def register():
    if 'photo' not in request.files or 'name' not in request.form:
        return jsonify({"error": "Faltan datos"}), 400
        
    file = request.files['photo']
    nombre = request.form['name'].strip().lower().replace(" ", "_")
    
    usuario_dir = os.path.join(ROSTROS_DIR, nombre)
    if not os.path.exists(usuario_dir):
        os.makedirs(usuario_dir)
        
    local_path = os.path.join(usuario_dir, "registro.jpg")
    file.save(local_path)
    
    try:
        cloudinary.uploader.upload(
            local_path, 
            public_id = f"rostros/{nombre}/registro",
            overwrite = True
        )
        print(f"☁️ Foto de {nombre} respaldada con éxito en Cloudinary.")
    except Exception as e:
        print(f"❌ Error al subir a Cloudinary: {e}")

    return jsonify({"mensaje": f"Usuario {nombre} registrado con éxito"}), 200


@app.route("/api/facecheck", methods=["POST"])
def facecheck():
    if 'photo' not in request.files:
        return jsonify({"error": "No se envió ninguna foto"}), 400
        
    # Obtener qué hoja de Excel se debe utilizar (si el cliente envía la variable)
    target_sheet = request.form.get("sheet_name", "Registro de Asistencias")

    sincronizar_desde_cloudinary()
        
    file = request.files['photo']
    temp_path = os.path.join(ROSTROS_DIR, "temp_upload.jpg")
    file.save(temp_path)
    
    mejor_precision = 0.0
    usuario_mas_cercano = "Desconocido"
    match_encontrado = False
    
    for usuario in os.listdir(ROSTROS_DIR):
        usuario_path = os.path.join(ROSTROS_DIR, usuario)
        if os.path.isdir(usuario_path):
            foto_registro = os.path.join(usuario_path, "registro.jpg")
            if os.path.exists(foto_registro):
                autorizado, precision = comparar_imagenes(temp_path, foto_registro)
                
                if precision > mejor_precision:
                    mejor_precision = precision
                    usuario_mas_cercano = usuario
                
                if autorizado:
                    match_encontrado = True
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    registrar_asistencia(usuario, target_sheet_name=target_sheet)
                    
                    return jsonify({
                        "autorizado": True,
                        "usuario": usuario.replace("_", " ").title(),
                        "precision": precision
                    }), 200
                    
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return jsonify({
        "autorizado": False,
        "precision": mejor_precision,
        "usuario": usuario_mas_cercano.replace("_", " ").title() if mejor_precision > 0 else "Desconocido"
    }), 200


@app.route('/ver_rostros', methods=['GET'])
def ver_rostros():
    sincronizar_desde_cloudinary()
    if os.path.exists(ROSTROS_DIR):
        archivos = [d for d in os.listdir(ROSTROS_DIR) if os.path.isdir(os.path.join(ROSTROS_DIR, d))]
        return {"total_alumnos": len(archivos), "alumnos_registrados": archivos}, 200
    return {"error": "La carpeta de rostros no existe todavía"}, 404


if __name__ == "__main__":
    sincronizar_desde_cloudinary()
    app.run(host="0.0.0.0", port=10000)
