import os
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
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

# Base de datos provisional de usuarios y sus Hojas/Pestañas de Google Sheets
USUARIOS_CONFIG = {
    "steban": {
        "sheet_name": "Asistencia seminario sibimbe",
        "pin": "1999"
    },
    "liss": {
        "sheet_name": "Asistencia seminario riberas",
        "pin": "1302"
    },
    "prueba": {
        "sheet_name": "Pruebas",
        "pin": "1234"
    }
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


def registrar_asistencia(usuario_carpeta, target_sheet_name="Pruebas"):
    if client is None:
        print("❌ No se puede registrar la asistencia: No hay conexión con Google Sheets.")
        return

    try:
        doc = client.open("Registro de Asistencias") # Nombre de tu archivo principal
        try:
            sheet = doc.worksheet(target_sheet_name)
        except Exception:
            sheet = doc.sheet1

        nombre_bonito = usuario_carpeta.replace("_", " ").title()
        ahora = datetime.now()
        
        dias_espanol = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        dia_semana = dias_espanol[ahora.weekday()]
        
        fecha_actual = ahora.strftime("%Y-%m-%d")
        hora_actual = ahora.strftime("%H:%M:%S")
        
        nueva_fila = [nombre_bonito, dia_semana, fecha_actual, hora_actual]
        sheet.append_row(nueva_fila)
        print(f"🚀 ¡Asistencia guardada en la pestaña '{target_sheet_name}' para: {nombre_bonito}!")
        
    except Exception as e:
        print(f"❌ Error al registrar en Google Sheets ('{target_sheet_name}'): {e}")


def calcular_similitud_robusta(ruta_img1, ruta_img2):
    """
    Algoritmo mejorado de extracción de matriz facial.
    Aplica normalización L2 y transformación sigmoidal para mapear
    la distancia matemática a un porcentaje de precisión real.
    """
    try:
        # 1. Cargar imágenes, corregir orientación EXIF y redimensionar a un tamaño estándar
        img1 = ImageOps.exif_transpose(Image.open(ruta_img1)).convert('L').resize((100, 100))
        img2 = ImageOps.exif_transpose(Image.open(ruta_img2)).convert('L').resize((100, 100))

        # 2. Ecualización adaptativa de contraste (reduce impacto de luces/sombras)
        img1 = ImageOps.equalize(img1)
        img2 = ImageOps.equalize(img2)

        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)

        # 3. Recorte central enfocado en el rostro (remueve bordes/fondos)
        h, w = arr1.shape
        margin_h, margin_w = int(h * 0.15), int(w * 0.15)
        crop1 = arr1[margin_h:h-margin_h, margin_w:w-margin_w]
        crop2 = arr2[margin_h:h-margin_h, margin_w:w-margin_w]

        # 4. Normalización Estandarizada Z-score
        crop1 = (crop1 - np.mean(crop1)) / (np.std(crop1) + 1e-6)
        crop2 = (crop2 - np.mean(crop2)) / (np.std(crop2) + 1e-6)

        # 5. Distancia Euclidiana Normalizada
        distancia = np.linalg.norm(crop1 - crop2) / crop1.size

        # 6. Mapeo Sigmoide: Convierte distancia en Porcentaje de Similitud Coherente
        # Si las imágenes son muy parecidas, distancia tiende a < 0.05
        similitud_raw = 1.0 / (1.0 + np.exp( (distancia - 0.042) * 80.0 ))
        
        precision_pct = round(float(similitud_raw) * 100.0, 2)
        return precision_pct
        
    except Exception as e:
        print(f"❌ Error al comparar imágenes: {e}")
        return 0.0

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
            public_id = "registro",
            folder = f"rostros/{nombre}",
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
        
    target_sheet = request.form.get("sheet_name", "Pruebas")

    sincronizar_desde_cloudinary()
        
    file = request.files['photo']
    temp_path = os.path.join(ROSTROS_DIR, "temp_upload.jpg")
    file.save(temp_path)
    
    mejor_precision = 0.0
    mejor_usuario = "Desconocido"
    
    # 🔍 EVALUACIÓN COMPLETA: Compara con TODOS antes de decidir
    for usuario in os.listdir(ROSTROS_DIR):
        usuario_path = os.path.join(ROSTROS_DIR, usuario)
        if os.path.isdir(usuario_path):
            foto_registro = os.path.join(usuario_path, "registro.jpg")
            if os.path.exists(foto_registro):
                precision = calcular_similitud_robusta(temp_path, foto_registro)
                print(f"DEBUG: Evaluando {usuario}. Resultado: {precision}%")
                
                if precision > mejor_precision:
                    mejor_precision = precision
                    mejor_usuario = usuario

    if os.path.exists(temp_path):
        os.remove(temp_path)

    # 🛡️ UMBRAL DE SEGURIDAD AJUSTADO (65.0%)
    UMBRAL_SEGURIDAD = 65.0

    if mejor_precision >= UMBRAL_SEGURIDAD:
        registrar_asistencia(mejor_usuario, target_sheet_name=target_sheet)
        return jsonify({
            "autorizado": True,
            "usuario": mejor_usuario.replace("_", " ").title(),
            "precision": mejor_precision
        }), 200
    else:
        return jsonify({
            "autorizado": False,
            "precision": mejor_precision,
            "usuario": mejor_usuario.replace("_", " ").title() if mejor_precision > 0 else "Desconocido"
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
