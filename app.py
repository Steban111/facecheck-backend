import os
import shutil
import requests
import numpy as np
import cv2
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
CORS(app, resources={r"/*": {"origins": "*"}})

ROSTROS_DIR = "rostros"
NO_REGISTRADOS_DIR = os.path.join(ROSTROS_DIR, "no_registrados")

# Crear carpetas base
for directo in [ROSTROS_DIR, NO_REGISTRADOS_DIR]:
    if not os.path.exists(directo):
        os.makedirs(directo)

# Cargar o descargar el XML de Haar Cascade
xml_filename = "haarcascade_frontalface_default.xml"
if not os.path.exists(xml_filename):
    print("📥 Descargando archivo Haar Cascade...")
    url_cascade = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    res = requests.get(url_cascade)
    with open(xml_filename, "wb") as f:
        f.write(res.content)
    print("✅ Haar Cascade descargado exitosamente.")

face_cascade = cv2.CascadeClassifier(xml_filename)

# ==========================================
# ☁️ CONFIGURACIÓN DE CLOUDINARY
# ==========================================
cloudinary.config(
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME", "n04i6zmx"),
    api_key = os.environ.get("CLOUDINARY_API_KEY", "922889323116662"),
    api_secret = os.environ.get("CLOUDINARY_API_SECRET", "G8LWZb4xv_gwWuEh9xg8JV0veaE"),
    secure = True
)

# ==========================================
# 📊 GOOGLE SHEETS & BASE DE USUARIOS
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
    print(f"❌ Error al conectar Google Sheets: {e}")

# 📌 AQUÍ AGREGAS O EDITAS TUS USUARIOS/PROFESORES
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
# 🔄 FUNCIONES DE APOYO Y CORRECCIÓN
# ==========================================

def arreglar_orientacion_imagen(ruta_imagen):
    """ Corrige la rotación EXIF típica de las fotos tomadas en móviles """
    try:
        image = Image.open(ruta_imagen)
        image = ImageOps.exif_transpose(image)
        image.save(ruta_imagen)
    except Exception as e:
        print(f"⚠️ No se pudo ajustar la orientación de la imagen: {e}")


def sincronizar_desde_cloudinary():
    """ Descarga los rostros de Cloudinary si el disco local de Render está vacío """
    try:
        # Ignoramos la carpeta 'no_registrados' al contar carpetas de alumnos
        carpetas_locales = [
            d for d in os.listdir(ROSTROS_DIR) 
            if os.path.isdir(os.path.join(ROSTROS_DIR, d)) and d != "no_registrados"
        ]
        if len(carpetas_locales) > 0:
            return

        print("🔄 Servidor vacío detectado. Sincronizando rostros desde Cloudinary...")
        resources = cloudinary.api.resources(prefix="rostros/", type="upload")
        
        for resource in resources.get("resources", []):
            public_id = resource["public_id"]
            url = resource["secure_url"]
            
            partes = public_id.split('/')
            if len(partes) >= 3 and partes[1] != "no_registrados":
                nombre_usuario = partes[1]
                usuario_dir = os.path.join(ROSTROS_DIR, nombre_usuario)
                if not os.path.exists(usuario_dir):
                    os.makedirs(usuario_dir)
                
                response = requests.get(url)
                if response.status_code == 200:
                    dest_file = os.path.join(usuario_dir, "registro.jpg")
                    with open(dest_file, "wb") as f:
                        f.write(response.content)
                    arreglar_orientacion_imagen(dest_file)
                    print(f"📥 Rostro de '{nombre_usuario}' recuperado con éxito.")
                    
        print("✅ Sincronización completa.")
    except Exception as e:
        print(f"❌ Error al sincronizar con Cloudinary: {e}")


def registrar_asistencia(usuario_carpeta, target_sheet_name="Pruebas"):
    if client is None:
        print("❌ No hay conexión con Google Sheets.")
        return

    try:
        doc = client.open("Registro de Asistencias")
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
        print(f"🚀 ¡Asistencia guardada en '{target_sheet_name}' para: {nombre_bonito}!")
        
    except Exception as e:
        print(f"❌ Error en Google Sheets ('{target_sheet_name}'): {e}")


def extraer_y_normalizar_rostro(ruta_imagen):
    try:
        arreglar_orientacion_imagen(ruta_imagen)
        img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        img_equalized = cv2.equalizeHist(img)

        faces = face_cascade.detectMultiScale(
            img_equalized,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40)
        )

        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
            if len(faces) == 0:
                print(f"⚠️ No se detectó ninguna cara en: {ruta_imagen}")
                return None

        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        rostro_crop = img_equalized[y:y+h, x:x+w]
        rostro_final = cv2.resize(rostro_crop, (128, 128))
        return rostro_final

    except Exception as e:
        print(f"❌ Error al procesar rostro: {e}")
        return None


def calcular_similitud_facial_avanzada(ruta_captura, ruta_registro):
    try:
        rostro_cap = extraer_y_normalizar_rostro(ruta_captura)
        rostro_reg = extraer_y_normalizar_rostro(ruta_registro)

        if rostro_cap is None or rostro_reg is None:
            return 0.0

        arr1 = rostro_cap.astype(np.float32)
        arr2 = rostro_reg.astype(np.float32)

        arr1 = (arr1 - np.mean(arr1)) / (np.std(arr1) + 1e-6)
        arr2 = (arr2 - np.mean(arr2)) / (np.std(arr2) + 1e-6)

        distancia = np.linalg.norm(arr1 - arr2) / arr1.size

        similitud_raw = 1.0 / (1.0 + np.exp((distancia - 0.022) * 120.0))
        precision_pct = round(float(similitud_raw) * 100.0, 2)

        return precision_pct

    except Exception as e:
        print(f"❌ Error biométrico: {e}")
        return 0.0

# ==========================================
# 🛣️ RUTAS DEL SERVIDOR FLASK
# ==========================================

@app.route("/", methods=["GET", "HEAD"])
def status_check():
    return jsonify({
        "status": "online",
        "mensaje": "Servidor Biométrico FaceCheck Activo 🚀"
    }), 200

@app.route("/login", methods=["POST"])
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

@app.route("/register", methods=["POST"])
@app.route("/api/register", methods=["POST"])
@app.route("/api/register-face", methods=["POST"])
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
    
    arreglar_orientacion_imagen(local_path)
    
    try:
        upload_result = cloudinary.uploader.upload(
            local_path,
            public_id = "registro",
            folder = f"rostros/{nombre}",
            overwrite = True
        )
        print(f"☁️ Foto de {nombre} respaldada en Cloudinary.")
    except Exception as e:
        print(f"❌ Error al subir a Cloudinary: {e}")

    return jsonify({"mensaje": f"Usuario {nombre} registrado con éxito"}), 200

@app.route("/facecheck", methods=["POST"])
@app.route("/api/facecheck", methods=["POST"])
@app.route("/api/check-attendance", methods=["POST"])
def facecheck():
    if 'photo' not in request.files:
        return jsonify({"error": "No se envió ninguna foto"}), 400
        
    target_sheet = request.form.get("sheet_name", "Pruebas")

    sincronizar_desde_cloudinary()
        
    file = request.files['photo']
    temp_path = os.path.join(ROSTROS_DIR, "temp_upload.jpg")
    file.save(temp_path)
    
    arreglar_orientacion_imagen(temp_path)
    
    mejor_precision = 0.0
    mejor_usuario = "Desconocido"
    
    # Evaluar contra cada carpeta de alumno (omitiendo no_registrados)
    for usuario in os.listdir(ROSTROS_DIR):
        if usuario == "no_registrados":
            continue
            
        usuario_path = os.path.join(ROSTROS_DIR, usuario)
        if os.path.isdir(usuario_path):
            foto_registro = os.path.join(usuario_path, "registro.jpg")
            if os.path.exists(foto_registro):
                precision = calcular_similitud_facial_avanzada(temp_path, foto_registro)
                print(f"DEBUG: Evaluando {usuario}. Coincidencia: {precision}%")
                
                if precision > mejor_precision:
                    mejor_precision = precision
                    mejor_usuario = usuario

    # UMBRAL DE SEGURIDAD ESTRICTO
    UMBRAL_SEGURIDAD = 80.0

    if mejor_precision >= UMBRAL_SEGURIDAD:
        # Borrar archivo temporal ya que sí fue reconocido
        if os.path.exists(temp_path):
            os.remove(temp_path)

        registrar_asistencia(mejor_usuario, target_sheet_name=target_sheet)
        return jsonify({
            "autorizado": True,
            "usuario": mejor_usuario.replace("_", " ").title(),
            "precision": mejor_precision
        }), 200
    else:
        # 🚨 NO RECONOCIDO: Guardar la foto del rostro en no_registrados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_no_reg = f"desconocido_{timestamp}.jpg"
        path_no_reg = os.path.join(NO_REGISTRADOS_DIR, nombre_no_reg)
        
        # Mover o guardar foto en carpeta no_registrados
        if os.path.exists(temp_path):
            shutil.move(temp_path, path_no_reg)
        
        # Respaldar en Cloudinary dentro de rostros/no_registrados
        try:
            cloudinary.uploader.upload(
                path_no_reg,
                public_id = f"desconocido_{timestamp}",
                folder = "rostros/no_registrados",
                overwrite = False
            )
            print(f"⚠️ Foto no reconocida guardada localmente y en Cloudinary: {nombre_no_reg}")
        except Exception as e:
            print(f"❌ Error subiendo captura no reconocida a Cloudinary: {e}")

        return jsonify({
            "autorizado": False,
            "mensaje": "No registrado",
            "precision": mejor_precision,
            "usuario": "No registrado"
        }), 200

@app.route('/ver_rostros', methods=['GET'])
def ver_rostros():
    sincronizar_desde_cloudinary()
    if os.path.exists(ROSTROS_DIR):
        archivos = [
            d for d in os.listdir(ROSTROS_DIR) 
            if os.path.isdir(os.path.join(ROSTROS_DIR, d)) and d != "no_registrados"
        ]
        return {"total_alumnos": len(archivos), "alumnos_registrados": archivos}, 200
    return {"error": "La carpeta de rostros no existe todavía"}, 404

@app.route('/limpiar_cache', methods=['GET'])
def limpiar_cache():
    """ Elimina la carpeta local para forzar a sincronizar de nuevo desde Cloudinary """
    try:
        if os.path.exists(ROSTROS_DIR):
            shutil.rmtree(ROSTROS_DIR)
            os.makedirs(ROSTROS_DIR)
            os.makedirs(NO_REGISTRADOS_DIR)
        return jsonify({"mensaje": "Caché borrada exitosamente"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    sincronizar_desde_cloudinary()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
