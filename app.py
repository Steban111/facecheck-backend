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

# Cloudinary
import cloudinary
import cloudinary.uploader
import cloudinary.api

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

ROSTROS_DIR = "rostros"

# Ya no creamos la carpeta NO_REGISTRADOS_DIR porque eliminaremos las fotos al instante.
if not os.path.exists(ROSTROS_DIR):
    os.makedirs(ROSTROS_DIR)

xml_filename = "haarcascade_frontalface_default.xml"
if not os.path.exists(xml_filename):
    print("📥 Descargando Haar Cascade...")
    url_cascade = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    res = requests.get(url_cascade)
    with open(xml_filename, "wb") as f:
        f.write(res.content)

face_cascade = cv2.CascadeClassifier(xml_filename)

cloudinary.config(
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME", "n04i6zmx"),
    api_key = os.environ.get("CLOUDINARY_API_KEY", "922889323116662"),
    api_secret = os.environ.get("CLOUDINARY_API_SECRET", "G8LWZb4xv_gwWuEh9xg8JV0veaE"),
    secure = True
)

SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]

client = None
try:
    if os.path.exists("credenciales.json"):
        creds = ServiceAccountCredentials.from_json_keyfile_name("credenciales.json", SCOPE)
        client = gspread.authorize(creds)
        print("✅ Google Sheets conectado.")
except Exception as e:
    print(f"❌ Error Google Sheets: {e}")

USUARIOS_CONFIG = {
    "steban": {"sheet_name": "Asistencia seminario sibimbe", "pin": "1999"},
    "liss": {"sheet_name": "Asistencia seminario riberas", "pin": "1302"},
    "prueba": {"sheet_name": "Pruebas", "pin": "1234"}
}

def arreglar_orientacion_imagen(ruta_imagen):
    try:
        image = Image.open(ruta_imagen)
        image = ImageOps.exif_transpose(image)
        image.thumbnail((800, 800))
        image.save(ruta_imagen, quality=85)
    except Exception as e:
        print(f"⚠️ Error orientación/resize: {e}")

def sincronizar_desde_cloudinary(forzar=False):
    try:
        carpetas = [d for d in os.listdir(ROSTROS_DIR) if os.path.isdir(os.path.join(ROSTROS_DIR, d))]
        if len(carpetas) > 0 and not forzar:
            return

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
    except Exception as e:
        print(f"❌ Error sync Cloudinary: {e}")

def registrar_asistencia(usuario_carpeta, target_sheet_name="Pruebas"):
    if client is None: return
    try:
        doc = client.open("Registro de Asistencias")
        try:
            sheet = doc.worksheet(target_sheet_name)
        except Exception:
            sheet = doc.sheet1
        nombre = usuario_carpeta.replace("_", " ").title()
        ahora = datetime.now()
        dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        nueva_fila = [nombre, dias[ahora.weekday()], ahora.strftime("%Y-%m-%d"), ahora.strftime("%H:%M:%S")]
        sheet.append_row(nueva_fila)
    except Exception as e:
        print(f"❌ Error Sheets: {e}")

def extraer_y_normalizar_rostro(ruta_imagen):
    try:
        arreglar_orientacion_imagen(ruta_imagen)
        img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        img_eq = cv2.equalizeHist(img)
        faces = face_cascade.detectMultiScale(img_eq, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
            if len(faces) == 0: return None
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        return cv2.resize(img_eq[y:y+h, x:x+w], (128, 128))
    except Exception:
        return None

def calcular_similitud_facial_avanzada(ruta_captura, ruta_registro):
    try:
        r1 = extraer_y_normalizar_rostro(ruta_captura)
        r2 = extraer_y_normalizar_rostro(ruta_registro)
        if r1 is None or r2 is None: return 0.0

        arr1 = r1.astype(np.float32)
        arr2 = r2.astype(np.float32)

        mse = np.mean((arr1 - arr2) ** 2)
        if mse > 4500:
            return 15.0

        arr1_norm = (arr1 - np.mean(arr1)) / (np.std(arr1) + 1e-6)
        arr2_norm = (arr2 - np.mean(arr2)) / (np.std(arr2) + 1e-6)
        distancia = np.linalg.norm(arr1_norm - arr2_norm) / arr1.size

        similitud = 1.0 / (1.0 + np.exp((distancia - 0.025) * 80.0))
        return round(float(similitud) * 100.0, 2)
    except Exception:
        return 0.0

@app.route("/", methods=["GET", "HEAD"])
def status_check():
    return jsonify({"status": "online", "mensaje": "Servidor Biométrico Activo 🚀"}), 200

# ==========================================
# 🚀 NUEVO ENDPOINT PARA DETECCIÓN EN VIVO
# ==========================================
@app.route("/api/stream_detect", methods=["POST"])
def stream_detect():
    """Recibe fotos en baja calidad, detecta si hay rostro y responde rapidísimo"""
    if 'photo' not in request.files:
        return jsonify({"detectado": False}), 400
    
    file = request.files['photo']
    npimg = np.fromfile(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return jsonify({"detectado": False}), 400
        
    img_eq = cv2.equalizeHist(img)
    faces = face_cascade.detectMultiScale(img_eq, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    
    if len(faces) > 0:
        return jsonify({"detectado": True}), 200
    else:
        return jsonify({"detectado": False}), 200
# ==========================================

@app.route("/login", methods=["POST"])
@app.route("/api/login", methods=["POST"])
def login():
    data = request.json or {}
    user = data.get("username", "").strip().lower()
    pin = data.get("pin", "").strip()
    if user in USUARIOS_CONFIG and USUARIOS_CONFIG[user]["pin"] == pin:
        return jsonify({"mensaje": "Exito", "sheet_assigned": USUARIOS_CONFIG[user]["sheet_name"]}), 200
    return jsonify({"error": "Credenciales incorrectas"}), 401

# ==========================================
# 📷 REGISTRO: AQUÍ SÍ SE GUARDA EN CLOUDINARY
# ==========================================
@app.route("/register", methods=["POST"])
@app.route("/api/register", methods=["POST"])
@app.route("/api/register-face", methods=["POST"])
def register():
    if 'photo' not in request.files or 'name' not in request.form:
        return jsonify({"error": "Faltan datos"}), 400
    file = request.files['photo']
    nombre = request.form['name'].strip().lower().replace(" ", "_")
    
    usuario_dir = os.path.join(ROSTROS_DIR, nombre)
    if not os.path.exists(usuario_dir): os.makedirs(usuario_dir)
        
    local_path = os.path.join(usuario_dir, "registro.jpg")
    file.save(local_path)
    arreglar_orientacion_imagen(local_path)
    
    # ÚNICO LUGAR donde subimos foto a Cloudinary
    try:
        cloudinary.uploader.upload(local_path, public_id="registro", folder=f"rostros/{nombre}", overwrite=True)
    except Exception as e:
        print(f"Cloudinary error: {e}")

    return jsonify({"mensaje": f"Usuario {nombre} registrado"}), 200

# ==========================================
# ✅ ASISTENCIA: NO SE GUARDA NADA EN CLOUDINARY
# ==========================================
@app.route("/facecheck", methods=["POST"])
@app.route("/api/facecheck", methods=["POST"])
@app.route("/api/check-attendance", methods=["POST"])
def facecheck():
    if 'photo' not in request.files:
        return jsonify({"error": "Falta foto"}), 400
        
    target_sheet = request.form.get("sheet_name", "Pruebas")
    sincronizar_desde_cloudinary(forzar=False)
        
    file = request.files['photo']
    temp_path = os.path.join(ROSTROS_DIR, "temp_upload.jpg")
    
    # 1. Guardamos la foto temporalmente para que OpenCV pueda leerla
    file.save(temp_path)
    arreglar_orientacion_imagen(temp_path)
    
    mejor_precision = 0.0
    mejor_usuario = "Desconocido"
    
    for usuario in os.listdir(ROSTROS_DIR):
        u_path = os.path.join(ROSTROS_DIR, usuario)
        if os.path.isdir(u_path):
            foto_reg = os.path.join(u_path, "registro.jpg")
            if os.path.exists(foto_reg):
                precision = calcular_similitud_facial_avanzada(temp_path, foto_reg)
                if precision > mejor_precision:
                    mejor_precision = precision
                    mejor_usuario = usuario

    UMBRAL_SEGURIDAD = 72.0

    # 2. Pase lo que pase (sea válido o inválido), BORRAMOS la foto temporal de la asistencia
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if mejor_precision >= UMBRAL_SEGURIDAD:
        registrar_asistencia(mejor_usuario, target_sheet_name=target_sheet)
        return jsonify({
            "autorizado": True,
            "usuario": mejor_usuario.replace("_", " ").title(),
            "precision": mejor_precision
        }), 200
    else:
        # Ya no hay lógica de mover a NO_REGISTRADOS_DIR ni subir a Cloudinary
        return jsonify({
            "autorizado": False,
            "mensaje": "No registrado",
            "precision": mejor_precision,
            "usuario": "No registrado"
        }), 200

@app.route('/limpiar_cache', methods=['GET'])
def limpiar_cache():
    try:
        if os.path.exists(ROSTROS_DIR):
            shutil.rmtree(ROSTROS_DIR)
            os.makedirs(ROSTROS_DIR)
        sincronizar_desde_cloudinary(forzar=True)
        return jsonify({"mensaje": "Caché borrada y resincronizada exitosamente"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    sincronizar_desde_cloudinary(forzar=True)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
