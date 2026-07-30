import os

# 🛑 1. LIMITAR RECURSOS ANTES DE CARGAR TENSORFLOW
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"

import requests
import numpy as np
import cv2
import gc
from flask import Flask, request, jsonify
from flask_cors import CORS
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from deepface import DeepFace
import cloudinary
import cloudinary.api

app = Flask(__name__)
# CORS abierto para permitir peticiones desde Expo / Móvil
CORS(app, resources={r"/*": {"origins": "*"}})

ROSTROS_DIR = "rostros"
if not os.path.exists(ROSTROS_DIR):
    os.makedirs(ROSTROS_DIR)

# Haar Cascade para detección ligera de rostros
xml_filename = "haarcascade_frontalface_default.xml"
if not os.path.exists(xml_filename):
    try:
        url_cascade = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        res = requests.get(url_cascade, timeout=10)
        with open(xml_filename, "wb") as f:
            f.write(res.content)
    except Exception as e:
        print(f"Error descargando Haar Cascade: {e}")

face_cascade = cv2.CascadeClassifier(xml_filename)

# Configuración de Cloudinary
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", "n04i6zmx"),
    api_key=os.environ.get("CLOUDINARY_API_KEY", "922889323116662"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET", "G8LWZb4xv_gwWuEh9xg8JV0veaE"),
    secure=True
)

# Conexión con Google Sheets
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
    print(f"⚠️ Google Sheets no disponible: {e}")

# CONFIGURACIÓN DE USUARIOS, PINS Y HOJAS
USUARIOS_CONFIG = {
    "steban": {"sheet_name": "Asistencia seminario sibimbe", "pin": "1999"},
    "liss": {"sheet_name": "Asistencia seminario riberas", "pin": "1302"},
    "prueba": {"sheet_name": "Pruebas", "pin": "1234"}
}

def comprimir_y_guardar_foto(file_storage, destino_path):
    try:
        in_memory_bytes = np.frombuffer(file_storage.read(), np.uint8)
        img = cv2.imdecode(in_memory_bytes, cv2.IMREAD_COLOR)
        del in_memory_bytes

        if img is not None:
            h, w = img.shape[:2]
            max_size = 300
            if max(h, w) > max_size:
                scale = max_size / float(max(h, w))
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            cv2.imwrite(destino_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            del img
            gc.collect()
            return True
    except Exception as e:
        print(f"⚠️ Error al comprimir imagen: {e}")
    
    file_storage.seek(0)
    file_storage.save(destino_path)
    return False

def sincronizar_desde_cloudinary():
    """Descarga los rostros de Cloudinary al arrancar el servidor si no existen localmente."""
    try:
        if len(os.listdir(ROSTROS_DIR)) > 0:
            return

        print("☁️ Descargando rostros de Cloudinary...")
        resources = cloudinary.api.resources(prefix="rostros/", type="upload", max_results=500)
        
        for resource in resources.get("resources", []):
            public_id = resource["public_id"]
            url = resource["secure_url"]
            partes = public_id.split('/')
            
            if len(partes) >= 3 and partes[1] != "no_registrados":
                nombre_usuario = partes[1]
                usuario_dir = os.path.join(ROSTROS_DIR, nombre_usuario)
                if not os.path.exists(usuario_dir):
                    os.makedirs(usuario_dir)
                
                dest_file = os.path.join(usuario_dir, "registro.jpg")
                if not os.path.exists(dest_file):
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        with open(dest_file, "wb") as f:
                            f.write(response.content)
            
    except Exception as e:
        print(f"⚠️ Error sync Cloudinary: {e}")

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
        print(f"❌ Error al escribir en Google Sheets: {e}")

# Ruta de Salud (Ping)
@app.route("/", methods=["GET", "HEAD"])
def status_check():
    return jsonify({"status": "online", "mensaje": "Servidor Activo 🚀"}), 200

# Endpoint 1: LOGIN
@app.route("/login", methods=["POST", "OPTIONS"])
@app.route("/api/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS": return jsonify({"status": "ok"}), 200

    data = request.get_json(silent=True) or request.form or {}
    user = str(data.get("username") or data.get("user") or data.get("usuario") or "").strip().lower()
    pin = str(data.get("pin") or data.get("password") or "").strip()

    if user in USUARIOS_CONFIG and str(USUARIOS_CONFIG[user]["pin"]) == pin:
        target_sheet = USUARIOS_CONFIG[user]["sheet_name"]
        return jsonify({
            "mensaje": "Acceso Correcto",
            "success": True,
            "sheet_name": target_sheet,
            "usuario": user
        }), 200

    return jsonify({"error": "Usuario o PIN incorrectos", "success": False}), 401

# Endpoint 2: STREAM EN VIVO
@app.route("/stream_detect", methods=["POST", "OPTIONS"])
@app.route("/api/stream_detect", methods=["POST", "OPTIONS"])
def stream_detect():
    if request.method == "OPTIONS": return jsonify({"status": "ok"}), 200

    file = request.files.get('photo') or request.files.get('file') or request.files.get('image')
    if not file: return jsonify({"detectado": False}), 200

    try:
        in_memory_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(in_memory_bytes, cv2.IMREAD_GRAYSCALE)
        del in_memory_bytes

        if img is None: return jsonify({"detectado": False}), 200

        faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=3, minSize=(40, 40))
        hay_rostro = len(faces) > 0
        del img
        gc.collect()

        return jsonify({"detectado": hay_rostro}), 200
    except Exception:
        return jsonify({"detectado": False}), 200

# Endpoint 3: ASISTENCIA Y RECONOCIMIENTO
@app.route("/facecheck", methods=["POST", "OPTIONS"])
@app.route("/api/facecheck", methods=["POST", "OPTIONS"])
def facecheck():
    if request.method == "OPTIONS": return jsonify({"status": "ok"}), 200

    file = request.files.get('photo') or request.files.get('file') or request.files.get('image')
    if not file:
        return jsonify({"autorizado": False, "success": False, "mensaje": "No se recibió foto"}), 200

    target_sheet = request.form.get("sheet_name") or "Pruebas"
    temp_path = os.path.join(ROSTROS_DIR, f"temp_{int(datetime.now().timestamp())}.jpg")

    try:
        comprimir_y_guardar_foto(file, temp_path)
        mejor_precision = 0.0
        mejor_usuario = "Desconocido"
        autorizado = False

        gc.collect()

        # Modelo ultraligero OpenFace para no congelar Render
        dfs = DeepFace.find(
            img_path=temp_path,
            db_path=ROSTROS_DIR,
            model_name="OpenFace",
            detector_backend="skip",
            distance_metric="cosine",
            enforce_detection=False,
            silent=True
        )

        if len(dfs) > 0 and not dfs[0].empty:
            df = dfs[0].sort_values(by="distance")
            mejor_match = df.iloc[0]
            distancia = float(mejor_match["distance"])
            ruta_match = str(mejor_match["identity"])

            precision = round(max(0.0, (1.0 - distancia) * 100.0), 2)

            if precision >= 55.0:
                mejor_usuario = os.path.basename(os.path.dirname(ruta_match))
                autorizado = True
                mejor_precision = precision

        del dfs
        gc.collect()

        if autorizado:
            registrar_asistencia(mejor_usuario, target_sheet_name=target_sheet)
            return jsonify({
                "autorizado": True,
                "success": True,
                "usuario": mejor_usuario.replace("_", " ").title(),
                "precision": mejor_precision,
                "mensaje": f"Asistencia registrada: {mejor_usuario}"
            }), 200
        else:
            return jsonify({
                "autorizado": False,
                "success": False,
                "mensaje": "Rostro no reconocido",
                "precision": mejor_precision
            }), 200

    except Exception as e:
        print(f"⚠️ Error en asistencia: {e}")
        return jsonify({"autorizado": False, "success": False, "mensaje": f"Error: {str(e)}"}), 200

    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except Exception: pass
        gc.collect()

sincronizar_desde_cloudinary()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
