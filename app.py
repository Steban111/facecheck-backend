import os

# 🛑 CONFIGURACIÓN DE MEMORIA Y CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"

import shutil
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

# Cloudinary
import cloudinary
import cloudinary.uploader
import cloudinary.api

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

ROSTROS_DIR = "rostros"
if not os.path.exists(ROSTROS_DIR):
    os.makedirs(ROSTROS_DIR)

# Detector Haar Cascade ligero
xml_filename = "haarcascade_frontalface_default.xml"
if not os.path.exists(xml_filename):
    print("📥 Descargando Haar Cascade...")
    url_cascade = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    res = requests.get(url_cascade, timeout=10)
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

def procesar_y_guardar_foto_ligera(file_storage, destino_path):
    try:
        in_memory_bytes = np.frombuffer(file_storage.read(), np.uint8)
        img = cv2.imdecode(in_memory_bytes, cv2.IMREAD_COLOR)
        del in_memory_bytes

        if img is not None:
            h, w = img.shape[:2]
            max_size = 400
            if max(h, w) > max_size:
                scale = max_size / float(max(h, w))
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            cv2.imwrite(destino_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            del img
            gc.collect()
            return True
    except Exception as e:
        print(f"⚠️ Error procesando imagen: {e}")
    
    file_storage.seek(0)
    file_storage.save(destino_path)
    return False

def borrar_cache_biometrico():
    pkl_path = os.path.join(ROSTROS_DIR, "representations_facenet.pkl")
    if os.path.exists(pkl_path):
        try:
            os.remove(pkl_path)
            print("♻️ Caché biométrico eliminado.")
        except Exception as e:
            print(f"⚠️ Error borrando PKL: {e}")

def sincronizar_desde_cloudinary(forzar=False):
    try:
        carpetas = [d for d in os.listdir(ROSTROS_DIR) if os.path.isdir(os.path.join(ROSTROS_DIR, d))]
        if len(carpetas) > 0 and not forzar:
            return

        resources = cloudinary.api.resources(prefix="rostros/", type="upload", max_results=500)
        hubo_cambios = False
        
        for resource in resources.get("resources", []):
            public_id = resource["public_id"]
            url = resource["secure_url"]
            partes = public_id.split('/')
            
            if len(partes) >= 3 and partes[1] != "no_registrados":
                nombre_usuario = partes[1]
                usuario_dir = os.path.join(ROSTROS_DIR, nombre_usuario)
                if not os.path.exists(usuario_dir):
                    os.makedirs(usuario_dir)
                    hubo_cambios = True
                
                dest_file = os.path.join(usuario_dir, "registro.jpg")
                if not os.path.exists(dest_file) or forzar:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        with open(dest_file, "wb") as f:
                            f.write(response.content)
                        hubo_cambios = True
                        
        if hubo_cambios:
            borrar_cache_biometrico()
            
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

@app.route("/", methods=["GET", "HEAD"])
def status_check():
    return jsonify({"status": "online", "mensaje": "Servidor Activo 🚀"}), 200

# ==========================================
# 🚀 DETECCIÓN EN VIVO (STREAM)
# ==========================================
@app.route("/stream_detect", methods=["POST", "OPTIONS"])
@app.route("/api/stream_detect", methods=["POST", "OPTIONS"])
def stream_detect():
    if request.method == "OPTIONS":
        res = jsonify({"status": "ok"})
        res.headers.add("Access-Control-Allow-Origin", "*")
        return res, 200

    file = request.files.get('photo') or request.files.get('file') or request.files.get('image')
    
    if not file:
        res = jsonify({"detectado": False})
        res.headers.add("Access-Control-Allow-Origin", "*")
        return res, 200

    try:
        in_memory_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(in_memory_bytes, cv2.IMREAD_GRAYSCALE)
        del in_memory_bytes

        if img is None:
            res = jsonify({"detectado": False})
            res.headers.add("Access-Control-Allow-Origin", "*")
            return res, 200

        faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))

        if len(faces) == 0:
            img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            faces = face_cascade.detectMultiScale(img_rot, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))
            del img_rot

        hay_rostro = len(faces) > 0
        del img
        gc.collect()

        print(f"📸 Stream -> Caras: {len(faces)} | Color: {'Morado' if hay_rostro else 'Gris'}")

        res = jsonify({"detectado": hay_rostro})
        res.headers.add("Access-Control-Allow-Origin", "*")
        return res, 200

    except Exception as e:
        res = jsonify({"detectado": False})
        res.headers.add("Access-Control-Allow-Origin", "*")
        return res, 200

# ==========================================
# 🔑 LOGIN
# ==========================================
@app.route("/login", methods=["POST", "OPTIONS"])
@app.route("/api/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        res = jsonify({"status": "ok"})
        res.headers.add("Access-Control-Allow-Origin", "*")
        return res, 200

    data = request.get_json(silent=True) or request.form or {}
    user = str(data.get("username") or data.get("user") or data.get("usuario") or "").strip().lower()
    pin = str(data.get("pin") or data.get("password") or "").strip()

    if user in USUARIOS_CONFIG and str(USUARIOS_CONFIG[user]["pin"]) == pin:
        target_sheet = USUARIOS_CONFIG[user]["sheet_name"]
        res = jsonify({
            "mensaje": "Exito",
            "success": True,
            "status": "ok",
            "sheet_assigned": target_sheet,
            "sheet_name": target_sheet,
            "usuario": user
        })
        res.headers.add("Access-Control-Allow-Origin", "*")
        return res, 200

    res = jsonify({"error": "Credenciales incorrectas", "success": False})
    res.headers.add("Access-Control-Allow-Origin", "*")
    return res, 401

# ==========================================
# ✅ RECONOCIMIENTO FACIAL Y ASISTENCIA (MULTIRUTA)
# ==========================================
@app.route("/facecheck", methods=["POST", "OPTIONS"])
@app.route("/api/facecheck", methods=["POST", "OPTIONS"])
@app.route("/check-attendance", methods=["POST", "OPTIONS"])
@app.route("/api/check-attendance", methods=["POST", "OPTIONS"])
@app.route("/asistencia", methods=["POST", "OPTIONS"])
@app.route("/api/asistencia", methods=["POST", "OPTIONS"])
def facecheck():
    if request.method == "OPTIONS":
        res = jsonify({"status": "ok"})
        res.headers.add("Access-Control-Allow-Origin", "*")
        return res, 200

    # Captura la foto bajo cualquier nombre posible
    file = request.files.get('photo') or request.files.get('file') or request.files.get('image') or request.files.get('picture')

    if not file:
        res = jsonify({"autorizado": False, "success": False, "mensaje": "No se recibió foto"})
        res.headers.add("Access-Control-Allow-Origin", "*")
        return res, 200

    target_sheet = request.form.get("sheet_name") or request.form.get("sheet") or "Pruebas"

    temp_path = os.path.join(ROSTROS_DIR, f"temp_{int(datetime.now().timestamp())}.jpg")

    try:
        procesar_y_guardar_foto_ligera(file, temp_path)

        mejor_precision = 0.0
        mejor_usuario = "Desconocido"
        autorizado = False

        dfs = DeepFace.find(
            img_path=temp_path,
            db_path=ROSTROS_DIR,
            model_name="Facenet",
            detector_backend="opencv",
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

            if precision >= 60.0:
                mejor_usuario = os.path.basename(os.path.dirname(ruta_match))
                autorizado = True
                mejor_precision = precision

        if autorizado:
            registrar_asistencia(mejor_usuario, target_sheet_name=target_sheet)
            res = jsonify({
                "autorizado": True,
                "success": True,
                "usuario": mejor_usuario.replace("_", " ").title(),
                "precision": mejor_precision,
                "mensaje": f"Asistencia registrada: {mejor_usuario}"
            })
        else:
            res = jsonify({
                "autorizado": False,
                "success": False,
                "mensaje": "Rostro no reconocido",
                "precision": mejor_precision,
                "usuario": "No registrado"
            })

    except Exception as e:
        print(f"⚠️ Error en asistencia: {e}")
        res = jsonify({
            "autorizado": False,
            "success": False,
            "mensaje": f"Error: {str(e)}"
        })

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        gc.collect()

    res.headers.add("Access-Control-Allow-Origin", "*")
    return res, 200

sincronizar_desde_cloudinary(forzar=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
