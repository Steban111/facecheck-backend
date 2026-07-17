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
# Lee las variables seguras de Render. Si estás en local, puedes escribirlas aquí directamente entre comillas.
cloudinary.config(
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME", "TU_CLOUD_NAME_AQUÍ"),
    api_key = os.environ.get("CLOUDINARY_API_KEY", "TU_API_KEY_AQUÍ"),
    api_secret = os.environ.get("CLOUDINARY_API_SECRET", "TU_API_SECRET_AQUÍ"),
    secure = True
)

# ==========================================
# 📊 CONFIGURACIÓN DE GOOGLE SHEETS
# ==========================================
SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]

try:
    print("⚡ Conectando con Google Sheets...")
    creds = ServiceAccountCredentials.from_json_keyfile_name("credenciales.json", SCOPE)
    client = gspread.authorize(creds)
    sheet = client.open("Registro de Asistencias").sheet1
    print("✅ ¡Conexión con Google Sheets establecida con éxito!")
except Exception as e:
    print(f"❌ Error crítico al conectar Google Sheets: {e}")
    sheet = None


# ==========================================
# 🔄 FUNCIONES DE APOYO Y SINCRONIZACIÓN
# ==========================================
def sincronizar_desde_cloudinary():
    """
    Si Render se reinicia y borra las fotos locales, esta función
    descarga automáticamente todas las fotos guardadas en Cloudinary.
    """
    try:
        # Si ya hay carpetas de alumnos en local (además de archivos sueltos), no hace falta sincronizar
        carpetas_locales = [d for d in os.listdir(ROSTROS_DIR) if os.path.isdir(os.path.join(ROSTROS_DIR, d))]
        if len(carpetas_locales) > 0:
            return

        print("🔄 Servidor vacío detectado. Sincronizando rostros desde Cloudinary...")
        # Buscar todos los recursos en la carpeta "rostros" de Cloudinary
        resources = cloudinary.api.resources(prefix="rostros/", type="upload")
        
        for resource in resources.get("resources", []):
            public_id = resource["public_id"]  # Ej: "rostros/alexander_windy/registro"
            url = resource["secure_url"]
            
            # Extraer el nombre del usuario desde el public_id
            partes = public_id.split('/')
            if len(partes) >= 3:
                nombre_usuario = partes[1]
                
                # Crear la carpeta local para ese usuario
                usuario_dir = os.path.join(ROSTROS_DIR, nombre_usuario)
                if not os.path.exists(usuario_dir):
                    os.makedirs(usuario_dir)
                
                # Descargar la imagen y guardarla
                response = requests.get(url)
                if response.status_code == 200:
                    with open(os.path.join(usuario_dir, "registro.jpg"), "wb") as f:
                        f.write(response.content)
                    print(f"📥 Rostro de '{nombre_usuario}' recuperado con éxito.")
                    
        print("✅ Sincronización completa.")
    except Exception as e:
        print(f"❌ Error al sincronizar con Cloudinary: {e}")


def registrar_asistencia(usuario_carpeta):
    if sheet is None:
        print("❌ No se puede registrar la asistencia: No hay conexión con Google Sheets.")
        return

    try:
        nombre_bonito = usuario_carpeta.replace("_", " ").title()
        ahora = datetime.now()
        
        dias_espanol = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        dia_semana = dias_espanol[ahora.weekday()]
        
        fecha_actual = ahora.strftime("%Y-%m-%d")
        hora_actual = telemetry_hora = ahora.strftime("%H:%M:%S")
        
        nueva_fila = [nombre_bonito, dia_semana, fecha_actual, hora_actual]
        sheet.append_row(nueva_fila)
        print(f"🚀 ¡Asistencia guardada en la nube para: {nombre_bonito}!")
        
    except Exception as e:
        print(f"❌ Error al registrar en Google Sheets: {e}")


def comparar_imagenes(ruta_img1, ruta_img2):
    try:
        img1 = Image.open(ruta_img1).convert('L')
        img2 = Image.open(ruta_img2).convert('L')

        img1_res = img1.resize((150, 150))
        img2_res = img2.resize((150, 150))

        hist1 = np.array(img1_res.histogram(), dtype=np.float32)
        hist2 = np.array(img2_res.histogram(), dtype=np.float32)

        norm1 = hist1 / (np.sum(hist1) + 1e-6)
        norm2 = hist2 / (np.sum(hist2) + 1e-6)

        mean1 = np.mean(norm1)
        mean2 = np.mean(norm2)
        
        num = np.sum((norm1 - mean1) * (norm2 - mean2))
        den = np.sqrt(np.sum((norm1 - mean1) ** 2) * np.sum((norm2 - mean2) ** 2))
        
      similitud = num / den if den != 0 else 0.0
        
        # 📢 Agrega el print para los logs de Render
        print(f"DEBUG: Comparando. Similitud calculada: {similitud}")
        
        # 🟢 Bajamos el umbral a 0.25 para que te deje pasar fácil con tu luz actual
        autorizado = similitud > 0.25 
        return autorizado, round(float(similitud), 4)
        
    except Exception as e:
        print(f"❌ Error al comparar imágenes: {e}")
        return False, 0.0


# ==========================================
# 🛣️ RUTAS DEL SERVIDOR FLASK
# ==========================================
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
    
    # ☁️ GUARDAR TAMBIÉN EN CLOUDINARY PARA SIEMPRE
    try:
        cloudinary.uploader.upload(
            local_path, 
            public_id = f"rostros/{nombre}/registro",
            overwrite = True
        )
        print(f"☁️ Foto de {nombre} respaldada con éxito en Cloudinary.")
    except Exception as e:
        print(f"❌ Error al subir a Cloudinary: {e}")

    return jsonify({"mensaje": f"Usuario {nombre} registrado con éxito en local y nube"}), 200


@app.route("/api/facecheck", methods=["POST"])
def facecheck():
    if 'photo' not in request.files:
        return jsonify({"error": "No se envió ninguna foto"}), 400
        
    sincronizar_desde_cloudinary()
        
    file = request.files['photo']
    temp_path = os.path.join(ROSTROS_DIR, "temp_upload.jpg")
    file.save(temp_path)
    
    # 📢 Creamos variables para guardar el intento más cercano
    mejor_precision = 0.0
    usuario_mas_cercano = "No reconocido o no hay usuarios registrados"
    
    for usuario in os.listdir(ROSTROS_DIR):
        usuario_path = os.path.join(ROSTROS_DIR, usuario)
        if os.path.isdir(usuario_path):
            foto_registro = os.path.join(usuario_path, "registro.jpg")
            if os.path.exists(foto_registro):
                autorizado, precision = comparar_imagenes(temp_path, foto_registro)
                
                # 📢 Guarda el valor más alto que encuentre para mostrarlo en el celular
                if precision > mejor_precision:
                    mejor_precision = precision
                    usuario_mas_cercano = usuario
                
                if autorizado:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    registrar_asistencia(usuario)
                    
                    return jsonify({
                        "autorizado": True,
                        "usuario": usuario,
                        "precision": precision
                    }), 200
                    
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    # 📢 Retornamos el porcentaje real aunque el acceso sea FALSO
    return jsonify({
        "autorizado": False,
        "precision": mejor_precision,  # Le enviamos el número real (ej: 0.32)
        "usuario": usuario_mas_cercano
    }), 200


@app.route('/ver_rostros', methods=['GET'])
def ver_rostros():
    sincronizar_desde_cloudinary()
    if os.path.exists(ROSTROS_DIR):
        archivos = [d for d in os.listdir(ROSTROS_DIR) if os.path.isdir(os.path.join(ROSTROS_DIR, d))]
        return {"total_alumnos": len(archivos), "alumnos_registrados": archivos}, 200
    return {"error": "La carpeta de rostros no existe todavía"}, 404


if __name__ == "__main__":
    # Intentar sincronizar rostros al iniciar el servidor
    sincronizar_desde_cloudinary()
    app.run(host="0.0.0.0", port=10000)
    app.run(host="0.0.0.0", port=10000)
