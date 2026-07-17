import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

app = Flask(__name__)
CORS(app)

ROSTROS_DIR = "rostros"
if not os.path.exists(ROSTROS_DIR):
    os.makedirs(ROSTROS_DIR)

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


def registrar_asistencia(usuario_carpeta):
    """
    Toma el nombre de la carpeta (ej: 'alexander_windy'), lo embellece
    (ej: 'Alexander Windy') y registra el día, fecha y hora en Google Sheets.
    """
    if sheet is None:
        print("❌ No se puede registrar la asistencia: No hay conexión con Google Sheets.")
        return

    try:
        # Formatear el nombre para que se vea profesional en el Excel
        nombre_bonito = usuario_carpeta.replace("_", " ").title()
        
        # Obtener fecha y hora actual del servidor
        ahora = datetime.now()
        
        # Mapear el día de la semana a español
        dias_espanol = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        dia_semana = dias_espanol[ahora.weekday()]
        
        fecha_actual = ahora.strftime("%Y-%m-%d")
        hora_actual = ahora.strftime("%H:%M:%S")
        
        # Preparar la fila para Google Sheets
        nueva_fila = [nombre_bonito, dia_semana, fecha_actual, hora_actual]
        
        # Guardar en la nube
        sheet.append_row(nueva_fila)
        print(f"🚀 ¡Asistencia guardada en la nube para: {nombre_bonito}!")
        
    except Exception as e:
        print(f"❌ Error al registrar en Google Sheets: {e}")


# ==========================================
# 🧠 ALGORITMO ULTRA-LIGERO DE COMPARACIÓN
# ==========================================
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
        
        autorizado = similitud > 0.65
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
        
    file.save(os.path.join(usuario_dir, "registro.jpg"))
    return jsonify({"mensaje": f"Usuario {nombre} registrado con éxito"}), 200


@app.route("/api/facecheck", methods=["POST"])
def facecheck():
    if 'photo' not in request.files:
        return jsonify({"error": "No se envió ninguna foto"}), 400
        
    file = request.files['photo']
    temp_path = os.path.join(ROSTROS_DIR, "temp_upload.jpg")
    file.save(temp_path)
    
    # Buscar y comparar en las carpetas de los usuarios registrados
    for usuario in os.listdir(ROSTROS_DIR):
        usuario_path = os.path.join(ROSTROS_DIR, usuario)
        if os.path.isdir(usuario_path):
            foto_registro = os.path.join(usuario_path, "registro.jpg")
            if os.path.exists(foto_registro):
                autorizado, precision = comparar_imagenes(temp_path, foto_registro)
                if autorizado:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    # 🔥 ¡AQUÍ OCURRE LA MAGIA! 🔥
                    # Registramos la asistencia en Google Sheets automáticamente
                    registrar_asistencia(usuario)
                    
                    return jsonify({
                        "autorizado": True,
                        "usuario": usuario,
                        "precision": precision
                    }), 200
                    
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return jsonify({
        "autorizado": False,
        "precision": 0,
        "usuario": "No reconocido o no hay usuarios registrados"
    }), 200


@app.route('/ver_rostros', methods=['GET'])
def ver_rostros():
    if os.path.exists(ROSTROS_DIR):
        archivos = os.listdir(ROSTROS_DIR)
        return {"total_alumnos": len(archivos), "alumnos_registrados": archivos}, 200
    return {"error": "La carpeta de rostros no existe todavía"}, 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
