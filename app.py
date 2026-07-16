import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ROSTROS_DIR = "rostros"
if not os.path.exists(ROSTROS_DIR):
    os.makedirs(ROSTROS_DIR)

def comparar_imagenes(ruta_img1, ruta_img2):
    """
    Compara dos imágenes de forma matemática usando histogramas.
    ¡Ultra ligero, sin IA pesada, procesa en milisegundos y consume 0 RAM!
    """
    try:
        # Cargar imágenes en escala de grises
        img1 = cv2.imread(ruta_img1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(ruta_img2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return False, 0.0

        # Redimensionar a un tamaño estándar (150x150 píxeles) para comparar con precisión
        img1_res = cv2.resize(img1, (150, 150))
        img2_res = cv2.resize(img2, (150, 150))

        # Calcular histogramas para ver la distribución de luz y sombras en las facciones
        hist1 = cv2.calcHist([img1_res], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2_res], [0], None, [256], [0, 256])
        
        # Normalizar para que los cambios de luz no afecten la comparación
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # Comparar qué tan parecidos son los dos rostros
        similitud = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Umbral de similitud (si es mayor a 65%, es la misma persona)
        autorizado = similitud > 0.65
        return autorizado, round(float(similitud), 4)
    except Exception as e:
        print(f"❌ Error al comparar imágenes: {e}")
        return False, 0.0

@app.route("/api/register", methods=["POST"])
def register():
    """
    Ruta para registrar a un nuevo usuario guardando su foto
    """
    if 'photo' not in request.files or 'name' not in request.form:
        return jsonify({"error": "Faltan datos"}), 400
        
    file = request.files['photo']
    nombre = request.form['name'].strip().lower().replace(" ", "_")
    
    usuario_dir = os.path.join(ROSTROS_DIR, nombre)
    if not os.path.exists(usuario_dir):
        os.makedirs(usuario_dir)
        
    # Guardar foto de registro
    file.save(os.path.join(usuario_dir, "registro.jpg"))
    return jsonify({"mensaje": f"Usuario {nombre} registrado con éxito"}), 200

@app.route("/api/facecheck", methods=["POST"])
def facecheck():
    """
    Ruta para verificar el rostro que envía el celular/Postman
    """
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
                    return jsonify({
                        "autorizado": True,
                        "usuario": usuario,
                        "precision": precision
                    }), 200
                    
    # Si no coincide con ninguno o no hay usuarios
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return jsonify({
        "autorizado": False,
        "precision": 0,
        "usuario": "No reconocido o no hay usuarios registrados"
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
