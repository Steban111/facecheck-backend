import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

ROSTROS_DIR = "rostros"
if not os.path.exists(ROSTROS_DIR):
    os.makedirs(ROSTROS_DIR)

def comparar_imagenes(ruta_img1, ruta_img2):
    """
    Compara dos imágenes de forma matemática usando histogramas con Pillow.
    ¡Ultra ligero, sin OpenCV ni IA pesada, optimizado para servidores de 512MB!
    """
    try:
        # Cargar imágenes en escala de grises ('L') usando Pillow
        img1 = Image.open(ruta_img1).convert('L')
        img2 = Image.open(ruta_img2).convert('L')

        # Redimensionar a un tamaño estándar (150x150)
        img1_res = img1.resize((150, 150))
        img2_res = img2.resize((150, 150))

        # Obtener los histogramas directamente desde Pillow (lista de 256 valores)
        hist1 = np.array(img1_res.histogram(), dtype=np.float32)
        hist2 = np.array(img2_res.histogram(), dtype=np.float32)

        # Normalizar los histogramas para evitar que los cambios de luz afecten la comparación
        norm1 = hist1 / (np.sum(hist1) + 1e-6)
        norm2 = hist2 / (np.sum(hist2) + 1e-6)

        # Calcular la correlación de Pearson (equivalente matemático a cv2.compareHist)
        mean1 = np.mean(norm1)
        mean2 = np.mean(norm2)
        
        num = np.sum((norm1 - mean1) * (norm2 - mean2))
        den = np.sqrt(np.sum((norm1 - mean1) ** 2) * np.sum((norm2 - mean2) ** 2))
        
        similitud = num / den if den != 0 else 0.0
        
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
