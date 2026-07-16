import os
# Desactivar GPUs y alertas pesadas de TensorFlow para ahorrar RAM
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
# Forzar a TensorFlow a consumir memoria de forma dinámica y no de golpe
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import os
from PIL import Image, ImageOps
from datetime import datetime

app = Flask(__name__)
CORS(app)

ROSTROS_DIR = os.path.join(os.path.dirname(__file__), "rostros")
ASISTENCIAS_FILE = os.path.join(os.path.dirname(__file__), "asistencias.csv")
# Forzar la creación de la carpeta al iniciar la aplicación en Render
if not os.path.exists(ROSTROS_DIR):
    os.makedirs(ROSTROS_DIR)
def obtener_fecha_bonita_espanol():
    """Retorna la fecha en formato amigable, ej: 'Miércoles 16 de Julio'"""
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    
    ahora = datetime.now()
    dia_semana = dias[ahora.weekday()]
    dia_mes = ahora.day
    mes = meses[ahora.month - 1]
    
    return f"{dia_semana} {dia_mes} de {mes}"

def registrar_asistencia_en_archivo(nombre_usuario):
    """Guarda el nombre, la fecha bonita y la hora en el archivo CSV."""
    try:
        if not os.path.exists(ASISTENCIAS_FILE):
            with open(ASISTENCIAS_FILE, "w", encoding="utf-8") as f:
                f.write("Nombre,Fecha,Hora\n")
        
        fecha_bonita = obtener_fecha_bonita_espanol()
        hora = datetime.now().strftime("%H:%M:%S")
        
        with open(ASISTENCIAS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{nombre_usuario},{fecha_bonita},{hora}\n")
            
        print(f"📝 [Excel] Asistencia guardada: {nombre_usuario} | {fecha_bonita} a las {hora}")
    except Exception as e:
        print(f"⚠️ No se pudo escribir en el archivo de asistencias: {str(e)}")

def corregir_y_limpiar_imagen(path_imagen):
    try:
        img = Image.open(path_imagen)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.save(path_imagen, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"⚠️ Error al limpiar imagen: {str(e)}")
        return False

def verificar_rostro(foto_celular_path):
    try:
        if not os.path.exists(ROSTROS_DIR):
            return {"autorizado": False, "usuario": "Error: Sin carpeta de rostros", "precision": 0}
            
        archivos = os.listdir(ROSTROS_DIR)
        imagenes_registro = [a for a in archivos if a.endswith(('.jpg', '.jpeg', '.png')) and not a.startswith("temp_upload")]
        
        if len(imagenes_registro) == 0:
            return {"autorizado": False, "usuario": "No hay usuarios registrados", "precision": 0}

        corregir_y_limpiar_imagen(foto_celular_path)

        for archivo in imagenes_registro:
            path_autorizado = os.path.join(ROSTROS_DIR, archivo)
            corregir_y_limpiar_imagen(path_autorizado)
            
            print(f"🔄 Comparando foto recibida con: {archivo} (Modo directo)...")
            
            try:
                resultado = DeepFace.verify(
                    img1_path=foto_celular_path, 
                    img2_path=path_autorizado, 
                    model_name="VGG-Face",
                    detector_backend="skip",
                    distance_metric="cosine",
                    enforce_detection=False
                )
                
                distancia = resultado["distance"]
                precision = round((1 - distancia) * 100, 1)
                if precision > 100: precision = 100.0
                
                print(f"   📊 Resultado -> Distancia: {distancia:.4f} | Precisión: {precision}%")
                
                if distancia < 0.62:
                    nombre_usuario = os.path.splitext(archivo)[0].replace("_", " ").title()
                    
                    # 🔔 AQUÍ PERSONALIZAMOS EL MENSAJE EN CONSOLA
                    print(f"   ✅ ¡ASISTENCIA CONFIRMADA! Bienvenido {nombre_usuario}")
                    
                    # Registramos la asistencia con el nuevo formato de fecha
                    registrar_asistencia_en_archivo(nombre_usuario)
                    
                    return {
                        "autorizado": True, 
                        "usuario": nombre_usuario, 
                        "precision": precision,
                        "mensaje": "Asistencia Confirmada"
                    }
                else:
                    print("   ❌ No coincide.")
                    
            except Exception as e_ia:
                print(f"   ❌ Error al analizar la imagen: {str(e_ia)}")
                
        print("\n🔒 Fin de la comparación: Ningún rostro coincidió.\n")
        return {"autorizado": False, "usuario": "Rostro Desconocido", "precision": 0}
        
    except Exception as e:
        print(f"💥 Error general: {str(e)}")
        return {"autorizado": False, "usuario": "Error en el servidor", "precision": 0}

@app.route("/api/facecheck", methods=["POST"])
def facecheck():
    print("\n--- NUEVA PETICIÓN DE VERIFICACIÓN RECIBIDA ---")
    if 'photo' not in request.files:
        return jsonify({"error": "No se envió ninguna foto"}), 400
        
    file = request.files['photo']
    temp_path = os.path.join(ROSTROS_DIR, "temp_upload.jpg")
    file.save(temp_path)
    
    resultado = verificar_rostro(temp_path)
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return jsonify(resultado)

@app.route("/api/register", methods=["POST"])
def register():
    print("\n--- NUEVA PETICIÓN DE REGISTRO RECIBIDA ---")
    if 'photo' not in request.files or 'name' not in request.form:
        return jsonify({"error": "Faltan datos"}), 400
        
    file = request.files['photo']
    nombre = request.form['name'].strip().lower().replace(" ", "_")
    
    if not nombre:
        return jsonify({"error": "Nombre vacío"}), 400
        
    file_path = os.path.join(ROSTROS_DIR, f"{nombre}.jpg")
    file.save(file_path)
    print(f"💾 Nuevo rostro guardado como: {nombre}.jpg")
    
    corregir_y_limpiar_imagen(file_path)
    return jsonify({"success": True, "message": f"Rostro de {nombre.capitalize()} registrado con éxito"})

if __name__ == "__main__":
    if not os.path.exists(ROSTROS_DIR):
        os.makedirs(ROSTROS_DIR)
    # Render nos dará el puerto dinámicamente mediante la variable PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)