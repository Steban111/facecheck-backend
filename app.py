import os
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import gspread
from google.oauth2.service_account import Credentials

# Importar el cerebro de IA
from deepface import DeepFace

app = Flask(__name__)
CORS(app)

CARPETA_ROSTROS = 'rostros_registrados'
os.makedirs(CARPETA_ROSTROS, exist_ok=True)

# ----------------------------------------------------
# CONFIGURACIÓN DE GOOGLE SHEETS
# ----------------------------------------------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

SHEET_ID = '1PTZEluLo7IpHvWzvQ6SpCfbN9mSiRLKMO5fn1MfhDv0' 

def conectar_sheets():
    """Conecta con Google Sheets usando tu archivo credentials.json"""
    try:
        creds = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID)
    except Exception as e:
        print(f"❌ Error conectando a Sheets: {e}")
        return None

# ----------------------------------------------------
# 1. PING INICIAL (Para despertar el servidor)
# ----------------------------------------------------
@app.route('/', methods=['GET'])
def ping():
    return jsonify({"status": "online", "message": "Servidor FaceCheck AI activo"}), 200

# ----------------------------------------------------
# 2. LOGIN DE USUARIO (Acceso al escáner)
# ----------------------------------------------------
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    usuario = data.get('usuario', '').strip().lower()
    pin = data.get('pin', '').strip()

    usuarios_validos = {
        "steban": {"pin": "1234", "sheet_name": "Asistencia seminario sibimbe"},
        "admin": {"pin": "0000", "sheet_name": "Asistencia seminario riberas"}
    }

    if usuario in usuarios_validos and usuarios_validos[usuario]['pin'] == pin:
        return jsonify({
            "success": True,
            "usuario": usuario,
            "sheet_name": usuarios_validos[usuario]['sheet_name']
        }), 200
    else:
        return jsonify({"success": False, "error": "Usuario o PIN incorrectos."}), 401

# ----------------------------------------------------
# 3. CONFIRMAR ASISTENCIA (IA REAL)
# ----------------------------------------------------
@app.route('/api/facecheck', methods=['POST'])
def facecheck():
    if 'photo' not in request.files:
        return jsonify({"success": False, "mensaje": "No se recibió la foto del celular"}), 400

    foto = request.files['photo']
    sheet_name = request.form.get('sheet_name', 'Asistencia seminario sibimbe')

    # Si la carpeta de rostros está vacía, no hay con quién comparar
    if not os.listdir(CARPETA_ROSTROS) or (len(os.listdir(CARPETA_ROSTROS)) == 1 and os.listdir(CARPETA_ROSTROS)[0].endswith('.pkl')):
        return jsonify({"success": False, "autorizado": False, "mensaje": "No hay rostros registrados en la base de datos."}), 400

    # Guardar foto temporalmente para que DeepFace la analice
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_scan.jpg")
    foto.save(temp_path)

    try:
        # Ejecutar reconocimiento facial contra la carpeta
        # enforce_detection=False evita que la app crashee si la foto sale borrosa y no se ve una cara
        resultados = DeepFace.find(img_path=temp_path, db_path=CARPETA_ROSTROS, enforce_detection=False, silent=True)
        
        # Validar si hubo match
        if len(resultados) > 0 and not resultados[0].empty:
            # Match encontrado
            match_row = resultados[0].iloc[0]
            ruta_identidad = match_row['identity'] # ej: rostros_registrados/juan_perez.jpg
            distancia = match_row['distance'] # Entre más bajo, más parecido
            
            # Calcular una precisión falsa pero realista basada en la distancia (solo visual para la app)
            precision_obtenida = round(max(0, (1 - distancia) * 100), 1)

            # Limpiar nombre (de 'juan_perez.jpg' a 'juan perez')
            nombre_archivo = os.path.basename(ruta_identidad)
            usuario_detectado = os.path.splitext(nombre_archivo)[0].replace("_", " ").title()

            # Escribir en Google Sheets
            documento = conectar_sheets()
            if documento:
                hoja = documento.worksheet(sheet_name)
                ahora = datetime.now()
                dias_espanol = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
                
                dia_semana = dias_espanol[ahora.weekday()]       
                fecha_iso = ahora.strftime("%Y-%m-%d")            
                hora_iso = ahora.strftime("%H:%M:%S")             
                
                nueva_fila = [usuario_detectado, dia_semana, fecha_iso, hora_iso]
                hoja.append_row(nueva_fila)
                
                return jsonify({
                    "success": True,
                    "autorizado": True,
                    "usuario": usuario_detectado,
                    "precision": precision_obtenida,
                    "mensaje": f"✅ Asistencia registrada para {usuario_detectado}"
                }), 200
            else:
                return jsonify({"success": False, "mensaje": "Error de conexión con Sheets"}), 500
        else:
            return jsonify({"success": False, "autorizado": False, "mensaje": "Rostro no reconocido en el sistema."}), 400

    except Exception as e:
        print(f"Error DeepFace: {e}")
        return jsonify({"success": False, "autorizado": False, "mensaje": "Error procesando biometría."}), 500

    finally:
        # Siempre borrar la imagen temporal por seguridad
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ----------------------------------------------------
# 4. REGISTRAR NUEVO ROSTRO
# ----------------------------------------------------
@app.route('/api/register_face', methods=['POST'])
def register_face():
    if 'photo' not in request.files:
        return jsonify({"success": False, "mensaje": "No se envió ninguna imagen"}), 400

    nombre_completo = request.form.get('nombre_completo', '').strip()
    if not nombre_completo:
        return jsonify({"success": False, "mensaje": "El nombre completo es obligatorio"}), 400

    foto = request.files['photo']
    # Guardar todo en minúsculas y con guión bajo para que DeepFace lo lea bien
    nombre_archivo = f"{nombre_completo.lower().replace(' ', '_')}.jpg"
    ruta_guardado = os.path.join(CARPETA_ROSTROS, nombre_archivo)
    foto.save(ruta_guardado)

    # IMPORTANTE: Borrar el archivo de caché de DeepFace para que reconozca la nueva foto
    pkl_path = os.path.join(CARPETA_ROSTROS, "representations_vgg_face.pkl")
    if os.path.exists(pkl_path):
        os.remove(pkl_path)

    return jsonify({"success": True, "mensaje": f"Rostro de '{nombre_completo}' guardado exitosamente."}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
