import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import gspread
from google.oauth2.service_account import Credentials

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

# ID de tu hoja de cálculo en Google Sheets
SHEET_ID = '1PTZEluLo7IpHvWzvQ6SpCfbN9mSiRLKMO5fn1MfhDv0' 

def conectar_sheets():
    """Conecta con Google Sheets usando tu archivo credentials.json"""
    try:
        creds = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID)
    except Exception as e:
        print(f"❌ Error conectando a Sheets (Falta credentials.json o está mal): {e}")
        return None

# ----------------------------------------------------
# 1. PING INICIAL
# ----------------------------------------------------
@app.route('/', methods=['GET'])
def ping():
    return jsonify({"status": "online", "message": "Servidor FaceCheck AI activo"}), 200

# ----------------------------------------------------
# 2. LOGIN DE USUARIO
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
# 3. CONFIRMAR ASISTENCIA (AGREGAR FILA NUEVA ABAJO)
# ----------------------------------------------------
@app.route('/api/facecheck', methods=['POST'])
def facecheck():
    if 'photo' not in request.files:
        return jsonify({"success": False, "mensaje": "No se recibió la foto"}), 400

    foto = request.files['photo']
    sheet_name = request.form.get('sheet_name', 'Asistencia seminario sibimbe')

    # Simulamos el resultado del modelo de IA
    reconocido = True
    usuario_detectado = "juan perez" 
    precision_obtenida = 98.5

    if reconocido:
        documento = conectar_sheets()
        if documento:
            try:
                hoja = documento.worksheet(sheet_name)
                
                # Obtener la fecha y la hora exactas
                ahora = datetime.now()
                dias_espanol = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
                
                nombre_limpio = usuario_detectado.strip().lower() # Columna A
                dia_semana = dias_espanol[ahora.weekday()]       # Columna B
                fecha_iso = ahora.strftime("%Y-%m-%d")            # Columna C
                hora_iso = ahora.strftime("%H:%M:%S")             # Columna D
                
                # Fila estructurada: Nombre | Día | Fecha | Hora
                nueva_fila = [nombre_limpio, dia_semana, fecha_iso, hora_iso]
                
                # Insertar en la última línea disponible de Google Sheets
                hoja.append_row(nueva_fila)
                
                return jsonify({
                    "success": True,
                    "autorizado": True,
                    "usuario": nombre_limpio,
                    "precision": precision_obtenida,
                    "mensaje": f"✅ Asistencia registrada para {nombre_limpio} el {dia_semana}"
                }), 200

            except Exception as e:
                return jsonify({"success": False, "mensaje": f"Error escribiendo en Sheets: {e}"}), 500
        else:
            return jsonify({"success": False, "mensaje": "Error de conexión con Google Sheets"}), 500
            
    else:
        return jsonify({"success": False, "autorizado": False, "mensaje": "Rostro no reconocido"}), 400

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
    nombre_archivo = f"{nombre_completo.lower().replace(' ', '_')}.jpg"
    ruta_guardado = os.path.join(CARPETA_ROSTROS, nombre_archivo)
    foto.save(ruta_guardado)

    return jsonify({"success": True, "mensaje": f"Rostro de '{nombre_completo}' guardado."}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
