import os
import datetime
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
# Define los permisos para editar el Drive
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# ID de tu Excel (El código largo que está en el link de tu Google Sheets)
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
# 3. CONFIRMAR ASISTENCIA (ESCRIBIR EN EXCEL)
# ----------------------------------------------------
@app.route('/api/facecheck', methods=['POST'])
def facecheck():
    if 'photo' not in request.files:
        return jsonify({"success": False, "mensaje": "No se recibió la foto"}), 400

    foto = request.files['photo']
    sheet_name = request.form.get('sheet_name', 'Asistencia seminario sibimbe')

    # Aquí iría tu código de IA para reconocer la foto y sacar el nombre.
    # Por ahora simulamos que la IA detectó a un alumno llamado "Juan Perez"
    reconocido = True
    usuario_detectado = "juan perez" 
    precision_obtenida = 98.5

    if reconocido:
        # 🟢 LÓGICA PARA ESCRIBIR EN GOOGLE SHEETS
        documento = conectar_sheets()
        if documento:
            try:
                hoja = documento.worksheet(sheet_name)
                
                # Obtener todos los datos de la hoja
                datos = hoja.get_all_values()
                
                # Fila 1 = Encabezados (Fechas, etc.)
                encabezados = [str(e).strip().lower() for e in datos[0]]
                
                # Fecha actual (Ejemplo: '2026-07-30' o el formato que uses)
                fecha_hoy = datetime.datetime.now().strftime("%Y-%m-%d") 
                
                # Buscar la columna de la fecha de hoy
                # Si no quieres fecha automática, tendrías que enviar el día desde la app.
                if fecha_hoy not in encabezados:
                    # Si no existe la fecha de hoy, la agregamos en la primera columna vacía
                    col_fecha = len(encabezados) + 1
                    hoja.update_cell(1, col_fecha, fecha_hoy)
                else:
                    col_fecha = encabezados.index(fecha_hoy) + 1

                # Buscar la fila del alumno en la Columna A (índice 0)
                fila_alumno = None
                for i, fila in enumerate(datos):
                    if len(fila) > 0 and fila[0].strip().lower() == usuario_detectado.lower():
                        fila_alumno = i + 1 # Las celdas en gspread empiezan en 1
                        break

                if fila_alumno:
                    # Anotar "Presente" en la intersección (Fila Alumno, Columna Fecha)
                    hoja.update_cell(fila_alumno, col_fecha, "Presente")
                else:
                    return jsonify({"success": False, "mensaje": f"Alumno '{usuario_detectado}' no está en la lista de Excel."}), 400

            except Exception as e:
                return jsonify({"success": False, "mensaje": f"Error escribiendo en Sheets: {e}"}), 500

        return jsonify({
            "success": True,
            "autorizado": True,
            "usuario": usuario_detectado,
            "precision": precision_obtenida,
            "mensaje": "Asistencia registrada correctamente en el Excel"
        }), 200
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
