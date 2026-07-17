import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# 1. Definir los accesos necesarios
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]

# 2. Cargar tus credenciales
creds = ServiceAccountCredentials.from_json_keyfile_name("credenciales.json", scope)
client = gspread.authorize(creds)

# 3. Conectarse a la hoja (con el nombre exacto corregido)
print("Conectando con Google Sheets...")
sheet = client.open("Registro de Asistencias").sheet1


# 4. Función para registrar los rostros detectados
def registrar_asistencia(nombre_persona):
    try:
        # Obtenemos la fecha y hora actual del sistema
        ahora = datetime.now()
        
        # Mapeo para poner el día de la semana en español
        dias_espanol = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        dia_semana = dias_espanol[ahora.weekday()] # Obtiene el día (0=Lunes, 6=Domingo)
        
        fecha_actual = ahora.strftime("%Y-%m-%d")  # Formato: Año-Mes-Día
        hora_actual = ahora.strftime("%H:%M:%S")     # Formato: Hora:Minutos:Segundos
        
        # Estructura limpia: Nombre, Día, Fecha, Hora
        nueva_fila = [nombre_persona, dia_semana, fecha_actual, hora_actual]
        
        # Agregamos la fila al final de la tabla
        sheet.append_row(nueva_fila)
        
        print(f"✅ ¡Registro exitoso para {nombre_persona} el día {dia_semana}!")
    
    except Exception as e:
        print(f"❌ Error al registrar: {e}")


# --- PRUEBA DE REGISTRO ---
nombre_detectado = "Alexander Windy" # Aquí irá la variable que capture tu app de reconocimiento
registrar_asistencia(nombre_detectado)