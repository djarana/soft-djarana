import cv2
import face_recognition
import numpy as np
import mysql.connector
import pickle
from mysql.connector import Binary
from datetime import datetime

# --- Configuración base de datos ---
DB_CONFIG = {
    'host': '192.168.1.222',
    'user': 'williams',
    'password': '0987654321',
    'database': 'reconocimiento_facial',
    'charset': 'latin1',
    'collation': 'latin1_general_ci',
    'use_unicode': False,
    'raise_on_warnings': True
}

# --- Conectar a la base de datos ---
def conectar_db():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        print(f"Error de conexión: {e}")
        return None

# --- Guardar rostro ---
def guardar_rostro(nombre, embedding):
    if not isinstance(embedding, (bytes, bytearray)):
        print("Embedding inválido, no es binario")
        return

    try:
        conn = conectar_db()
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO known_faces (name, embedding, date_added) VALUES (%s, %s, %s)",
                (nombre, Binary(embedding), datetime.now())
            )
            conn.commit()
            print(f"Rostro '{nombre}' guardado en la base de datos.")
    except mysql.connector.Error as e:
        print(f"Error al guardar rostro: {e}")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

# --- Cargar rostros conocidos (una vez) ---
def cargar_rostros():
    rostros = []
    conn = conectar_db()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name, embedding FROM known_faces")
        for nombre, emb in cursor.fetchall():
            nombre = nombre or "SinNombre"
            vector = pickle.loads(emb)
            rostros.append((nombre, vector))
        cursor.close()
        conn.close()
    return rostros

# --- Extraer embedding facial ---
def extraer_embedding(imagen):
    rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if encodings:
        return pickle.dumps(encodings[0])  # guardar en binario
    return None

# --- Reconocimiento facial en tiempo real ---
def reconocer_rostros():
    conocidos = cargar_rostros()
    known_names = [n for n, _ in conocidos]
    known_encodings = [e for _, e in conocidos]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            nombre = "Desconocido"
            matches = face_recognition.compare_faces(known_encodings, encoding)
            face_distances = face_recognition.face_distance(known_encodings, encoding)

            if True in matches:
                best_match_index = np.argmin(face_distances)
                nombre = known_names[best_match_index]

            color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            #cv2.putText(frame, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, str(nombre), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Reconocimiento Facial", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s') and face_encodings:
            nombre = input("Nombre del nuevo rostro: ").strip()
            if nombre:
                emb_bin = pickle.dumps(face_encodings[0])
                guardar_rostro(nombre, emb_bin)

                # Añadir a memoria local sin recargar toda la BD
                conocidos.append((nombre, face_encodings[0]))
                known_names.append(nombre)
                known_encodings.append(face_encodings[0])
                print("Nuevo rostro agregado a memoria.")
            else:
                print("Nombre inválido.")

    cap.release()
    cv2.destroyAllWindows()

# --- Ejecutar ---
if __name__ == "__main__":
    reconocer_rostros()
