import os
import sqlite3
import base64
import json
import threading
from datetime import datetime
import numpy as np
import cv2
import requests
from flask import Flask, request, jsonify, render_template, make_response, session, redirect, url_for
from flask_cors import CORS
from fpdf import FPDF

# Initialize Flask app
app = Flask(__name__)
CORS(app) # Allow external projects to connect to this Flask API
app.secret_key = 'auraguard_secure_key'
detector_lock = threading.Lock()

# Constants
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
DETECTOR_PATH = os.path.join(MODEL_DIR, 'face_detection_yunet_2023mar.onnx')
RECOGNIZER_PATH = os.path.join(MODEL_DIR, 'face_recognition_sface_2021dec.onnx')
DB_PATH = os.path.join(os.path.dirname(__file__), 'attendance.db')

DETECTOR_URL = 'https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx'
RECOGNIZER_URL = 'https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx'

# Global variables for OpenCV models
detector = None
recognizer = None

# SFace cosine similarity threshold (raised to 0.65 for strict security to prevent false positive matches)
MATCH_THRESHOLD = 0.65

# Time-Based Access Control Globals
TIME_CONTROL_ENABLED = False
ACCESS_START_TIME = "09:00"
ACCESS_END_TIME = "17:00"

def download_model(url, path, name):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    if not os.path.exists(path):
        print(f"Downloading {name} model from {url}...")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers=headers, stream=True)
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 32):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = min(100, (downloaded * 100) / total_size) if total_size > 0 else 0
                        print(f"\rDownloading {name}: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end="")
            print(f"\n{name} download complete.")
        except Exception as e:
            print(f"\nFailed to download {name} model: {e}")
            raise RuntimeError(f"Could not download {name}. Error: {e}")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Check if database has old schema by checking if gender column exists in users
    try:
        cursor.execute("SELECT gender FROM users LIMIT 1")
    except sqlite3.OperationalError:
        # Table doesn't exist or doesn't have gender column, recreate tables
        cursor.execute('DROP TABLE IF EXISTS users')
        cursor.execute('DROP TABLE IF EXISTS logs')
        
    # Table for registered users
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            gender TEXT NOT NULL,
            embedding BLOB NOT NULL,
            photo TEXT,
            registered_at TEXT
        )
    ''')
    # Handle schema migration for existing databases
    try:
        cursor.execute('ALTER TABLE users ADD COLUMN photo TEXT')
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute('ALTER TABLE users ADD COLUMN registered_at TEXT')
    except sqlite3.OperationalError:
        pass
        
    # Table for access logs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            gender TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            photo TEXT NOT NULL,
            status TEXT NOT NULL
        )
    ''')
    
    # Table for tasks history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            data_json TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

import csv
def log_to_csv(gender, name, timestamp, status):
    filename = f"{gender.lower()}_attendance.csv" if gender.lower() in ['male', 'female'] else "intruders_attendance.csv"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Name', 'Timestamp', 'Status'])
        writer.writerow([name, timestamp, status])

def load_models():
    global detector, recognizer
    download_model(DETECTOR_URL, DETECTOR_PATH, "YuNet Face Detector")
    download_model(RECOGNIZER_URL, RECOGNIZER_PATH, "SFace Face Recognizer")
    
    print("Loading models into OpenCV DNN...")
    try:
        # Initialize Face Detector (YuNet expects size initially, we set 320x320 and reset on inference)
        detector = cv2.FaceDetectorYN.create(DETECTOR_PATH, "", (320, 320))
        # Initialize Face Recognizer (SFace)
        recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_PATH, "")
        print("Models successfully loaded.")
    except Exception as e:
        print(f"Error initializing OpenCV Face models: {e}")
        detector = None
        recognizer = None

# Initialize database and load models
init_db()
load_models()

@app.route('/')
def index():
    if not session.get('authenticated'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == 'leo':
            session['authenticated'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Access Denied: Incorrect Security Code')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    return redirect(url_for('login'))

@app.route('/register', methods=['POST'])
def register():
    if detector is None or recognizer is None:
        return jsonify({'error': 'Face recognition models are not loaded.'}), 500
        
    data = request.get_json()
    if not data or 'image' not in data or 'name' not in data:
        return jsonify({'error': 'Image data and name are required.'}), 400

    name = data['name'].strip()
    if not name:
        return jsonify({'error': 'Invalid name provided.'}), 400
        
    gender = data.get('gender', 'Male').strip()
    if gender not in ['Male', 'Female']:
        gender = 'Male'

    try:
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image.'}), 400

        # Set detector input size dynamically and detect face (with thread lock to prevent race conditions)
        h, w, _ = img.shape
        with detector_lock:
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
        
        # Filter detected faces to ignore small background false positives or people far away
        valid_faces = []
        if faces is not None:
            for face in faces:
                bbox = face[0:4].astype(int)
                fw, fh = bbox[2], bbox[3]
                if fw >= 110 and fh >= 110: # Only count faces larger than 110x110 pixels
                    valid_faces.append(face)
                    
        if len(valid_faces) == 0:
            return jsonify({'error': 'No face detected in close proximity. Please stand closer to the camera.'}), 400

        if len(valid_faces) > 1:
            return jsonify({'error': 'Multiple faces detected! Please ensure only one person is in the camera view during registration.'}), 400

        # Take the valid face
        face = valid_faces[0]
        
        # Crop and align the face
        aligned_face = recognizer.alignCrop(img, face)
        
        # Crop and resize face crop for registry avatar saving (100x100 to save DB space)
        bbox = face[0:4].astype(int)
        x, y, fw, fh = bbox[0], bbox[1], bbox[2], bbox[3]
        face_crop = img[max(0, y):min(h, y+fh), max(0, x):min(w, x+fw)]
        if face_crop.size > 0:
            face_crop_resized = cv2.resize(face_crop, (100, 100))
            _, reg_buffer = cv2.imencode('.jpg', face_crop_resized)
            reg_photo_base64 = f"data:image/jpeg;base64,{base64.b64encode(reg_buffer).decode('utf-8')}"
        else:
            reg_photo_base64 = data['image'] # Fallback
            
        # Extract features (128-dimensional embedding)
        emb = recognizer.feature(aligned_face)
        
        # Verify embedding integrity (must not be a zero vector or constant dummy values)
        if np.std(emb) < 1e-5:
            return jsonify({'error': 'Face features could not be extracted correctly. Please ensure good lighting and look straight at the camera.'}), 400
            
        emb_bytes = emb.tobytes()

        # Save to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT OR REPLACE INTO users (name, gender, embedding, photo, registered_at) VALUES (?, ?, ?, ?, ?)', 
                           (name, gender, emb_bytes, reg_photo_base64, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
            success = True
        except sqlite3.IntegrityError:
            success = False
        finally:
            conn.close()

        if success:
            return jsonify({'success': True, 'message': f'Face registered successfully as {gender} for {name}!'})
        else:
            return jsonify({'error': 'Integrity error saving to database.'}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server processing error: {str(e)}'}), 500

last_log_times = {}

@app.route('/verify', methods=['POST'])
def verify():
    if TIME_CONTROL_ENABLED:
        now_str = datetime.now().strftime("%H:%M")
        if not (ACCESS_START_TIME <= now_str <= ACCESS_END_TIME):
            return jsonify({
                'error': 'Access Denied: Outside of Allowed Hours.',
                'status': 'Out of Hours'
            }), 403

    if detector is None or recognizer is None:
        return jsonify({'error': 'Face recognition models are not loaded.'}), 500

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided.'}), 400

    try:
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image.'}), 400

        # Set detector input size dynamically and detect faces (with thread lock to prevent race conditions)
        h, w, _ = img.shape
        with detector_lock:
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)

        # Get all registered users from database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT name, gender, embedding FROM users')
        db_users = cursor.fetchall()
        conn.close()

        results = []
        annotated_img = img.copy()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Filter out tiny distant faces that cause false positives due to blur
        valid_faces = []
        if faces is not None:
            for face in faces:
                bbox = face[0:4].astype(int)
                if bbox[2] >= 110 and bbox[3] >= 110:
                    valid_faces.append(face)

        # If no valid faces are detected
        if len(valid_faces) == 0:
            return jsonify({'faces_count': 0, 'results': []})

        for face in valid_faces:
            # Face bounding box details
            bbox = face[0:4].astype(int)
            x, y, fw, fh = bbox[0], bbox[1], bbox[2], bbox[3]

            # Crop and align face
            aligned_face = recognizer.alignCrop(img, face)
            
            # Extract query embedding
            query_emb = recognizer.feature(aligned_face)

            # Verify query embedding integrity (must not be a zero/null vector)
            is_valid_query = np.std(query_emb) >= 1e-5
            if not is_valid_query:
                print("[Warning] Extracted query embedding is empty/invalid. Skipping match check.", flush=True)

            # Crop face crop for thumbnail saving
            face_crop = img[max(0, y):min(h, y+fh), max(0, x):min(w, x+fw)]

            best_match_name = "Unknown Intruder"
            best_match_gender = "Unknown"
            best_match_score = -1.0
            is_authorized = False

            # Match against database users
            if is_valid_query:
                for name, gender, emb_bytes in db_users:
                    # Reconstruct embedding numpy array (SFace generates float32 1x128 array)
                    db_emb = np.frombuffer(emb_bytes, dtype=np.float32).reshape(1, 128)
                    
                    # Verify database embedding integrity
                    if np.std(db_emb) < 1e-5:
                        print(f"[Warning] Database profile for {name} has a corrupted zero embedding. Skipping.", flush=True)
                        continue
                    
                    # SFace match returns similarity score
                    score = recognizer.match(query_emb, db_emb, cv2.FaceRecognizerSF_FR_COSINE)
                    print(f"[Face Match Check] Candidate: {name} | Computed Cosine Similarity: {score:.4f} (Threshold: {MATCH_THRESHOLD})", flush=True)
                    
                    if score > best_match_score:
                        best_match_score = float(score)
                        best_match_name = name
                        best_match_gender = gender

            # Determine authorization status
            if best_match_score >= MATCH_THRESHOLD:
                is_authorized = True
                print(f"[Access Granted] Matched {best_match_name} with score {best_match_score:.4f}", flush=True)
                session['user_name'] = best_match_name
                session['gender'] = best_match_gender
            else:
                best_match_name = "Unknown Intruder"
                best_match_gender = "Unknown"
                print(f"[Access Denied] Best score was only {best_match_score:.4f} (Under threshold {MATCH_THRESHOLD})", flush=True)

            status = "Authorized" if is_authorized else "Access Denied"
            
            # Prepare logs parameters
            # Crop checking face to save as log photo thumbnail (to save DB space, resize it to 80x80)
            if face_crop.size > 0:
                face_crop_resized = cv2.resize(face_crop, (80, 80))
                _, log_buffer = cv2.imencode('.jpg', face_crop_resized)
                log_photo_base64 = f"data:image/jpeg;base64,{base64.b64encode(log_buffer).decode('utf-8')}"
            else:
                log_photo_base64 = data['image'] # Fallback to original image if crop failed

            # Log check-in in SQLite with 5-second cooldown to avoid duplicate log spam
            import time
            curr_time = time.time()
            last_logged = last_log_times.get(best_match_name, 0)
            if curr_time - last_logged > 5.0:
                last_log_times[best_match_name] = curr_time
                
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute('INSERT INTO logs (name, gender, timestamp, photo, status) VALUES (?, ?, ?, ?, ?)',
                               (best_match_name, best_match_gender, timestamp, log_photo_base64, status))
                conn.commit()
                conn.close()

                # Append to separate CSV files
                log_to_csv(best_match_gender, best_match_name, timestamp, status)

            # Bounding box drawing colors (BGR)
            box_color = (74, 222, 128) if is_authorized else (239, 68, 68) # Green if authorized, Red if unknown

            # Draw glowing bounding boxes
            length = int(fw * 0.15)
            cv2.rectangle(annotated_img, (x, y), (x+fw, y+fh), box_color, 1)
            # Corner design
            cv2.line(annotated_img, (x, y), (x + length, y), box_color, 4)
            cv2.line(annotated_img, (x, y), (x, y + length), box_color, 4)
            cv2.line(annotated_img, (x+fw, y), (x+fw - length, y), box_color, 4)
            cv2.line(annotated_img, (x+fw, y), (x+fw, y + length), box_color, 4)
            cv2.line(annotated_img, (x, y+fh), (x + length, y+fh), box_color, 4)
            cv2.line(annotated_img, (x, y+fh), (x, y+fh - length), box_color, 4)
            cv2.line(annotated_img, (x+fw, y+fh), (x+fw - length, y+fh), box_color, 4)
            cv2.line(annotated_img, (x+fw, y+fh), (x+fw, y+fh - length), box_color, 4)

            # Draw Name Tag banner
            label = f"{best_match_name} ({best_match_gender})" if is_authorized else "Intruder Alert!"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_img, (x, y - text_size[1] - 10), (x + text_size[0] + 10, y), box_color, -1)
            cv2.putText(annotated_img, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            results.append({
                'name': best_match_name,
                'gender': best_match_gender,
                'score': best_match_score,
                'is_authorized': is_authorized,
                'status': status
            })

        # Encode annotated image to return
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        annotated_data_url = f"data:image/jpeg;base64,{annotated_base64}"

        # Get gate status: Unlock gate if at least one authorized person is detected
        gate_unlocked = any(r['is_authorized'] for r in results)
        unlock_name = next((r['name'] for r in results if r['is_authorized']), None)
        unlock_gender = next((r['gender'] for r in results if r['is_authorized']), None)

        return jsonify({
            'faces_count': len(results),
            'results': results,
            'gate_unlocked': gate_unlocked,
            'unlock_name': unlock_name,
            'unlock_gender': unlock_gender,
            'annotated_image': annotated_data_url
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server processing error: {str(e)}'}), 500

@app.route('/logs', methods=['GET'])
def get_logs():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT name, gender, timestamp, photo, status FROM logs ORDER BY id DESC LIMIT 50')
    records = cursor.fetchall()
    conn.close()

    logs_list = []
    for r in records:
        logs_list.append({
            'name': r[0],
            'gender': r[1],
            'timestamp': r[2],
            'photo': r[3],
            'status': r[4]
        })
    return jsonify(logs_list)

@app.route('/users', methods=['GET'])
def get_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT name, gender, photo, registered_at FROM users ORDER BY name ASC')
    records = cursor.fetchall()
    conn.close()
    
    users = []
    for r in records:
        users.append({
            'name': r[0],
            'gender': r[1],
            'photo': r[2] or '',
            'registered_at': r[3] or 'N/A'
        })
    return jsonify(users)

@app.route('/users/<name>', methods=['DELETE'])
def delete_user(name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE name = ?', (name,))
    
    # Table for tasks history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            data_json TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': f'User {name} deleted successfully.'})


@app.route('/dashboard')
def dashboard():
    if not session.get('user_name'):
        return redirect(url_for('index'))
    return render_template('dashboard.html', user_name=session.get('user_name'))

@app.route('/get_task_data', methods=['GET'])
def get_task_data():
    if not session.get('user_name'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    date_key = request.args.get('date')
    name = session.get('user_name')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT data_json FROM tasks WHERE name = ? AND date = ?', (name, date_key))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return jsonify({'value': row[0]})
    else:
        return jsonify({'value': None})

@app.route('/submit_task', methods=['POST'])
def submit_task():
    if not session.get('user_name'):
        return jsonify({'error': 'Unauthorized'}), 401
        
    data = request.get_json()
    date_key = data.get('date')
    data_json = data.get('data_json')
    name = session.get('user_name')
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id FROM tasks WHERE name = ? AND date = ?', (name, date_key))
    row = cursor.fetchone()
    if row:
        cursor.execute('UPDATE tasks SET data_json = ? WHERE id = ?', (data_json, row[0]))
    else:
        cursor.execute('INSERT INTO tasks (name, date, data_json) VALUES (?, ?, ?)', (name, date_key, data_json))
        
    conn.commit()
    conn.close()
    return jsonify({'success': True})
    
@app.route('/history_keys', methods=['GET'])
def history_keys():
    if not session.get('user_name'):
        return jsonify({'error': 'Unauthorized'}), 401
    name = session.get('user_name')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT date FROM tasks WHERE name = ?', (name,))
    rows = cursor.fetchall()
    conn.close()
    keys = [f"day:{r[0]}" for r in rows]
    return jsonify({'keys': keys})

class TaskHistoryPDF(FPDF):
    def header(self):
        self.set_fill_color(30, 41, 59)
        self.rect(0, 0, 210, 30, 'F')
        self.set_font('helvetica', 'B', 18)
        self.set_text_color(255, 255, 255)
        self.set_y(12)
        self.cell(0, 10, 'DAILY CARE TASK REPORT', new_x="LMARGIN", new_y="NEXT", align='C')
        self.ln(15)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

@app.route('/export_task_pdf', methods=['GET'])
def export_task_pdf():
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT name, date, data_json FROM tasks ORDER BY date DESC, id DESC')
    records = cursor.fetchall()
    conn.close()
    
    pdf = TaskHistoryPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('helvetica', '', 10)
    
    for r in records:
        name, date_val, data_json = r
        pdf.set_font('helvetica', 'B', 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, f"Date: {date_val} | Name: {name}", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font('helvetica', '', 10)
        try:
            parsed = json.loads(data_json)
            for period, items in parsed.items():
                pdf.set_font('helvetica', 'B', 10)
                pdf.cell(0, 8, f"  Period: {period.capitalize()}", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font('helvetica', '', 10)
                for item_key, item_val in items.items():
                    if 'done' in item_val:
                        val_str = 'Yes' if item_val['done'] else 'No'
                    else:
                        val_str = item_val.get('text', '')
                    if val_str:
                        # Replace troublesome unicode characters
                        val_str = val_str.encode('latin-1', 'replace').decode('latin-1')
                        item_key_safe = item_key.encode('latin-1', 'replace').decode('latin-1')
                        pdf.cell(0, 6, f"    - {item_key_safe}: {val_str}", new_x="LMARGIN", new_y="NEXT")
        except:
            pdf.cell(0, 6, "    (Error parsing data)", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
    response = make_response(bytes(pdf.output()))
    response.headers['Content-Type'] = 'application/pdf'
    fn = f"task_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    response.headers['Content-Disposition'] = f'attachment; filename={fn}'
    return response

@app.route('/clear_logs',
 methods=['POST'])
def clear_logs():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM logs')
    
    # Table for tasks history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            data_json TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'All logs cleared successfully.'})

class AttendancePDF(FPDF):
    def header(self):
        # Professional Header Banner
        self.set_fill_color(30, 41, 59) # Dark slate background
        self.rect(0, 0, 210, 30, 'F')
        
        self.set_font('helvetica', 'B', 18)
        self.set_text_color(255, 255, 255)
        self.set_y(12)
        self.cell(0, 10, 'AURA GUARD SECURITY REPORT', new_x="LMARGIN", new_y="NEXT", align='C')
        
        self.set_font('helvetica', 'I', 10)
        self.set_text_color(148, 163, 184)
        self.cell(0, 5, 'Biometric Access Control & Attendance System', new_x="LMARGIN", new_y="NEXT", align='C')
        
        self.ln(15)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} - Secured by Aura Guard', align='C')

@app.route('/export_pdf', methods=['GET'])
def export_pdf():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Fetch all logs (Boys, Girls, and Intruders) chronologically in one report
    cursor.execute('SELECT id, name, gender, timestamp, status FROM logs ORDER BY id DESC')
    records = cursor.fetchall()
    conn.close()
    
    # Generate PDF using fpdf2
    pdf = AttendancePDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('helvetica', '', 10)
    
    # Professional Metadata Box
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 40, 190, 25, 'DF')
    
    pdf.set_xy(15, 43)
    pdf.set_text_color(71, 85, 105)
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(35, 6, "REPORT DATE:")
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 6, datetime.now().strftime('%B %d, %Y - %H:%M:%S'), new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_x(15)
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(35, 6, "SYSTEM SCOPE:")
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 6, "Consolidated Biometric Logs (Male, Female & Intruders)", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_x(15)
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(35, 6, "TOTAL RECORDS:")
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 6, str(len(records)), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(12)
    
    # Table Headers
    pdf.set_font('helvetica', 'B', 10)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(51, 65, 85) # Slate-700
    pdf.set_draw_color(71, 85, 105) # Slate-600 border
    
    pdf.cell(20, 10, 'LOG ID', border=1, fill=True, align='C')
    pdf.cell(60, 10, 'IDENTITY PROFILE', border=1, fill=True, align='C')
    pdf.cell(25, 10, 'GENDER', border=1, fill=True, align='C')
    pdf.cell(45, 10, 'CAPTURE TIME', border=1, fill=True, align='C')
    pdf.cell(40, 10, 'SECURITY CLEARANCE', border=1, fill=True, align='C', new_x="LMARGIN", new_y="NEXT")
    
    # Table Rows
    pdf.set_font('helvetica', '', 9)
    pdf.set_draw_color(226, 232, 240) # Light gray borders
    
    fill = False
    for r in records:
        pdf.set_fill_color(248, 250, 252) # Subtle zebra striping
        pdf.set_text_color(15, 23, 42)
        
        rec_id, name, gender, timestamp, status = r
        pdf.cell(20, 9, str(rec_id), border=1, align='C', fill=fill)
        pdf.cell(60, 9, f" {name}", border=1, fill=fill)
        pdf.cell(25, 9, gender, border=1, align='C', fill=fill)
        pdf.cell(45, 9, timestamp, border=1, align='C', fill=fill)
        
        # Color coding status
        if status == 'Authorized':
            pdf.set_text_color(22, 163, 74) # Professional green
            status_text = "ACCESS GRANTED"
        else:
            pdf.set_text_color(220, 38, 38) # Professional red
            status_text = "ACCESS DENIED"
            
        pdf.set_font('helvetica', 'B', 9)
        pdf.cell(40, 9, status_text, border=1, align='C', fill=fill, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font('helvetica', '', 9)
        fill = not fill
        pdf.set_text_color(40, 40, 40) # Reset color
        
    response = make_response(bytes(pdf.output()))
    response.headers['Content-Type'] = 'application/pdf'
    
    # Setup download filename
    fn = f"master_attendance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    response.headers['Content-Disposition'] = f'attachment; filename={fn}'
    return response

@app.route('/get_time_config', methods=['GET'])
def get_time_config():
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify({
        'enabled': TIME_CONTROL_ENABLED,
        'start_time': ACCESS_START_TIME,
        'end_time': ACCESS_END_TIME
    })

@app.route('/set_time_config', methods=['POST'])
def set_time_config():
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    global TIME_CONTROL_ENABLED, ACCESS_START_TIME, ACCESS_END_TIME
    data = request.json
    TIME_CONTROL_ENABLED = data.get('enabled', False)
    ACCESS_START_TIME = data.get('start_time', '09:00')
    ACCESS_END_TIME = data.get('end_time', '17:00')
    
    return jsonify({'message': 'Time configuration saved successfully!'})

if __name__ == '__main__':
    # Run the server on all interfaces so it can be accessed over local network
    app.run(host='0.0.0.0', port=5002, debug=True)
