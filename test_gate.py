import os
import sys
import numpy as np
import cv2

def test_face_recognition_pipeline():
    print("Testing face recognition pipeline dependencies...")
    try:
        import requests
        import flask
        import sqlite3
        print("SUCCESS: Core dependencies are imported!")
    except ImportError as e:
        print(f"FAILED: Dependency import failed: {e}")
        return False

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'model')
    detector_path = os.path.join(model_dir, 'face_detection_yunet_2023mar.onnx')
    recognizer_path = os.path.join(model_dir, 'face_recognition_sface_2021dec.onnx')

    # Trigger model download
    print("Checking model files status...")
    try:
        from app import load_models
        load_models()
    except Exception as e:
        print(f"FAILED during load_models() execution: {e}")
        return False

    if not os.path.exists(detector_path):
        print(f"FAILED: YuNet face detector model not found at {detector_path}")
        return False
    if not os.path.exists(recognizer_path):
        print(f"FAILED: SFace face recognizer model not found at {recognizer_path}")
        return False

    print("SUCCESS: Model files successfully verified on disk.")

    # Test running face detector and recognizer on dummy image
    try:
        print("Initializing face detector and recognizer...")
        detector = cv2.FaceDetectorYN.create(detector_path, "", (320, 240))
        recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")
        
        # Create a dummy image
        print("Creating mock image and testing face detection run...")
        dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Run detection (should return 0 faces, but run without throwing errors)
        _, faces = detector.detect(dummy_img)
        print("Face detection run completed successfully.")
        
        if faces is None:
            print("No faces detected in black image (expected result).")
        else:
            print(f"Detected {len(faces)} faces (unexpected, but detector executed).")

        # Database connection test
        print("Testing SQLite database initialization...")
        db_path = os.path.join(current_dir, 'attendance.db')
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            conn.close()
            print("Tables found in database:", [t[0] for t in tables])
            if 'users' in [t[0] for t in tables] and 'logs' in [t[0] for t in tables]:
                print("SUCCESS: SQLite tables validated!")
            else:
                print("FAILED: SQLite tables missing.")
                return False
        else:
            print("FAILED: SQLite database file not created.")
            return False

        print("SUCCESS: Face recognition system tests completed successfully!")
        return True

    except Exception as e:
        print(f"FAILED during pipeline execution test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_face_recognition_pipeline()
    sys.exit(0 if success else 1)
