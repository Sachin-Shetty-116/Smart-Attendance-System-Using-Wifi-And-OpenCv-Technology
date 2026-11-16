"""
camera.py - Webcam Capture and Real-Time Face Recognition
Wi-Fi Based Smart Attendance System Using OpenCV and Face Recognition

This module manages the webcam stream and performs real-time face detection
and recognition.  It runs in a background thread so that multiple Flask
routes can read frames without blocking.
"""

import cv2
import threading
import time
import numpy as np
import requests
import os
from datetime import datetime

from face_utils import (
    detect_faces,
    load_recognizer,
    recognize_face,
    draw_face_box,
    CONFIDENCE_THRESHOLD,
)
from database import get_student_by_label, is_already_marked

# ──────────────────────────── Globals ────────────────────────────

_lock          = threading.Lock()
_output_frame  = None        # Latest JPEG-encoded frame
_camera_active = False       # Whether the camera loop is running
_camera_thread = None        # Background thread reference
_recognizer    = None        # Loaded LBPH recognizer (or None)
_current_session = "Morning" # Current attendance session name

# Recognition cool-down per student to avoid rapid duplicate HTTP calls
# { student_id: last_marked_timestamp }
_mark_cooldown: dict[int, float] = {}
COOLDOWN_SECONDS = 5  # Minimum seconds between successive HTTP mark calls


# ──────────────────────────── Frame Generator ────────────────────────────

def generate_frames():
    """
    Generator that yields MJPEG frames for Flask's streaming response.
    Called by the /video_feed route.
    """
    global _output_frame
    while True:
        with _lock:
            if _output_frame is None:
                # Provide a blank "No Camera" image when camera is off
                blank = _build_no_camera_frame()
                _, buf = cv2.imencode(".jpg", blank)
                frame_bytes = buf.tobytes()
            else:
                frame_bytes = _output_frame

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )
        time.sleep(0.04)  # ~25 fps


def _build_no_camera_frame() -> np.ndarray:
    """Return a dark placeholder frame with a 'Camera Offline' message."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 30, 50)  # Dark navy background
    msg1 = "Camera Offline"
    msg2 = "Start attendance monitoring to enable the camera"
    cv2.putText(frame, msg1, (170, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 255), 2)
    cv2.putText(frame, msg2, (50, 265),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    return frame


# ──────────────────────────── Camera Thread ────────────────────────────

def start_camera(session_name: str = "Morning",
                 flask_base_url: str = "http://127.0.0.1:5000") -> dict:
    """
    Start the background camera thread.
    Returns a dict with 'success' and 'message'.
    """
    global _camera_active, _camera_thread, _recognizer, _current_session

    if _camera_active:
        return {"success": False, "message": "Camera is already running."}

    # Try to load the recognition model
    try:
        _recognizer = load_recognizer()
    except FileNotFoundError as e:
        _recognizer = None
        print(f"[CAMERA] Warning: {e}  – Running in detection-only mode.")

    _current_session = session_name
    _camera_active   = True
    _camera_thread   = threading.Thread(
        target=_camera_loop,
        args=(flask_base_url,),
        daemon=True
    )
    _camera_thread.start()
    return {"success": True, "message": "Camera started successfully."}


def stop_camera() -> dict:
    """Signal the camera thread to stop and wait for it to finish."""
    global _camera_active, _camera_thread, _output_frame

    _camera_active = False
    if _camera_thread and _camera_thread.is_alive():
        _camera_thread.join(timeout=5)
    _camera_thread = None

    with _lock:
        _output_frame = None

    return {"success": True, "message": "Camera stopped."}


def is_camera_running() -> bool:
    """Return True if the camera background thread is active."""
    return _camera_active


# ──────────────────────────── Main Loop ────────────────────────────

def _camera_loop(flask_base_url: str):
    """
    Background thread: reads webcam frames, runs face detection/recognition,
    annotates frames, and sends attendance HTTP requests to the Flask server.
    """
    global _output_frame, _camera_active

    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        # Try index 1 (external webcam) as fallback
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[CAMERA] ERROR: Could not open any webcam.")
        _camera_active = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[CAMERA] Webcam opened. Starting recognition loop…")

    while _camera_active:
        ret, frame = cap.read()
        if not ret:
            print("[CAMERA] Failed to read frame. Retrying…")
            time.sleep(0.5)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Enhance contrast for better detection under poor lighting
        gray = cv2.equalizeHist(gray)

        faces = detect_faces(gray)

        for (x, y, w, h) in faces:
            face_roi_gray = gray[y:y + h, x:x + w]

            if _recognizer is not None:
                label_id, confidence = recognize_face(_recognizer, face_roi_gray)

                if confidence < CONFIDENCE_THRESHOLD:
                    # ── Recognized student ──
                    student = get_student_by_label(label_id)
                    if student:
                        display_name = f"{student['name']} [{student['roll_number']}]"
                        draw_face_box(frame, x, y, w, h, display_name,
                                      confidence, recognized=True)
                        _try_mark_attendance(
                            student, flask_base_url, confidence
                        )
                    else:
                        draw_face_box(frame, x, y, w, h, "Unknown",
                                      confidence, recognized=False)
                else:
                    draw_face_box(frame, x, y, w, h, "Unknown",
                                  confidence, recognized=False)
            else:
                # Detection-only mode (no model trained)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 2)
                cv2.putText(frame, "Face Detected – No Model",
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 165, 0), 1)

        # Overlay status bar at the top
        _draw_status_bar(frame)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with _lock:
            _output_frame = buf.tobytes()

    cap.release()
    print("[CAMERA] Webcam released.")


# ──────────────────────────── Helpers ────────────────────────────

def _try_mark_attendance(student, flask_base_url: str, confidence: float):
    """
    Send an HTTP POST to the Flask backend to mark attendance.
    Uses a cooldown to avoid hammering the server for every frame.
    """
    student_id = student["id"]
    now = time.time()

    # Cooldown check
    last_marked = _mark_cooldown.get(student_id, 0)
    if now - last_marked < COOLDOWN_SECONDS:
        return

    _mark_cooldown[student_id] = now

    try:
        resp = requests.post(
            f"{flask_base_url}/api/mark_attendance",
            json={
                "student_id":  student_id,
                "roll_number": student["roll_number"],
                "name":        student["name"],
                "section":     student["section"],
                "session":     _current_session,
                "confidence":  round(float(confidence), 2)
            },
            timeout=3
        )
        data = resp.json()
        if data.get("marked"):
            print(f"[CAMERA] [OK] Attendance marked for {student['name']}")
        else:
            print(f"[CAMERA] [--] Already marked for {student['name']}")
    except requests.exceptions.RequestException as e:
        print(f"[CAMERA] [!!] Could not reach server: {e}")


def _draw_status_bar(frame: np.ndarray):
    """Overlay a semi-transparent status bar at the top of the frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0, 0), (w, 30), (15, 15, 40), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    status = "Recognition: ACTIVE" if _recognizer else "Recognition: NO MODEL"
    color  = (0, 220, 100) if _recognizer else (0, 140, 255)
    now_str = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, status, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    cv2.putText(frame, now_str, (w - 80, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
