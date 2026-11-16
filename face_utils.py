"""
face_utils.py - Face Detection and Recognition Utilities
Wi-Fi Based Smart Attendance System Using OpenCV and Face Recognition
"""

import cv2
import os
import numpy as np

# ──────────────────────────── Paths ────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")    # student face images
MODEL_DIR   = os.path.join(BASE_DIR, "models")     # trained recognizer
MODEL_FILE  = os.path.join(MODEL_DIR, "face_model.yml")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Create directories if they don't exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)

# ──────────────────────────── Constants ────────────────────────────
CONFIDENCE_THRESHOLD = 80   # Lower = stricter recognition
MIN_IMAGES_TO_TRAIN  = 30   # Minimum face images per student


# ──────────────────────────── Detector ────────────────────────────

def load_face_detector():
    """Load and return the Haar Cascade face detector."""
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(
            f"[FACE] Could not load Haar Cascade from: {CASCADE_PATH}"
        )
    return detector


FACE_DETECTOR = load_face_detector()


def detect_faces(gray_frame: np.ndarray):
    """
    Detect faces in a grayscale frame.
    Returns a list of (x, y, w, h) bounding boxes.
    """
    faces = FACE_DETECTOR.detectMultiScale(
        gray_frame,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces if len(faces) > 0 else []


# ──────────────────────────── Dataset ────────────────────────────

def get_student_dataset_dir(face_label: int) -> str:
    """Return the directory path for a student's face images."""
    path = os.path.join(DATASET_DIR, f"student_{face_label}")
    os.makedirs(path, exist_ok=True)
    return path


def save_face_image(face_roi: np.ndarray, face_label: int, img_index: int) -> str:
    """
    Save a single face ROI (region of interest) to the student's dataset folder.
    Returns the saved file path.
    """
    folder   = get_student_dataset_dir(face_label)
    filename = f"face_{img_index}.jpg"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, face_roi)
    return filepath


def count_student_images(face_label: int) -> int:
    """Return the number of saved face images for a student."""
    folder = get_student_dataset_dir(face_label)
    return len([f for f in os.listdir(folder) if f.endswith(".jpg")])


def delete_student_images(face_label: int) -> bool:
    """
    Delete all face images for a student.
    Returns True if the folder existed and was removed.
    """
    import shutil
    folder = get_student_dataset_dir(face_label)
    if os.path.exists(folder):
        shutil.rmtree(folder)
        return True
    return False


# ──────────────────────────── Training ────────────────────────────

def load_training_data():
    """
    Walk through the dataset directory, read face images, and build
    (faces, labels) arrays required by the LBPH recognizer.
    Returns (face_samples, labels, label_map) where label_map maps
    integer label → student folder name.
    """
    face_samples = []
    labels       = []
    label_map    = {}

    if not os.path.exists(DATASET_DIR):
        return face_samples, labels, label_map

    for folder_name in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Expect folder names like "student_1", "student_2", …
        try:
            label = int(folder_name.split("_")[1])
        except (IndexError, ValueError):
            continue

        label_map[label] = folder_name

        for img_name in os.listdir(folder_path):
            if not img_name.endswith(".jpg"):
                continue
            img_path = os.path.join(folder_path, img_name)
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                continue
            # Resize for consistency
            img_resized = cv2.resize(img_gray, (200, 200))
            face_samples.append(img_resized)
            labels.append(label)

    return face_samples, labels, label_map


def train_recognizer() -> dict:
    """
    Train an LBPH face recognizer on all images in the dataset directory.
    Saves the trained model to MODEL_FILE.
    Returns a result dict with keys: success, message, student_count, image_count.
    """
    face_samples, labels, label_map = load_training_data()

    if len(face_samples) == 0:
        return {
            "success": False,
            "message": "No training images found. Register students first.",
            "student_count": 0,
            "image_count": 0
        }

    unique_labels = set(labels)
    student_count = len(unique_labels)

    # LBPH (Local Binary Patterns Histograms) recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
        threshold=CONFIDENCE_THRESHOLD
    )

    recognizer.train(face_samples, np.array(labels))
    recognizer.save(MODEL_FILE)

    return {
        "success": True,
        "message": (
            f"Model trained successfully with {student_count} student(s) "
            f"and {len(face_samples)} image(s)."
        ),
        "student_count": student_count,
        "image_count":   len(face_samples)
    }


def is_model_trained() -> bool:
    """Return True if a trained model file exists."""
    return os.path.exists(MODEL_FILE)


def load_recognizer():
    """
    Load and return the trained LBPH face recognizer.
    Raises FileNotFoundError if the model has not been trained yet.
    """
    if not is_model_trained():
        raise FileNotFoundError(
            "Face recognition model not found. "
            "Please train the model first."
        )
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)
    return recognizer


# ──────────────────────────── Recognition ────────────────────────────

def recognize_face(recognizer, face_roi: np.ndarray):
    """
    Predict the face label and confidence for a given face ROI.
    Returns (label, confidence).
    A confidence below CONFIDENCE_THRESHOLD means a known student.
    """
    face_resized = cv2.resize(face_roi, (200, 200))
    label, confidence = recognizer.predict(face_resized)
    return label, confidence


def draw_face_box(frame: np.ndarray, x: int, y: int, w: int, h: int,
                  label: str, confidence: float, recognized: bool):
    """
    Draw a bounding box and label on the frame.
    Green = recognized student, Red = unknown.
    """
    color = (0, 200, 0) if recognized else (0, 0, 220)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Background for the label text
    text = f"{label} ({confidence:.1f})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 6, y), color, -1)
    cv2.putText(
        frame, text, (x + 3, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )
    return frame


# ──────────────────────────── Anti-Spoofing Notice ────────────────────────────
ANTI_SPOOFING_NOTICE = (
    "NOTICE: This system uses 2D face detection (Haar Cascade) and LBPH "
    "recognition. It does NOT include liveness detection. A printed photo or "
    "screen image may be recognized. For production use, add a liveness check "
    "(e.g., blink detection, 3D depth sensor, or a dedicated anti-spoofing model)."
)
