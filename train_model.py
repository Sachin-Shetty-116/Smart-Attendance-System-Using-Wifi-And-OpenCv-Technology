"""
train_model.py - Standalone Model Training Script
Wi-Fi Based Smart Attendance System Using OpenCV and Face Recognition

Run this script directly to retrain the face recognition model:
    python train_model.py
"""

import sys
import os

# Ensure project root is on the path when run as a standalone script
sys.path.insert(0, os.path.dirname(__file__))

from face_utils import train_recognizer, is_model_trained, MODEL_FILE
from database import init_db, get_all_students


def main():
    print("=" * 55)
    print("  Wi-Fi Smart Attendance – Face Recognition Trainer")
    print("=" * 55)

    # Ensure the DB is initialised
    init_db()

    students = get_all_students()
    if not students:
        print("\n[!] No students registered in the database.")
        print("    Register students via the web dashboard first.")
        sys.exit(1)

    print(f"\n[INFO] Registered students: {len(students)}")
    for s in students:
        print(f"       - {s['name']} ({s['roll_number']}) | "
              f"Images: {s['image_count']}")

    print("\n[INFO] Starting training...")
    result = train_recognizer()

    if result["success"]:
        print(f"\n[OK] {result['message']}")
        print(f"     Model saved to: {MODEL_FILE}")
    else:
        print(f"\n[FAIL] Training failed: {result['message']}")
        sys.exit(1)

    print("\n[INFO] Training complete. You can now start attendance monitoring.")
    print("=" * 55)


if __name__ == "__main__":
    main()
