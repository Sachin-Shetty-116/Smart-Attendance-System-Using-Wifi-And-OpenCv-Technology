"""
app.py - Flask Backend for Smart Attendance System
Wi-Fi Based Smart Attendance System Using OpenCV and Face Recognition

Run with:
    python app.py
Then open  http://<your-local-ip>:5000  on any device on the same Wi-Fi.
"""

import os
import io
import csv
import socket
import secrets
from datetime import date, datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, Response, send_file
)

# Project modules
from database import (
    init_db, verify_admin, change_admin_password,
    add_student, get_all_students, get_student_by_roll, get_next_face_label,
    mark_attendance, get_todays_attendance, get_attendance_by_date,
    get_attendance_report, get_dashboard_stats, get_all_sections,
    delete_student, get_student_by_label, update_student_image_count,
    is_already_marked
)
from face_utils import (
    detect_faces, save_face_image, count_student_images,
    delete_student_images, train_recognizer, is_model_trained,
    ANTI_SPOOFING_NOTICE, MIN_IMAGES_TO_TRAIN
)
import camera as cam

import cv2
import numpy as np

# ──────────────────────────── App Setup ────────────────────────────

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)   # Random secret key each run
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload


def get_local_ip() -> str:
    """Return the machine's local network IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


LOCAL_IP = get_local_ip()

# ──────────────────────────── Auth Decorator ────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ──────────────────────────── Auth Routes ────────────────────────────

@app.route("/", methods=["GET"])
def index():
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if verify_admin(username, password):
            session["logged_in"] = True
            session["username"]  = username
            flash(f"Welcome, {username}!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ──────────────────────────── Dashboard ────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    stats = get_dashboard_stats()
    model_ready = is_model_trained()
    return render_template(
        "dashboard.html",
        stats=stats,
        model_ready=model_ready,
        camera_running=cam.is_camera_running(),
        local_ip=LOCAL_IP,
        anti_spoofing_notice=ANTI_SPOOFING_NOTICE,
        username=session.get("username")
    )


# ──────────────────────────── Student Registration ────────────────────────────

@app.route("/register", methods=["GET", "POST"])
@login_required
def register():
    students = get_all_students()
    if request.method == "POST":
        name        = request.form.get("name", "").strip()
        roll_number = request.form.get("roll_number", "").strip()
        section     = request.form.get("section", "").strip()
        consent     = request.form.get("consent") == "on"

        # Validation
        errors = []
        if not name:        errors.append("Student name is required.")
        if not roll_number: errors.append("Roll number is required.")
        if not section:     errors.append("Section is required.")
        if not consent:     errors.append("Face image consent is required.")
        if get_student_by_roll(roll_number):
            errors.append(f"Roll number '{roll_number}' already registered.")

        if errors:
            for e in errors:
                flash(e, "danger")
            return render_template("register.html", students=students)

        face_label = get_next_face_label()
        student_id = add_student(roll_number, name, section, face_label, consent)

        if student_id is None:
            flash("Registration failed. Roll number may already exist.", "danger")
            return render_template("register.html", students=students)

        flash(
            f"Student '{name}' registered with label #{face_label}. "
            "Now capture face images.",
            "success"
        )
        return redirect(url_for("capture_face", student_id=student_id))

    return render_template("register.html", students=students)


# ──────────────────────────── Face Capture ────────────────────────────

@app.route("/capture/<int:student_id>")
@login_required
def capture_face(student_id: int):
    """Page that streams the webcam for face image capture."""
    student = None
    students = get_all_students()
    for s in students:
        if s["id"] == student_id:
            student = dict(s)
            break
    if student is None:
        flash("Student not found.", "danger")
        return redirect(url_for("register"))

    img_count = count_student_images(student["face_label"])
    return render_template(
        "capture.html",
        student=student,
        img_count=img_count,
        required=MIN_IMAGES_TO_TRAIN
    )


@app.route("/api/capture_frame", methods=["POST"])
@login_required
def capture_frame():
    """
    Receive a Base64-encoded JPEG from the browser, detect faces,
    and save the face ROI to the student's dataset folder.
    """
    data = request.get_json()
    student_id  = data.get("student_id")
    image_data  = data.get("image")  # data:image/jpeg;base64,…

    if not student_id or not image_data:
        return jsonify({"success": False, "message": "Missing data."}), 400

    # Decode base64 image
    try:
        header, encoded = image_data.split(",", 1)
        img_bytes = __import__("base64").b64decode(encoded)
        np_arr    = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray      = cv2.equalizeHist(gray)
    except Exception as e:
        return jsonify({"success": False, "message": f"Image decode error: {e}"}), 400

    faces = detect_faces(gray)
    if len(faces) == 0:
        return jsonify({"success": False, "message": "No face detected. Please look at the camera."}), 200

    # Fetch student from DB
    students = get_all_students()
    student  = next((s for s in students if s["id"] == student_id), None)
    if student is None:
        return jsonify({"success": False, "message": "Student not found."}), 404

    face_label = student["face_label"]
    current_count = count_student_images(face_label)

    # Save the largest face detected
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y + h, x:x + w]
    img_index = current_count + 1
    save_face_image(face_roi, face_label, img_index)

    new_count = count_student_images(face_label)
    update_student_image_count(student_id, new_count)

    return jsonify({
        "success":   True,
        "message":   f"Image {new_count} captured.",
        "count":     new_count,
        "required":  MIN_IMAGES_TO_TRAIN
    })


# ──────────────────────────── Model Training ────────────────────────────

@app.route("/train", methods=["GET", "POST"])
@login_required
def train():
    result = None
    if request.method == "POST":
        result = train_recognizer()
        if result["success"]:
            flash(result["message"], "success")
        else:
            flash(result["message"], "danger")
    return render_template("train.html", result=result,
                           model_ready=is_model_trained())


@app.route("/api/train_model", methods=["POST"])
@login_required
def api_train_model():
    result = train_recognizer()
    return jsonify(result)


# ──────────────────────────── Face Recognition / Camera ────────────────────────────

@app.route("/recognition")
@login_required
def recognition():
    """Face recognition monitoring page."""
    return render_template(
        "recognition.html",
        camera_running=cam.is_camera_running(),
        model_ready=is_model_trained(),
        local_ip=LOCAL_IP
    )


@app.route("/video_feed")
@login_required
def video_feed():
    """MJPEG stream for the recognition page."""
    return Response(
        cam.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/start_camera", methods=["POST"])
@login_required
def api_start_camera():
    data    = request.get_json() or {}
    session_name = data.get("session", "Morning")
    base_url = f"http://127.0.0.1:{request.host.split(':')[-1] if ':' in request.host else 5000}"
    result = cam.start_camera(session_name=session_name, flask_base_url=base_url)
    return jsonify(result)


@app.route("/api/stop_camera", methods=["POST"])
@login_required
def api_stop_camera():
    result = cam.stop_camera()
    return jsonify(result)


@app.route("/api/camera_status")
@login_required
def api_camera_status():
    return jsonify({
        "running":     cam.is_camera_running(),
        "model_ready": is_model_trained()
    })


# ──────────────────────────── Attendance API ────────────────────────────

@app.route("/api/mark_attendance", methods=["POST"])
def api_mark_attendance():
    """
    Called by camera.py via HTTP POST.
    Accessible internally (127.0.0.1) only; camera.py runs on the same machine.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    student_id  = data.get("student_id")
    roll_number = data.get("roll_number")
    name        = data.get("name")
    section     = data.get("section")
    sess        = data.get("session", "Morning")

    if not all([student_id, roll_number, name, section]):
        return jsonify({"error": "Missing required fields"}), 400

    marked = mark_attendance(student_id, roll_number, name, section, sess)
    return jsonify({"marked": marked,
                    "student": name,
                    "roll": roll_number})


@app.route("/api/dashboard_stats")
@login_required
def api_dashboard_stats():
    """Return live dashboard statistics as JSON (polled by JS)."""
    return jsonify(get_dashboard_stats())


# ──────────────────────────── Attendance Records ────────────────────────────

@app.route("/attendance")
@login_required
def attendance():
    selected_date = request.args.get("date", date.today().strftime("%Y-%m-%d"))
    records = get_attendance_by_date(selected_date)
    sections = get_all_sections()
    return render_template(
        "attendance.html",
        records=[dict(r) for r in records],
        selected_date=selected_date,
        sections=sections
    )


# ──────────────────────────── Reports ────────────────────────────

@app.route("/reports")
@login_required
def reports():
    start_date = request.args.get("start_date", "")
    end_date   = request.args.get("end_date",   "")
    section    = request.args.get("section",    "")

    records  = get_attendance_report(
        start_date or None, end_date or None, section or None
    )
    sections = get_all_sections()
    return render_template(
        "reports.html",
        records=[dict(r) for r in records],
        sections=sections,
        start_date=start_date,
        end_date=end_date,
        selected_section=section
    )


@app.route("/export_csv")
@login_required
def export_csv():
    """Export filtered attendance as a downloadable CSV file."""
    start_date = request.args.get("start_date", "")
    end_date   = request.args.get("end_date",   "")
    section    = request.args.get("section",    "")

    records = get_attendance_report(
        start_date or None, end_date or None, section or None
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Roll Number", "Student Name", "Section",
                     "Date", "Time", "Session", "Marked By"])
    for r in records:
        writer.writerow([
            r["id"], r["roll_number"], r["student_name"],
            r["section"], r["date"], r["time"], r["session"], r["marked_by"]
        ])

    output.seek(0)
    filename = f"attendance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ──────────────────────────── Settings ────────────────────────────

@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        action = request.form.get("action")

        if action == "change_password":
            old_pw  = request.form.get("old_password", "")
            new_pw  = request.form.get("new_password", "")
            confirm = request.form.get("confirm_password", "")
            username = session.get("username")

            if not verify_admin(username, old_pw):
                flash("Current password is incorrect.", "danger")
            elif len(new_pw) < 6:
                flash("New password must be at least 6 characters.", "danger")
            elif new_pw != confirm:
                flash("New passwords do not match.", "danger")
            else:
                change_admin_password(username, new_pw)
                flash("Password changed successfully.", "success")

        elif action == "delete_student":
            student_id_str = request.form.get("student_id")
            try:
                student_id = int(student_id_str)
            except (ValueError, TypeError):
                flash("Invalid student ID.", "danger")
                return redirect(url_for("settings"))

            # Find face_label before deletion
            students = get_all_students()
            target = next((s for s in students if s["id"] == student_id), None)
            if target:
                delete_student_images(target["face_label"])
                delete_student(student_id)
                flash(
                    f"Student '{target['name']}' and their face images deleted.",
                    "success"
                )
            else:
                flash("Student not found.", "danger")

    students = get_all_students()
    return render_template(
        "settings.html",
        students=[dict(s) for s in students],
        local_ip=LOCAL_IP,
        model_ready=is_model_trained()
    )


# ──────────────────────────── Entry Point ────────────────────────────

if __name__ == "__main__":
    # Initialise DB on startup
    init_db()

    print("=" * 60)
    print("  Wi-Fi Smart Attendance System")
    print("=" * 60)
    print(f"  Local  URL : http://127.0.0.1:5000")
    print(f"  Network URL: http://{LOCAL_IP}:5000")
    print()
    print("  Share the Network URL with devices on the same Wi-Fi.")
    print("  Default login -> username: admin | password: admin123")
    print("=" * 60)

    # host="0.0.0.0" makes the server reachable on the local network
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
