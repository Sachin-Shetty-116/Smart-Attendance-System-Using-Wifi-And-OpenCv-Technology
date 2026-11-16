# 📡 Wi-Fi Based Smart Attendance System
### Using OpenCV & Face Recognition

> A college-grade academic project demonstrating real-time face recognition, a Flask web dashboard, SQLite storage, and wireless access over a local Wi-Fi network.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [How Wi-Fi and OpenCV Work Together](#how-wi-fi-and-opencv-work-together)
4. [Features](#features)
5. [Project Structure](#project-structure)
6. [Installation Guide](#installation-guide)
7. [Running the Application](#running-the-application)
8. [Usage Workflow](#usage-workflow)
9. [Accessing from Other Devices (Wi-Fi)](#accessing-from-other-devices-wi-fi)
10. [Privacy & Consent](#privacy--consent)
11. [Anti-Spoofing Notice](#anti-spoofing-notice)
12. [Troubleshooting](#troubleshooting)
13. [Future Extensions (ESP32 / IoT)](#future-extensions-esp32--iot)

---

## Project Overview

This system allows a college administrator to:
1. **Register students** with name, roll number, and face images.
2. **Train a face recognition model** using OpenCV's LBPH algorithm.
3. **Monitor attendance** in real time – the webcam automatically detects and recognizes student faces.
4. **View and export reports** from any device connected to the same Wi-Fi network.

---

## Technologies Used

| Component | Technology |
|-----------|-----------|
| Backend   | Python + Flask |
| Face Detection | OpenCV Haar Cascade |
| Face Recognition | OpenCV LBPH Recognizer |
| Database  | SQLite (via Python's `sqlite3`) |
| Frontend  | HTML5, CSS3, Bootstrap 5, JavaScript |
| Wireless  | Wi-Fi (local network HTTP via Flask) |
| Camera    | Laptop webcam or USB webcam (via OpenCV) |

---

## How Wi-Fi and OpenCV Work Together

These two technologies serve **completely different purposes** and complement each other:

```
┌─────────────────────────────────────────────────────────┐
│                    HOST COMPUTER                        │
│                                                         │
│  ┌──────────────────┐    ┌──────────────────────────┐  │
│  │  OpenCV Camera   │───▶│  Flask Server (:5000)    │  │
│  │  (face detect /  │    │  • stores attendance in  │  │
│  │   recognition)   │    │    SQLite                │  │
│  └──────────────────┘    │  • serves web dashboard  │  │
│                           └────────────┬─────────────┘  │
└────────────────────────────────────────┼────────────────┘
                                         │  HTTP over Wi-Fi
                              ┌──────────▼──────────────┐
                              │  Mobile Phone / Laptop  │
                              │  Browser → Dashboard    │
                              └─────────────────────────┘
```

- **OpenCV** performs face detection and recognition **locally on the host machine**.
- **Wi-Fi** provides the wireless network layer that lets other devices access the web dashboard hosted by Flask.
- They are **independent layers**: OpenCV doesn't use Wi-Fi, and the Wi-Fi dashboard doesn't touch the camera.

---

## Features

### ✅ Student Registration
- Capture 30+ face images per student via the browser webcam (JavaScript `getUserMedia`).
- Stores roll number, name, section, and face label in SQLite.
- Consent checkbox required before capture (privacy compliance).

### ✅ Face Recognition Attendance
- Real-time MJPEG stream in the browser.
- LBPH recognizer identifies registered students.
- Green box = recognized; Red box = unknown.
- Auto-marks attendance via internal HTTP POST; prevents duplicates per session/day.

### ✅ Wi-Fi Dashboard Access
- Flask runs with `host="0.0.0.0"` so any device on the LAN can connect.
- Dashboard auto-refreshes stats every 8 seconds.

### ✅ Reports & CSV Export
- Filter by date range and section.
- One-click CSV download.

### ✅ Admin Authentication
- SHA-256 password hashing.
- Flask session management.
- Default credentials: `admin` / `admin123` (change immediately!).

### ✅ Privacy & Data Deletion
- Settings page allows permanent deletion of student images and records.
- Consent checkbox required during registration.

---

## Project Structure

```
attendance_system/
├── app.py              ← Flask backend (main entry point)
├── camera.py           ← Webcam thread + recognition loop
├── face_utils.py       ← Face detection, capture, training utilities
├── database.py         ← SQLite CRUD operations
├── train_model.py      ← Standalone training script
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
│
├── templates/
│   ├── base.html       ← Shared sidebar + navbar layout
│   ├── login.html      ← Admin login page
│   ├── dashboard.html  ← Live stats dashboard
│   ├── register.html   ← Student registration form
│   ├── capture.html    ← Webcam face capture page
│   ├── recognition.html← Live MJPEG feed + controls
│   ├── attendance.html ← View attendance by date
│   ├── reports.html    ← Filtered reports + CSV export
│   ├── train.html      ← Train model page
│   └── settings.html   ← Admin settings + delete student
│
├── static/
│   ├── css/style.css   ← Custom premium CSS
│   └── js/main.js      ← Global JavaScript
│
├── dataset/            ← Auto-created; stores student face images
│   └── student_1/
│       ├── face_1.jpg
│       └── …
│
├── models/             ← Auto-created; stores trained model
│   └── face_model.yml
│
└── attendance.db       ← Auto-created SQLite database
```

---

## Installation Guide

### Step 1 – Install Python

Download and install Python 3.10 or newer from [python.org](https://www.python.org/downloads/).

During installation: ✅ Check **"Add Python to PATH"**

Verify:
```powershell
python --version
```

### Step 2 – Create a Virtual Environment (Recommended)

```powershell
cd C:\sex\attendance_system
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your prompt.

### Step 3 – Install Dependencies

```powershell
pip install -r requirements.txt
```

> **Important:** `opencv-contrib-python` is required for the LBPH face recognizer.
> Do NOT install both `opencv-python` and `opencv-contrib-python` – uninstall one first if needed.

If you get a conflict:
```powershell
pip uninstall opencv-python
pip install opencv-contrib-python
```

### Step 4 – Verify Camera Access

```powershell
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera NOT found'); cap.release()"
```

---

## Running the Application

```powershell
# Activate virtual environment (if not already active)
venv\Scripts\activate

# Start the Flask server
python app.py
```

You will see output like:
```
============================================================
  Wi-Fi Smart Attendance System
============================================================
  Local  URL : http://127.0.0.1:5000
  Network URL: http://192.168.1.15:5000

  Share the Network URL with devices on the same Wi-Fi.
  Default login → username: admin | password: admin123
============================================================
```

Open **http://127.0.0.1:5000** in your browser.

---

## Usage Workflow

```
1. Login  →  admin / admin123
2. Register Student  →  fill form + capture 30 face images
3. Train Model  →  click "Start Training"  (5–30 seconds)
4. Face Recognition  →  select session + click "Start Attendance"
5. Attendance Records  →  view by date / section
6. Reports  →  filter + Export CSV
7. Settings  →  change password or delete student data
```

---

## Accessing from Other Devices (Wi-Fi)

### Find your local IP address

```powershell
ipconfig
```

Look for **IPv4 Address** under your Wi-Fi adapter, e.g. `192.168.1.15`.

The app prints this automatically when it starts.

### Connect from another device

1. Connect your phone or another laptop to the **same Wi-Fi network**.
2. Open a browser and go to: `http://192.168.1.15:5000`
3. Log in with the admin credentials.

> ⚠️ **Security:** The app is accessible only on your local network. Port 5000 is NOT exposed to the internet unless you explicitly configure port forwarding (which you should NOT do for this project).

---

## Privacy & Consent

- A consent checkbox is shown during student registration.
- Face images are stored only on the local machine in the `dataset/` folder.
- Attendance data is stored in `attendance.db` (local SQLite file).
- Students can request deletion via **Settings → Delete Student Data**.
- No data is transmitted over the internet.

---

## Anti-Spoofing Notice

> ⚠️ This system uses **2D face detection** (Haar Cascade) and **LBPH recognition**. It does NOT include liveness detection. A printed photograph or a face on a screen may be recognized. For production use, add a liveness check (e.g., blink detection, 3D depth sensing, or a dedicated anti-spoofing model).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera not found | Try `cv2.VideoCapture(1)` in camera.py |
| `cv2.face` not found | Install `opencv-contrib-python` |
| Port already in use | Kill process on 5000: `netsh http delete urlacl url=http://+:5000/` |
| Low recognition accuracy | Capture more images with varied lighting/angles |
| Other device can't connect | Check firewall: allow Python on port 5000 |
| Windows Firewall blocking | Run: `netsh advfirewall firewall add rule name="Attendance" dir=in action=allow protocol=TCP localport=5000` |

---

## Future Extensions (ESP32 / IoT)

This project can be extended with an **ESP32 microcontroller**:

```
ESP32 Camera (OV2640)
       │
       │ HTTP POST (JPEG frame)
       ▼
Flask Server  →  OpenCV Recognition  →  SQLite  →  Dashboard
```

**Steps to extend:**
1. Flash the ESP32-CAM with Arduino firmware to stream JPEG frames.
2. Add a Flask route `/api/esp32_frame` that receives the JPEG, runs `detect_faces()`, and marks attendance.
3. ESP32 connects to the same Wi-Fi network and sends frames every 2–3 seconds.
4. No PC webcam needed – the ESP32 acts as a remote camera.

**Why this is useful:**
- Mount the ESP32 at the classroom entrance (no laptop needed there).
- Multiple cameras in multiple rooms, all reporting to one Flask server.

---

## Default Credentials

| Field    | Value     |
|----------|-----------|
| Username | `admin`   |
| Password | `admin123` |

**Change the password immediately after first login via Settings.**

---

*Academic project – for educational use only.*
