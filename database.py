"""
database.py - SQLite Database Operations
Wi-Fi Based Smart Attendance System Using OpenCV and Face Recognition
"""

import sqlite3
import hashlib
import os
from datetime import datetime, date

# Path to the SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), 'attendance.db')


def get_db_connection():
    """Create and return a database connection with row factory.
    Sets a generous timeout and busy_timeout to mitigate locking issues.
    """
    # timeout=30 seconds, check_same_thread=False allows usage across threads
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrency
    conn.execute("PRAGMA journal_mode=WAL")
    # Ensure SQLite waits up to 30 s when the DB is busy
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db():
    """Initialize the database by creating all required tables."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Admin users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Students table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            section TEXT NOT NULL,
            face_label INTEGER UNIQUE,
            image_count INTEGER DEFAULT 0,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            consent_given INTEGER DEFAULT 0
        )
    """)

    # Attendance records table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            roll_number TEXT NOT NULL,
            student_name TEXT NOT NULL,
            section TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            session TEXT DEFAULT 'Morning',
            marked_by TEXT DEFAULT 'Face Recognition',
            FOREIGN KEY (student_id) REFERENCES students(id),
            UNIQUE(student_id, date, session)
        )
    """)

    # Sessions table to track recognition sessions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT NOT NULL,
            date TEXT NOT NULL,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            status TEXT DEFAULT 'active'
        )
    """)

    conn.commit()

    # Create default admin if none exists
    cursor.execute("SELECT COUNT(*) FROM admins")
    if cursor.fetchone()[0] == 0:
        default_hash = hash_password("admin123")
        cursor.execute(
            "INSERT INTO admins (username, password_hash) VALUES (?, ?)",
            ("admin", default_hash)
        )
        conn.commit()

    conn.close()
    print("[DB] Database initialized successfully.")


def hash_password(password: str) -> str:
    """Return a SHA-256 hex digest of the given password."""
    return hashlib.sha256(password.encode()).hexdigest()


# ──────────────────────────── Admin ────────────────────────────

def verify_admin(username: str, password: str) -> bool:
    """Return True if the username/password pair is valid."""
    conn = get_db_connection()
    row = conn.execute(
        "SELECT password_hash FROM admins WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    if row is None:
        return False
    return row["password_hash"] == hash_password(password)


def change_admin_password(username: str, new_password: str) -> bool:
    """Update the admin password. Returns True on success."""
    try:
        conn = get_db_connection()
        conn.execute(
            "UPDATE admins SET password_hash = ? WHERE username = ?",
            (hash_password(new_password), username)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[DB] Error changing password: {e}")
        return False


# ──────────────────────────── Students ────────────────────────────

def add_student(roll_number: str, name: str, section: str,
                face_label: int, consent_given: bool = True) -> int | None:
    """Insert a new student record with retry on locked DB.
    Returns the new row id, or None if the roll number already exists.
    """
    for attempt in range(3):
        try:
            with get_db_connection() as conn:
                cursor = conn.execute(
                    """INSERT INTO students
                       (roll_number, name, section, face_label, consent_given)
                       VALUES (?, ?, ?, ?, ?)""",
                    (roll_number.strip().upper(), name.strip(), section.strip().upper(),
                     face_label, 1 if consent_given else 0)
                )
                new_id = cursor.lastrowid
                conn.commit()
                return new_id
        except sqlite3.IntegrityError:
            return None
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < 2:
                import time; time.sleep(0.1)
                continue
            else:
                print(f"[DB] add_student error: {e}")
                return None


def get_all_students():
    """Return a list of all registered students as Row objects."""
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT * FROM students ORDER BY roll_number"
    ).fetchall()
    conn.close()
    return rows


def get_student_by_label(face_label: int):
    """Return the student Row whose face_label matches the given integer."""
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM students WHERE face_label = ?", (face_label,)
    ).fetchone()
    conn.close()
    return row


def get_student_by_roll(roll_number: str):
    """Return the student Row matching the given roll number."""
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM students WHERE roll_number = ?",
        (roll_number.strip().upper(),)
    ).fetchone()
    conn.close()
    return row


def update_student_image_count(student_id: int, count: int):
    """Update the stored image count for a student."""
    conn = get_db_connection()
    conn.execute(
        "UPDATE students SET image_count = ? WHERE id = ?",
        (count, student_id)
    )
    conn.commit()
    conn.close()


def get_next_face_label() -> int:
    """Return the next available face label integer (max existing + 1)."""
    conn = get_db_connection()
    row = conn.execute("SELECT MAX(face_label) FROM students").fetchone()
    conn.close()
    max_label = row[0]
    return 1 if max_label is None else max_label + 1


def delete_student(student_id: int) -> bool:
    """
    Delete a student and their attendance records.
    Face images must be deleted separately via face_utils.
    """
    try:
        conn = get_db_connection()
        # Remove attendance records first (FK constraint)
        conn.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[DB] Error deleting student: {e}")
        return False


# ──────────────────────────── Attendance ────────────────────────────

def mark_attendance(student_id: int, roll_number: str, student_name: str,
                    section: str, session: str = "Morning") -> bool:
    """
    Record attendance for a student.
    Returns True if a new record was inserted,
    False if a duplicate record already exists for today/session.
    """
    today = date.today().strftime("%Y-%m-%d")
    now   = datetime.now().strftime("%H:%M:%S")
    # Retry up to 3 times in case of SQLite locking
    for attempt in range(3):
        try:
            with get_db_connection() as conn:
                conn.execute(
                    """INSERT INTO attendance
                       (student_id, roll_number, student_name, section, date, time, session)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (student_id, roll_number, student_name, section, today, now, session)
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Duplicate – already marked present
            return False
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < 2:
                # brief pause before retry
                import time; time.sleep(0.1)
                continue
            else:
                print(f"[DB] mark_attendance error: {e}")
                return False
    return False


def get_attendance_by_date(target_date: str):
    """Return all attendance records for the given date (YYYY-MM-DD)."""
    conn = get_db_connection()
    rows = conn.execute(
        """SELECT a.*, s.name as student_name_full
           FROM attendance a
           JOIN students s ON a.student_id = s.id
           WHERE a.date = ?
           ORDER BY a.time DESC""",
        (target_date,)
    ).fetchall()
    conn.close()
    return rows


def get_todays_attendance():
    """Return all attendance records for today."""
    return get_attendance_by_date(date.today().strftime("%Y-%m-%d"))


def get_attendance_report(start_date: str = None, end_date: str = None,
                           section: str = None):
    """
    Return attendance rows filtered by optional date range and section.
    Dates should be in YYYY-MM-DD format.
    """
    conn = get_db_connection()
    query  = "SELECT * FROM attendance WHERE 1=1"
    params = []

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    if section:
        query += " AND section = ?"
        params.append(section.upper())

    query += " ORDER BY date DESC, time DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


def get_dashboard_stats():
    """
    Return a dict with statistics for the dashboard:
        total_students, present_today, absent_today,
        attendance_percentage, recent_records
    """
    today = date.today().strftime("%Y-%m-%d")
    conn  = get_db_connection()

    total = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]

    present = conn.execute(
        "SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date = ?",
        (today,)
    ).fetchone()[0]

    absent = total - present
    percentage = round((present / total * 100), 2) if total > 0 else 0

    recent = conn.execute(
        """SELECT student_name, roll_number, section, time
           FROM attendance
           WHERE date = ?
           ORDER BY time DESC
           LIMIT 10""",
        (today,)
    ).fetchall()

    conn.close()
    return {
        "total_students":        total,
        "present_today":         present,
        "absent_today":          absent,
        "attendance_percentage": percentage,
        "recent_records":        [dict(r) for r in recent],
        "date":                  today
    }


def get_all_sections():
    """Return a sorted list of unique section names."""
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT DISTINCT section FROM students ORDER BY section"
    ).fetchall()
    conn.close()
    return [r["section"] for r in rows]


def is_already_marked(student_id: int, session: str = "Morning") -> bool:
    """Return True if the student already has an entry for today's session."""
    today = date.today().strftime("%Y-%m-%d")
    conn  = get_db_connection()
    row   = conn.execute(
        "SELECT id FROM attendance WHERE student_id=? AND date=? AND session=?",
        (student_id, today, session)
    ).fetchone()
    conn.close()
    return row is not None
