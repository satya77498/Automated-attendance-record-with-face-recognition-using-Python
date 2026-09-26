import csv
import calendar
import datetime as dt
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import queue
import re
import shutil
import sys
import threading
import tkinter as tkinter
from tkinter import messagebox
import time
from types import SimpleNamespace

import customtkinter as ctk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk


if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = BASE_DIR / "TrainingImage"
MODEL_DIR = BASE_DIR / "TrainingImageLabel"
STUDENT_FILE = BASE_DIR / "StudentDetails" / "StudentDetails.csv"
ATTENDANCE_DIR = BASE_DIR / "Attendance"
ATTENDANCE_PROOF_DIR = BASE_DIR / "AttendanceProof"
CASCADE_FILE = BASE_DIR / "haarcascade_frontalface_default.xml"
MODEL_FILE = MODEL_DIR / "Trainer.yml"
SETTINGS_FILE = BASE_DIR / "Settings" / "settings.json"
ATTENDANCE_COLUMNS = ("Id", "Name", "Date", "Time", "Status")
SAMPLE_COUNT = 25
MATCH_THRESHOLD = 55
DUPLICATE_MATCH_THRESHOLD = 45
DUPLICATE_CONFIRMATIONS = 3
DUPLICATE_CHECK_SAMPLES = 7

DEFAULT_SETTINGS = {
    "camera_index": 0,
    "match_threshold": MATCH_THRESHOLD,
    "duplicate_threshold": DUPLICATE_MATCH_THRESHOLD,
    "sample_count": SAMPLE_COUNT,
    "late_cutoff": "09:15",
    "admin_pin_hash": None,
}

COLORS = {
    "ink": "#F1F5FA",
    "muted": "#9AA8B8",
    "paper": "#111720",
    "white": "#1A222D",
    "line": "#303B49",
    "green": "#38C4E2",
    "green_dark": "#2389A3",
    "lime": "#74E6CF",
    "coral": "#E7A36D",
    "red": "#F07882",
}


ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")
ctk.set_widget_scaling(1.08)


def _ctk_map_widget_kwargs(kwargs):
    if "bg" in kwargs and "fg_color" not in kwargs:
        kwargs["fg_color"] = kwargs.pop("bg")
    if "fg" in kwargs and "text_color" not in kwargs:
        kwargs["text_color"] = kwargs.pop("fg")
    if "activebackground" in kwargs:
        kwargs.setdefault("hover_color", kwargs.pop("activebackground"))
    if "activeforeground" in kwargs and "text_color" not in kwargs:
        kwargs["text_color"] = kwargs.pop("activeforeground")
    kwargs.pop("activeforeground", None)
    if "highlightbackground" in kwargs and "border_color" not in kwargs:
        kwargs["border_color"] = kwargs.pop("highlightbackground")
    if "highlightthickness" in kwargs and "border_width" not in kwargs:
        kwargs["border_width"] = kwargs.pop("highlightthickness")
    if "relief" in kwargs:
        kwargs.pop("relief")
    if "bd" in kwargs:
        kwargs.pop("bd")
    if "cursor" in kwargs:
        kwargs.pop("cursor")
    kwargs.pop("padx", None)
    kwargs.pop("pady", None)
    kwargs.pop("selectcolor", None)
    return kwargs


def _ctk_frame(*args, **kwargs):
    return ctk.CTkFrame(*args, **_ctk_map_widget_kwargs(kwargs))


def _ctk_label(*args, **kwargs):
    kwargs = _ctk_map_widget_kwargs(kwargs)
    return ctk.CTkLabel(*args, **kwargs)


def _ctk_button(*args, **kwargs):
    kwargs = _ctk_map_widget_kwargs(kwargs)
    kwargs.pop("justify", None)
    return ctk.CTkButton(*args, **kwargs)


def _ctk_entry(*args, **kwargs):
    kwargs = _ctk_map_widget_kwargs(kwargs)
    return ctk.CTkEntry(*args, **kwargs)


def _ctk_checkbutton(*args, **kwargs):
    kwargs = _ctk_map_widget_kwargs(kwargs)
    return ctk.CTkCheckBox(*args, **kwargs)


def _ctk_toplevel(*args, **kwargs):
    kwargs = _ctk_map_widget_kwargs(kwargs)
    return ctk.CTkToplevel(*args, **kwargs)


tk = SimpleNamespace(
    Frame=_ctk_frame,
    Label=_ctk_label,
    Button=_ctk_button,
    Entry=_ctk_entry,
    Checkbutton=_ctk_checkbutton,
    Toplevel=_ctk_toplevel,
    BooleanVar=tkinter.BooleanVar,
    Canvas=tkinter.Canvas,
    Listbox=tkinter.Listbox,
    TclError=tkinter.TclError,
)


def load_users():
    users = {}
    if not STUDENT_FILE.exists():
        return users
    with STUDENT_FILE.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.reader(file):
            if len(row) < 2 or row[0].strip().lower() == "id":
                continue
            person_id = row[0].strip()
            if person_id.isdigit():
                person_id = str(int(person_id))
            users[person_id] = row[1].strip()
    return users


def student_training_dir(person_id, name=None):
    person_id = str(int(person_id))
    if name is None:
        name = load_users().get(person_id, "user")
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_-") or "user"
    return TRAINING_DIR / f"{person_id}_{safe_name}"


def organize_training_images():
    users = load_users()
    for image_path in TRAINING_DIR.glob("*.jpg"):
        pieces = image_path.stem.split(".")
        if len(pieces) < 3 or not pieces[1].isdigit():
            continue
        person_id = str(int(pieces[1]))
        destination_dir = student_training_dir(person_id, users.get(person_id, "user"))
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / image_path.name
        if destination.exists():
            existing_numbers = []
            for candidate in destination_dir.glob(f"*.{person_id}.*.jpg"):
                try:
                    existing_numbers.append(int(candidate.stem.rsplit(".", 1)[1]))
                except ValueError:
                    continue
            next_number = max(existing_numbers, default=0) + 1
            destination = destination_dir / f"face.{person_id}.{next_number}.jpg"
        image_path.replace(destination)


def load_training_images():
    faces = []
    ids = []
    for image_path in TRAINING_DIR.rglob("*.jpg"):
        try:
            person_id = int(image_path.stem.rsplit(".", 2)[1])
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            faces.append(cv2.resize(image, (200, 200)))
            ids.append(person_id)
        except (IndexError, ValueError, cv2.error):
            continue
    return faces, ids


def train_recognizer():
    recognizer_factory = getattr(getattr(cv2, "face", None), "LBPHFaceRecognizer_create", None)
    if not callable(recognizer_factory):
        raise RuntimeError(
            "This Python environment does not have OpenCV's LBPH recognizer. "
            f"Python: {sys.executable}\n"
            "Use the project .venv interpreter, or install only opencv-contrib-python "
            "in this environment (remove the conflicting opencv-python package first)."
        )
    faces, ids = load_training_images()
    if not faces:
        raise RuntimeError("No face samples found. Register a user before training.")
    recognizer = recognizer_factory()
    recognizer.train(faces, np.asarray(ids, dtype=np.int32))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    recognizer.save(str(MODEL_FILE))


def read_csv_rows(path):
    try:
        path = path.resolve()
        stat = path.stat()
    except OSError:
        return []
    return _read_csv_rows_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=128)
def _read_csv_rows_cached(path_text, modified_ns, file_size):
    with Path(path_text).open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def load_settings():
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            with SETTINGS_FILE.open("r", encoding="utf-8") as file:
                stored = json.load(file)
            if isinstance(stored, dict):
                settings.update(stored)
        except (json.JSONDecodeError, OSError):
            pass
    return settings


def save_settings(settings):
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_FILE.open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2)


def hash_pin(pin):
    return hashlib.sha256(pin.encode("utf-8")).hexdigest()


def _record_duplicate_vote(votes, person_id, confidence, threshold):
    if confidence > threshold:
        return False
    votes[person_id] = votes.get(person_id, 0) + 1
    return votes[person_id] >= DUPLICATE_CONFIRMATIONS


def parse_cutoff_time(text, fallback="09:15"):
    for value in (text, fallback):
        try:
            hour, minute = value.split(":")
            return dt.time(int(hour), int(minute))
        except (ValueError, AttributeError):
            continue
    return dt.time(9, 15)


def normalize_person_id(value):
    if value is None:
        return ""
    person_id = str(value).strip()
    if not person_id:
        return ""
    if person_id.isdigit():
        return str(int(person_id))
    return person_id


def registered_attendance_rows(rows, users):
    normalized_users = {
        normalize_person_id(person_id): name
        for person_id, name in users.items()
    }
    registered_rows = []
    for row in rows:
        person_id = normalize_person_id(row.get("Id"))
        if person_id in normalized_users:
            registered_rows.append({**row, "Id": person_id, "Name": normalized_users[person_id]})
    return registered_rows


def iter_attendance_files(date_text=None):
    if date_text is not None:
        return [ATTENDANCE_DIR / f"Attendance_{date_text}.csv"]
    return sorted(ATTENDANCE_DIR.glob("Attendance_*.csv"))


def attendance_period_bounds(period, anchor_date):
    if period == "week":
        first_day = anchor_date - dt.timedelta(days=anchor_date.weekday())
        return first_day, first_day + dt.timedelta(days=6)
    if period == "month":
        first_day = anchor_date.replace(day=1)
        last_day = anchor_date.replace(day=calendar.monthrange(anchor_date.year, anchor_date.month)[1])
        return first_day, last_day
    raise ValueError(f"Unsupported attendance period: {period}")


def find_attendance_proof(person_id, date_text):
    normalized_id = normalize_person_id(person_id)
    if not normalized_id.isdigit():
        return None
    proof_dir = ATTENDANCE_PROOF_DIR / date_text
    proof_files = sorted(proof_dir.glob(f"{normalized_id}_*.jpg"))
    return proof_files[-1] if proof_files else None


def open_camera_resources(camera_index):
    if not CASCADE_FILE.exists():
        raise FileNotFoundError(f"Could not find {CASCADE_FILE.name}.")
    detector = cv2.CascadeClassifier(str(CASCADE_FILE))
    if detector.empty():
        raise RuntimeError("The face detector file could not be loaded.")
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera index {camera_index}.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return detector, camera


def build_attendance_summary(attendance_path=None, date_text=None, registered_users=None):
    summary = {"total": 0, "present": 0, "late": 0, "absent": 0, "leave": 0, "unknown": 0}
    if attendance_path is None:
        files = iter_attendance_files(date_text)
    else:
        files = [attendance_path]
    for path in files:
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            if not isinstance(row, dict):
                continue
            if registered_users is not None:
                person_id = normalize_person_id(row.get("Id"))
                if person_id not in registered_users:
                    continue
            status = (row.get("Status") or "").strip()
            summary["total"] += 1
            if status.lower() == "late":
                summary["late"] += 1
            elif status.lower() in {"present", "on time"}:
                summary["present"] += 1
            elif status.lower() == "absent":
                summary["absent"] += 1
            elif status.lower() == "leave":
                summary["leave"] += 1
            else:
                summary["unknown"] += 1
    return summary


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fieldnote | Face Attendance")
        self.root.geometry("1280x840")
        self.root.minsize(1080, 720)
        self.root.configure(bg=COLORS["paper"])
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.camera = None
        self.camera_job = None
        self.camera_startup = None
        self.camera_progress = None
        self.detector = None
        self.camera_mode = None
        self.page = None
        self.page_cache = {}
        self.current_photo = None
        self.profile_photo_cache = {}
        self.profile_sample_index = None
        self.selected_user_id = None
        self.history_period = "week"
        self.history_anchor_date = dt.date.today()
        self.history_photo = None
        self.recent_capture_photo = None
        self.recent_capture_person_id = None
        self.preview_overlay = []
        self.capture_samples = []
        self.last_sample_time = 0.0
        self.last_duplicate_check = 0.0
        self.duplicate_check_samples = 0
        self.duplicate_candidate_votes = {}
        self.registration_recognizer = None
        self.registration = None
        self.attendance_users = {}
        self.attendance_seen = set()
        self.candidate_id = None
        self.candidate_frames = 0
        self.settings = load_settings()

        self._build_shell()
        self.show_home()

    def _build_shell(self):
        self.body = tk.Frame(self.root, bg=COLORS["paper"])
        self.body.pack(fill="both", expand=True)

        self.sidebar = tk.Frame(self.body, bg="#0D1219", width=214)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        tk.Label(
            self.sidebar, text="FIELDNOTE", bg="#0D1219", fg=COLORS["green"],
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=22, pady=(28, 2))
        tk.Label(
            self.sidebar, text="Face attendance", bg="#0D1219", fg=COLORS["muted"],
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w", padx=22, pady=(0, 28))
        tk.Label(
            self.sidebar, text="WORKSPACE", bg="#0D1219", fg="#758294",
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", padx=22, pady=(0, 10))

        self.nav_buttons = {}
        nav_items = [
            ("Dashboard", self.show_home),
            ("Live attendance", self.show_attendance),
            ("History", self.show_history),
            ("People", self.show_users),
            ("Register face", self.show_registration),
            ("Reports", self.show_analytics),
            ("Settings", self.show_settings),
        ]
        for text, command in nav_items:
            button = tk.Button(
                self.sidebar, text=text, command=command,
                bg="#0D1219", fg=COLORS["muted"], activebackground="#19232F",
                anchor="w", justify="left", relief="flat", bd=0,
                font=("Segoe UI", 10, "bold"), cursor="hand2",
                width=174, height=40, corner_radius=8,
            )
            button.pack(fill="x", padx=12, pady=3)
            self.nav_buttons[text] = button

        self.main = tk.Frame(self.body, bg=COLORS["paper"])
        self.main.pack(side="left", fill="both", expand=True, padx=(22, 24))
        topbar = tk.Frame(self.main, bg=COLORS["paper"], height=58)
        topbar.pack(fill="x", pady=(12, 8))
        topbar.pack_propagate(False)
        self.header_meta = tk.Frame(topbar, bg=COLORS["paper"])
        self.header_meta.pack(side="right", anchor="center")
        self.header_date = tk.Label(
            self.header_meta, text=dt.date.today().strftime("%d %b %Y"),
            bg=COLORS["white"], fg=COLORS["ink"], font=("Segoe UI", 9, "bold"),
        )
        self.header_date.pack(side="left", padx=(0, 10), ipady=7, ipadx=10)
        self.header_time = tk.Label(
            self.header_meta, text=dt.datetime.now().strftime("%I:%M %p"),
            bg=COLORS["paper"], fg=COLORS["muted"], font=("Segoe UI", 10, "bold"),
        )
        self.header_time.pack(side="left")
        self.page_host = tk.Frame(self.main, bg=COLORS["paper"])
        self.page_host.pack(fill="both", expand=True, pady=(0, 20))
        self.page = None

        profile = tk.Frame(self.sidebar, bg="#151D27", corner_radius=10)
        profile.pack(side="bottom", fill="x", padx=12, pady=14)
        tk.Label(
            profile, text="●  Administrator", bg="#151D27", fg=COLORS["ink"],
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=12, pady=(11, 2))
        tk.Label(
            profile, text="     Local workspace", bg="#151D27", fg=COLORS["muted"],
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=12, pady=(0, 11))

        self._refresh_clock()

    def _new_page(self):
        self._cancel_camera_startup()
        self._stop_camera()
        if self.page is not None:
            if any(self.page is cached_page for cached_page in self.page_cache.values()):
                self.page.pack_forget()
            else:
                self.page.destroy()
        self.page = tk.Frame(self.page_host, bg=COLORS["paper"])
        self.page.pack(fill="both", expand=True)
        return self.page

    def _show_cached_page(self, key, nav_item):
        self._cancel_camera_startup()
        self._stop_camera()
        cached_page = self.page_cache.get(key)
        if cached_page is None:
            return None
        if self.page is not None and self.page is not cached_page:
            if any(self.page is page for page in self.page_cache.values()):
                self.page.pack_forget()
            else:
                self.page.destroy()
        cached_page.pack(fill="both", expand=True)
        self.page = cached_page
        self._set_active_nav(nav_item)
        return cached_page

    def _cache_current_page(self, key):
        self.page_cache[key] = self.page

    def _invalidate_page_cache(self, *keys):
        for key in keys:
            cached_page = self.page_cache.pop(key, None)
            if cached_page is not None and cached_page is not self.page:
                cached_page.destroy()

    def _set_active_nav(self, selected):
        for name, button in self.nav_buttons.items():
            active = name == selected
            button.configure(
                fg_color="#16313D" if active else "#0D1219",
                text_color=COLORS["green"] if active else COLORS["muted"],
                hover_color="#19232F",
                border_width=1 if active else 0,
                border_color=COLORS["green"],
            )

    def _refresh_clock(self):
        now = dt.datetime.now()
        self.header_date.configure(text=now.strftime("%A, %d %B %Y"))
        self.header_time.configure(text=now.strftime("%I:%M %p"))
        self.root.after(30000, self._refresh_clock)

    def _label(self, parent, text, **kwargs):
        options = {
            "bg": COLORS["paper"], "fg": COLORS["ink"],
            "font": ("Segoe UI", 10),
        }
        options.update(kwargs)
        return tk.Label(parent, text=text, **options)

    def _button(self, parent, text, command, primary=False, **kwargs):
        background = kwargs.pop("bg", COLORS["green"] if primary else COLORS["white"])
        active_background = kwargs.pop("activebackground", COLORS["green_dark"])
        hover_background = kwargs.pop(
            "hoverbackground",
            COLORS["green_dark"] if primary else COLORS["paper"],
        )
        foreground = COLORS["white"] if primary else COLORS["ink"]
        button = tk.Button(
            parent, text=text, command=command,
            bg=background, fg=foreground, activebackground=active_background,
            activeforeground=COLORS["white"], relief="flat", bd=0,
            padx=18, pady=11, cursor="hand2", font=("Segoe UI", 10, "bold"),
            hover_color=hover_background,
            **kwargs,
        )
        return button

    def _authorize(self, purpose="continue"):
        """Gate a sensitive action behind the admin PIN. Returns True if allowed."""
        if not self.settings.get("admin_pin_hash"):
            return self._first_time_pin_setup(purpose)
        pin = self._prompt_pin(f"Enter the admin PIN to {purpose}.")
        if pin is None:
            return False
        if hash_pin(pin) != self.settings["admin_pin_hash"]:
            messagebox.showerror("Incorrect PIN", "That PIN did not match.", parent=self.root)
            return False
        return True

    def _first_time_pin_setup(self, purpose):
        proceed = messagebox.askyesno(
            "Set an admin PIN",
            f"No admin PIN is set yet. Set one now to protect settings and profile removal, "
            f"and then {purpose}?",
            parent=self.root,
        )
        if not proceed:
            return False
        pin = self._prompt_pin("Choose a new admin PIN (4+ characters):")
        if not pin or len(pin) < 4:
            if pin is not None:
                messagebox.showwarning("PIN too short", "Use at least 4 characters.", parent=self.root)
            return False
        confirm = self._prompt_pin("Confirm the new admin PIN:")
        if confirm != pin:
            messagebox.showwarning("PIN mismatch", "The two PINs did not match.", parent=self.root)
            return False
        self.settings["admin_pin_hash"] = hash_pin(pin)
        save_settings(self.settings)
        messagebox.showinfo("PIN set", "Admin PIN saved.", parent=self.root)
        return True

    def _prompt_pin(self, message):
        dialog = tk.Toplevel(self.root)
        dialog.title("Admin PIN")
        dialog.configure(bg=COLORS["white"])
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        tk.Label(
            dialog, text=message, bg=COLORS["white"], fg=COLORS["ink"],
            font=("Segoe UI", 10, "bold"), wraplength=280, justify="left",
        ).pack(padx=22, pady=(20, 10))
        entry = tk.Entry(dialog, show="•", font=("Segoe UI", 12), relief="solid", bd=1, justify="center")
        entry.pack(padx=22, ipady=7, fill="x")
        entry.focus_set()

        result = {"value": None}

        def submit(_event=None):
            result["value"] = entry.get()
            dialog.destroy()

        def cancel():
            result["value"] = None
            dialog.destroy()

        entry.bind("<Return>", submit)
        dialog.protocol("WM_DELETE_WINDOW", cancel)

        controls = tk.Frame(dialog, bg=COLORS["white"])
        controls.pack(fill="x", padx=22, pady=18)
        self._button(controls, "Cancel", cancel).pack(side="left")
        self._button(controls, "Confirm", submit, primary=True).pack(side="right")

        self.root.wait_window(dialog)
        return result["value"]

    def show_settings(self):
        if not self._authorize("open settings"):
            return
        page = self._new_page()
        self._set_active_nav("Settings")
        self._label(page, "Settings", font=("Segoe UI", 22, "bold")).pack(anchor="w")
        self._label(
            page, "Tune recognition sensitivity, camera, and late-arrival cutoff for this device.",
            fg=COLORS["muted"],
        ).pack(anchor="w", pady=(5, 18))

        form = tk.Frame(page, bg=COLORS["white"], padx=24, pady=22,
                        highlightbackground=COLORS["line"], highlightthickness=1)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        fields = [
            ("camera_index", "Camera index (0 = default camera)"),
            ("match_threshold", "Match threshold (lower = stricter, 40-80 typical)"),
            ("duplicate_threshold", "Duplicate-check threshold (lower = stricter)"),
            ("sample_count", "Face samples captured per registration"),
            ("late_cutoff", "Late-arrival cutoff time (HH:MM, 24h)"),
        ]
        self.settings_entries = {}
        for row_index, (key, description) in enumerate(fields):
            tk.Label(form, text=description, bg=COLORS["white"], fg=COLORS["muted"],
                     font=("Segoe UI", 9)).grid(row=row_index, column=0, sticky="w", pady=8)
            entry = tk.Entry(form, font=("Segoe UI", 11), relief="solid", bd=1, width=12, justify="center")
            entry.insert(0, str(self.settings.get(key, DEFAULT_SETTINGS[key])))
            entry.grid(row=row_index, column=1, sticky="e", padx=(18, 0), ipady=6)
            self.settings_entries[key] = entry

        pin_row = tk.Frame(page, bg=COLORS["paper"])
        pin_row.pack(fill="x", pady=(16, 0))
        pin_state = "An admin PIN is set." if self.settings.get("admin_pin_hash") else "No admin PIN is set yet."
        self._label(pin_row, pin_state, fg=COLORS["muted"]).pack(side="left")
        self._button(pin_row, "Change admin PIN", self.change_admin_pin).pack(side="right")

        controls = tk.Frame(page, bg=COLORS["paper"])
        controls.pack(fill="x", pady=(20, 0))
        self._button(controls, "Back", self.show_home).pack(side="left")
        self._button(controls, "Save settings", self.save_settings_form, primary=True).pack(side="right")

    def change_admin_pin(self):
        current_hash = self.settings.get("admin_pin_hash")
        if current_hash:
            pin = self._prompt_pin("Enter the current admin PIN.")
            if pin is None:
                return
            if hash_pin(pin) != current_hash:
                messagebox.showerror("Incorrect PIN", "That PIN did not match.", parent=self.root)
                return
        new_pin = self._prompt_pin("Choose a new admin PIN (4+ characters):")
        if not new_pin or len(new_pin) < 4:
            if new_pin is not None:
                messagebox.showwarning("PIN too short", "Use at least 4 characters.", parent=self.root)
            return
        confirm = self._prompt_pin("Confirm the new admin PIN:")
        if confirm != new_pin:
            messagebox.showwarning("PIN mismatch", "The two PINs did not match.", parent=self.root)
            return
        self.settings["admin_pin_hash"] = hash_pin(new_pin)
        save_settings(self.settings)
        messagebox.showinfo("PIN updated", "Admin PIN was updated.", parent=self.root)
        self.show_settings()

    def save_settings_form(self):
        try:
            camera_index = int(self.settings_entries["camera_index"].get().strip())
            match_threshold = float(self.settings_entries["match_threshold"].get().strip())
            duplicate_threshold = float(self.settings_entries["duplicate_threshold"].get().strip())
            sample_count = int(self.settings_entries["sample_count"].get().strip())
            late_cutoff = self.settings_entries["late_cutoff"].get().strip()
            parse_cutoff_time(late_cutoff)  # validates format
            if sample_count < 5:
                raise ValueError("Sample count must be at least 5.")
        except ValueError as error:
            messagebox.showwarning(
                "Check the values",
                f"Could not save settings: {error}\nUse whole numbers and an HH:MM cutoff time.",
                parent=self.root,
            )
            return

        self.settings.update({
            "camera_index": camera_index,
            "match_threshold": match_threshold,
            "duplicate_threshold": duplicate_threshold,
            "sample_count": sample_count,
            "late_cutoff": late_cutoff,
        })
        save_settings(self.settings)
        messagebox.showinfo("Settings saved", "These settings will apply to the next camera session.", parent=self.root)
        self.show_home()

    def _change_history_period(self, period):
        self.history_period = period
        self.show_history()

    def _shift_history_period(self, amount):
        if self.history_period == "week":
            self.history_anchor_date += dt.timedelta(days=7 * amount)
        else:
            month_index = self.history_anchor_date.year * 12 + self.history_anchor_date.month - 1 + amount
            year, month_offset = divmod(month_index, 12)
            self.history_anchor_date = self.history_anchor_date.replace(
                year=year, month=month_offset + 1, day=1,
            )
        self.show_history()

    def show_history(self):
        page = self._new_page()
        self._set_active_nav("History")
        first_day, last_day = attendance_period_bounds(
            self.history_period, self.history_anchor_date,
        )
        users = load_users()

        self._label(page, "Attendance history", font=("Segoe UI", 22, "bold")).pack(anchor="w")
        self._label(
            page, "Browse recorded attendance and open the proof photo saved for each entry.",
            fg=COLORS["muted"],
        ).pack(anchor="w", pady=(5, 14))

        toolbar = tk.Frame(page, bg=COLORS["paper"])
        toolbar.pack(fill="x", pady=(0, 12))
        period_buttons = tk.Frame(toolbar, bg=COLORS["paper"])
        period_buttons.pack(side="left")
        self._button(
            period_buttons, "Week", lambda: self._change_history_period("week"),
            primary=self.history_period == "week",
        ).pack(side="left", padx=(0, 6))
        self._button(
            period_buttons, "Month", lambda: self._change_history_period("month"),
            primary=self.history_period == "month",
        ).pack(side="left")
        self._button(
            toolbar, "Next", lambda: self._shift_history_period(1),
        ).pack(side="right")
        self._button(
            toolbar, "Previous", lambda: self._shift_history_period(-1),
        ).pack(side="right", padx=(0, 8))
        if self.history_period == "week":
            range_label = f"{first_day:%d %b} - {last_day:%d %b %Y}"
        else:
            range_label = first_day.strftime("%B %Y")
        self._label(
            toolbar, range_label, fg=COLORS["ink"], font=("Segoe UI", 12, "bold"),
        ).pack(side="left", padx=18)

        records_by_day = {}
        for path in iter_attendance_files():
            try:
                record_day = dt.date.fromisoformat(path.stem.removeprefix("Attendance_"))
            except ValueError:
                continue
            if not first_day <= record_day <= last_day:
                continue
            records = registered_attendance_rows(read_csv_rows(path), users)
            if records:
                records_by_day[record_day.isoformat()] = sorted(
                    records,
                    key=lambda row: row.get("Time", ""),
                    reverse=True,
                )
        self.history_records_by_day = records_by_day
        days_with_records = sorted(records_by_day, reverse=True)
        total_records = sum(len(rows) for rows in records_by_day.values())
        summary = tk.Frame(page, bg=COLORS["white"], corner_radius=8)
        summary.pack(fill="x", pady=(0, 12))
        self._label(
            summary,
            f"{total_records} records  /  {len(days_with_records)} days with attendance",
            fg=COLORS["muted"], font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=14, pady=10)

        panels = tk.Frame(page, bg=COLORS["paper"])
        panels.pack(fill="both", expand=True)
        panels.columnconfigure(0, weight=2, uniform="history")
        panels.columnconfigure(1, weight=3, uniform="history")
        panels.columnconfigure(2, weight=4, uniform="history")
        panels.rowconfigure(0, weight=1)

        day_panel = tk.Frame(
            panels, bg=COLORS["white"], highlightbackground=COLORS["line"],
            highlightthickness=1, corner_radius=10,
        )
        day_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(
            day_panel, text="DAYS", bg=COLORS["white"], fg=COLORS["muted"],
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", padx=14, pady=(12, 8))
        self.history_days_list = ctk.CTkScrollableFrame(
            day_panel, fg_color=COLORS["white"], corner_radius=0,
        )
        self.history_days_list.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        records_panel = tk.Frame(
            panels, bg=COLORS["white"], highlightbackground=COLORS["line"],
            highlightthickness=1, corner_radius=10,
        )
        records_panel.grid(row=0, column=1, sticky="nsew", padx=6)
        tk.Label(
            records_panel, text="ATTENDANCE", bg=COLORS["white"], fg=COLORS["muted"],
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", padx=14, pady=(12, 8))
        self.history_records_list = ctk.CTkScrollableFrame(
            records_panel, fg_color=COLORS["white"], corner_radius=0,
        )
        self.history_records_list.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self.history_detail_panel = tk.Frame(
            panels, bg=COLORS["white"], highlightbackground=COLORS["line"],
            highlightthickness=1, corner_radius=10,
        )
        self.history_detail_panel.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        self.history_day_buttons = {}
        for day_text in days_with_records:
            record_day = dt.date.fromisoformat(day_text)
            count = len(records_by_day[day_text])
            button = self._button(
                self.history_days_list,
                f"{record_day:%a, %d %b}    {count}",
                lambda selected_day=day_text: self._select_history_day(selected_day),
                anchor="w", height=42, corner_radius=7,
            )
            button.pack(fill="x", padx=4, pady=3)
            self.history_day_buttons[day_text] = button

        if days_with_records:
            self._select_history_day(days_with_records[0])
        else:
            tk.Label(
                self.history_days_list,
                text="No attendance recorded in this period.",
                bg=COLORS["white"], fg=COLORS["muted"], font=("Segoe UI", 10),
                wraplength=190, justify="left",
            ).pack(anchor="w", padx=12, pady=14)
            tk.Label(
                self.history_records_list,
                text="Choose another week or month.",
                bg=COLORS["white"], fg=COLORS["muted"], font=("Segoe UI", 10),
            ).pack(anchor="w", padx=12, pady=14)
            tk.Label(
                self.history_detail_panel,
                text="No proof photo to display.",
                bg=COLORS["white"], fg=COLORS["muted"], font=("Segoe UI", 10),
            ).pack(anchor="center", expand=True, pady=40)

    def _select_history_day(self, day_text):
        for current_day, button in self.history_day_buttons.items():
            active = current_day == day_text
            button.configure(
                fg_color="#16313D" if active else COLORS["white"],
                text_color=COLORS["green"] if active else COLORS["ink"],
            )
        for child in self.history_records_list.winfo_children():
            child.destroy()
        for row in self.history_records_by_day.get(day_text, []):
            label = f"{row.get('Time', '')}   {row.get('Name', '')}   {row.get('Status', '')}"
            self._button(
                self.history_records_list,
                label,
                lambda selected_row=row, selected_day=day_text: self._show_history_record(
                    selected_day, selected_row,
                ),
                bg=COLORS["white"], anchor="w", height=42, corner_radius=7,
            ).pack(fill="x", padx=4, pady=3)
        rows = self.history_records_by_day.get(day_text, [])
        if rows:
            self._show_history_record(day_text, rows[0])

    def _show_history_record(self, day_text, row):
        for child in self.history_detail_panel.winfo_children():
            child.destroy()
        proof_path = find_attendance_proof(row.get("Id", ""), day_text)
        if proof_path is not None:
            try:
                with Image.open(proof_path) as source:
                    proof_image = ImageOps.fit(
                        source.convert("RGB"), (300, 220),
                        method=Image.Resampling.LANCZOS,
                    )
                self.history_photo = ctk.CTkImage(
                    light_image=proof_image,
                    dark_image=proof_image,
                    size=(300, 220),
                )
                image_label = tk.Label(
                    self.history_detail_panel, image=self.history_photo,
                    text="", bg=COLORS["white"],
                )
                image_label.image = self.history_photo
                image_label.pack(anchor="w", padx=16, pady=(16, 12))
            except (OSError, ValueError):
                proof_path = None
        if proof_path is None:
            tk.Label(
                self.history_detail_panel, text="No proof photo stored for this record.",
                bg=COLORS["white"], fg=COLORS["muted"], font=("Segoe UI", 10),
            ).pack(anchor="w", padx=16, pady=(18, 12))
        tk.Label(
            self.history_detail_panel, text=row.get("Name", ""),
            bg=COLORS["white"], fg=COLORS["ink"], font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w", padx=16)
        tk.Label(
            self.history_detail_panel,
            text=f"ID {row.get('Id', '')}  /  {day_text}  /  {row.get('Time', '')}",
            bg=COLORS["white"], fg=COLORS["muted"], font=("Segoe UI", 10),
        ).pack(anchor="w", padx=16, pady=(4, 8))
        tk.Label(
            self.history_detail_panel, text=row.get("Status", "Recorded"),
            bg=COLORS["white"], fg=COLORS["lime"],
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=16)

    def show_analytics(self):
        if self._show_cached_page("reports", "Reports") is not None:
            return
        page = self._new_page()
        self._set_active_nav("Reports")
        self._label(page, "Analytics", font=("Segoe UI", 22, "bold")).pack(anchor="w")
        self._label(
            page, "Attendance trends from the existing local data store.",
            fg=COLORS["muted"],
        ).pack(anchor="w", pady=(5, 18))

        users = load_users()
        today = dt.date.today()
        days = [today - dt.timedelta(days=offset) for offset in range(6, -1, -1)]
        daily_counts = []
        daily_late = []
        per_user_counts = {person_id: 0 for person_id in users}
        for day in days:
            path = ATTENDANCE_DIR / f"Attendance_{day.isoformat()}.csv"
            summary = build_attendance_summary(path, registered_users=users)
            daily_counts.append(summary["total"])
            daily_late.append(summary["late"])
            for row in read_csv_rows(path):
                person_id = normalize_person_id(row.get("Id"))
                if person_id in per_user_counts:
                    per_user_counts[person_id] += 1

        stats = tk.Frame(page, bg=COLORS["paper"])
        stats.pack(fill="x", pady=(0, 20))
        self._stat(stats, "REGISTERED PEOPLE", str(len(users)), COLORS["green"]).pack(side="left", padx=(0, 12))
        self._stat(stats, "RECORDED (7 DAYS)", str(sum(daily_counts)), COLORS["coral"]).pack(side="left", padx=(0, 12))
        self._stat(stats, "LATE (7 DAYS)", str(sum(daily_late)), COLORS["red"]).pack(side="left")

        chart_panel = tk.Frame(page, bg=COLORS["white"], padx=20, pady=18,
                               highlightbackground=COLORS["line"], highlightthickness=1)
        chart_panel.pack(fill="both", expand=True)
        tk.Label(chart_panel, text="DAILY ATTENDANCE", bg=COLORS["white"], fg=COLORS["green"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 10))
        canvas = tk.Canvas(chart_panel, bg=COLORS["white"], highlightthickness=0, height=220)
        canvas.pack(fill="both", expand=True)
        canvas.bind("<Configure>", lambda event: self._draw_bar_chart(
            canvas, days, daily_counts, daily_late,
        ))

        absentees_panel = tk.Frame(page, bg=COLORS["white"], padx=20, pady=16,
                                   highlightbackground=COLORS["line"], highlightthickness=1)
        absentees_panel.pack(fill="x", pady=(16, 0))
        tk.Label(absentees_panel, text="MOST ABSENT THIS WEEK", bg=COLORS["white"], fg=COLORS["red"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 8))
        ranked = sorted(per_user_counts.items(), key=lambda item: item[1])[:5]
        if ranked and users:
            for person_id, count in ranked:
                tk.Label(
                    absentees_panel,
                    text=f"{users.get(person_id, 'Unknown')}   —   present {count} / 7 days",
                    bg=COLORS["white"], fg=COLORS["ink"], font=("Segoe UI", 10),
                ).pack(anchor="w", pady=2)
        else:
            tk.Label(absentees_panel, text="No registered users yet.", bg=COLORS["white"],
                     fg=COLORS["muted"], font=("Segoe UI", 10)).pack(anchor="w")

        controls = tk.Frame(page, bg=COLORS["paper"])
        controls.pack(fill="x", pady=(16, 0))
        self._button(controls, "Back", self.show_home).pack(side="left")
        self._cache_current_page("reports")

    def _draw_bar_chart(self, canvas, days, counts, late_counts):
        canvas.delete("all")
        width = max(canvas.winfo_width(), 320)
        height = max(canvas.winfo_height(), 160)
        padding = 30
        chart_width = width - 2 * padding
        chart_height = height - 2 * padding
        max_value = max(counts + [1])
        bar_slot = chart_width / len(days)
        bar_width = bar_slot * 0.5

        canvas.create_line(padding, height - padding, width - padding, height - padding, fill=COLORS["line"])
        for index, (day, count, late) in enumerate(zip(days, counts, late_counts)):
            x_center = padding + bar_slot * index + bar_slot / 2
            bar_height = (count / max_value) * chart_height if max_value else 0
            late_height = (late / max_value) * chart_height if max_value else 0
            x0 = x_center - bar_width / 2
            x1 = x_center + bar_width / 2
            y_base = height - padding
            if count:
                canvas.create_rectangle(
                    x0, y_base - bar_height, x1, y_base,
                    fill=COLORS["green"], outline="",
                )
                if late:
                    canvas.create_rectangle(
                        x0, y_base - bar_height, x1, y_base - bar_height + late_height,
                        fill=COLORS["red"], outline="",
                    )
            canvas.create_text(
                x_center, y_base - bar_height - 10, text=str(count),
                fill=COLORS["ink"], font=("Segoe UI", 9, "bold"),
            )
            canvas.create_text(
                x_center, y_base + 14, text=day.strftime("%a"),
                fill=COLORS["muted"], font=("Segoe UI", 8),
            )

    def show_home(self):
        if self._show_cached_page("home", "Dashboard") is not None:
            return
        page = self._new_page()
        self._set_active_nav("Dashboard")
        users = load_users()
        today = dt.date.today()
        today_summary = build_attendance_summary(
            date_text=today.isoformat(), registered_users=users,
        )
        present_today = today_summary["present"] + today_summary["late"]
        total_users = len(users)
        not_recorded = max(total_users - present_today, 0)
        attendance_rate = (present_today / total_users * 100) if total_users else 0
        hour = dt.datetime.now().hour
        greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"

        intro = tk.Frame(page, bg=COLORS["paper"])
        intro.pack(fill="x", pady=(0, 12))
        self._label(
            intro, f"{greeting}, Admin",
            font=("Segoe UI", 24, "bold"),
        ).pack(anchor="w")
        self._label(
            intro, "Here’s today’s attendance overview.",
            fg=COLORS["muted"], font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(2, 0))

        content = tk.Frame(page, bg=COLORS["paper"])
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=3, uniform="dashboard")
        content.columnconfigure(1, weight=2, uniform="dashboard")
        content.rowconfigure(1, weight=3, minsize=205)
        content.rowconfigure(2, weight=2, minsize=170)

        metrics = [
            ("TOTAL PEOPLE", str(total_users), COLORS["green"], "Registered profiles"),
            ("PRESENT TODAY", str(present_today), COLORS["lime"], f"{today_summary['late']} late"),
            ("NOT RECORDED", str(not_recorded), COLORS["coral"], "No entry today"),
            ("ATTENDANCE RATE", f"{attendance_rate:.1f}%", COLORS["green"], "Of registered people"),
        ]
        metric_row = tk.Frame(content, bg=COLORS["paper"])
        metric_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for column, (title, value, accent, detail) in enumerate(metrics):
            metric_row.columnconfigure(column, weight=1, uniform="metric")
            self._stat(metric_row, title, value, accent, detail).grid(
                row=0, column=column, sticky="nsew",
                padx=(0 if column == 0 else 6, 0 if column == 3 else 6),
            )

        live_panel = tk.Frame(
            content, bg=COLORS["white"], highlightbackground=COLORS["green"],
            highlightthickness=1, corner_radius=10,
        )
        live_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 7), pady=(0, 12))
        live_heading = tk.Frame(live_panel, bg=COLORS["white"])
        live_heading.pack(fill="x", padx=16, pady=(13, 10))
        tk.Label(
            live_heading, text="LIVE RECOGNITION", bg=COLORS["white"],
            fg=COLORS["ink"], font=("Segoe UI", 12, "bold"),
        ).pack(side="left")
        tk.Label(
            live_heading, text="● CAMERA STANDBY" if users else "● REGISTER A PERSON FIRST",
            bg=COLORS["white"],
            fg=COLORS["lime"], font=("Segoe UI", 8, "bold"),
        ).pack(side="right")
        camera_stage = tk.Frame(live_panel, bg="#111923", height=112, corner_radius=7)
        camera_stage.pack(fill="both", expand=True, padx=14)
        camera_stage.pack_propagate(False)
        tk.Label(
            camera_stage,
            text=(
                "Camera is off\nStart a live session when you’re ready."
                if users else "No registered people yet\nRegister a person to begin attendance."
            ),
            bg="#111923", fg=COLORS["muted"], font=("Segoe UI", 11), justify="center",
        ).pack(expand=True)
        live_actions = tk.Frame(live_panel, bg=COLORS["white"])
        live_actions.pack(fill="x", padx=16, pady=(10, 14))
        tk.Label(
            live_actions, text="Unknown faces are never recorded.",
            bg=COLORS["white"], fg=COLORS["muted"], font=("Segoe UI", 9),
        ).pack(side="left", pady=8)
        self._button(
            live_actions,
            "Start camera" if users else "Register person",
            self.show_attendance if users else self.show_registration,
            primary=True,
        ).pack(side="right")

        recent_panel = tk.Frame(
            content, bg=COLORS["white"], highlightbackground=COLORS["line"],
            highlightthickness=1, corner_radius=10,
        )
        recent_panel.grid(row=1, column=1, sticky="nsew", padx=(7, 0), pady=(0, 12))
        recent_heading = tk.Frame(recent_panel, bg=COLORS["white"])
        recent_heading.pack(fill="x", padx=16, pady=(13, 8))
        tk.Label(
            recent_heading, text="RECENT ATTENDANCE", bg=COLORS["white"],
            fg=COLORS["ink"], font=("Segoe UI", 12, "bold"),
        ).pack(side="left")
        self._button(recent_heading, "Reports", self.show_analytics).pack(side="right")
        recent_rows = registered_attendance_rows(
            read_csv_rows(ATTENDANCE_DIR / f"Attendance_{today.isoformat()}.csv"),
            users,
        )
        recent_rows = recent_rows[-5:][::-1]
        if recent_rows:
            for row in recent_rows:
                self._recent_attendance_row(recent_panel, row)
        else:
            tk.Label(
                recent_panel, text="No registered attendance to show today.",
                bg=COLORS["white"], fg=COLORS["muted"], font=("Segoe UI", 10),
            ).pack(anchor="w", padx=16, pady=18)

        weekly_days = [today - dt.timedelta(days=offset) for offset in range(6, -1, -1)]
        weekly_summaries = [
            build_attendance_summary(date_text=day.isoformat(), registered_users=users)
            for day in weekly_days
        ]
        weekly_counts = [summary["total"] for summary in weekly_summaries]
        weekly_late_counts = [summary["late"] for summary in weekly_summaries]
        trend_panel = tk.Frame(
            content, bg=COLORS["white"], highlightbackground=COLORS["line"],
            highlightthickness=1, corner_radius=10,
        )
        trend_panel.grid(row=2, column=0, sticky="nsew", padx=(0, 7))
        tk.Label(
            trend_panel, text="ATTENDANCE ANALYTICS", bg=COLORS["white"],
            fg=COLORS["ink"], font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", padx=16, pady=(13, 0))
        trend_canvas = tk.Canvas(
            trend_panel, bg=COLORS["white"], highlightthickness=0, height=118,
        )
        trend_canvas.pack(fill="both", expand=True, padx=10, pady=(2, 8))
        trend_canvas.bind(
            "<Configure>",
            lambda _event: self._draw_weekly_bars(
                trend_canvas, weekly_days, weekly_counts, weekly_late_counts,
            ),
        )

        breakdown_panel = tk.Frame(
            content, bg=COLORS["white"], highlightbackground=COLORS["line"],
            highlightthickness=1, corner_radius=10,
        )
        breakdown_panel.grid(row=2, column=1, sticky="nsew", padx=(7, 0))
        tk.Label(
            breakdown_panel, text="TODAY’S BREAKDOWN", bg=COLORS["white"],
            fg=COLORS["ink"], font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", padx=16, pady=(13, 8))
        self._attendance_breakdown_row(breakdown_panel, "Present", present_today, total_users, COLORS["green"])
        self._attendance_breakdown_row(breakdown_panel, "Late", today_summary["late"], total_users, COLORS["coral"])
        self._attendance_breakdown_row(breakdown_panel, "Not recorded", not_recorded, total_users, COLORS["muted"])
        self._cache_current_page("home")

    def _stat(self, parent, title, value, accent, detail=""):
        panel = tk.Frame(parent, bg=COLORS["white"], width=190, height=92,
                         highlightbackground=COLORS["line"], highlightthickness=1)
        panel.pack_propagate(False)
        tk.Label(panel, text=title, bg=COLORS["white"], fg=COLORS["muted"],
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=13, pady=(10, 0))
        tk.Label(panel, text=value, bg=COLORS["white"], fg=accent,
                 font=("Segoe UI", 21, "bold")).pack(anchor="w", padx=13, pady=(1, 0))
        if detail:
            tk.Label(panel, text=detail, bg=COLORS["white"], fg=COLORS["muted"],
                     font=("Segoe UI", 8)).pack(anchor="w", padx=13, pady=(0, 8))
        return panel

    def _recent_attendance_row(self, parent, row):
        name = row.get("Name", "Unknown").strip() or "Unknown"
        status = row.get("Status", "Recorded").strip() or "Recorded"
        status_color = COLORS["coral"] if status.lower() == "late" else COLORS["lime"]
        item = tk.Frame(parent, bg=COLORS["white"])
        item.pack(fill="x", padx=16, pady=4)
        badge = tk.Frame(item, bg="#263746", width=30, height=30, corner_radius=15)
        badge.pack(side="left", padx=(0, 10))
        badge.pack_propagate(False)
        tk.Label(
            badge, text=name[:1].upper(), bg="#263746", fg=COLORS["green"],
            font=("Segoe UI", 10, "bold"),
        ).pack(expand=True)
        details = tk.Frame(item, bg=COLORS["white"])
        details.pack(side="left", fill="x", expand=True)
        tk.Label(
            details, text=name, bg=COLORS["white"], fg=COLORS["ink"],
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            details, text=row.get("Time", ""), bg=COLORS["white"],
            fg=COLORS["muted"], font=("Segoe UI", 8),
        ).pack(anchor="w")
        tk.Label(
            item, text=status, bg="#173A36" if status.lower() != "late" else "#3B3027",
            fg=status_color, font=("Segoe UI", 8, "bold"), corner_radius=8,
        ).pack(side="right", padx=(6, 0), ipadx=8, ipady=4)

    def _attendance_breakdown_row(self, parent, title, value, total, color):
        row = tk.Frame(parent, bg=COLORS["white"])
        row.pack(fill="x", padx=16, pady=5)
        label_row = tk.Frame(row, bg=COLORS["white"])
        label_row.pack(fill="x", pady=(0, 5))
        tk.Label(
            label_row, text=title, bg=COLORS["white"], fg=COLORS["muted"],
            font=("Segoe UI", 9),
        ).pack(side="left")
        tk.Label(
            label_row, text=str(value), bg=COLORS["white"], fg=COLORS["ink"],
            font=("Segoe UI", 9, "bold"),
        ).pack(side="right")
        track = tk.Frame(row, bg=COLORS["line"], height=6, corner_radius=3)
        track.pack(fill="x")
        ratio = value / total if total else 0
        if ratio:
            tk.Frame(track, bg=color, height=6, corner_radius=3).place(
                relx=0, rely=0, relwidth=ratio, relheight=1,
            )

    def _draw_weekly_bars(self, canvas, days, counts, late_counts):
        canvas.delete("all")
        if not any(counts):
            canvas.create_text(
                max(canvas.winfo_width(), 1) / 2,
                max(canvas.winfo_height(), 1) / 2,
                text="No attendance data for this period",
                fill=COLORS["muted"], font=("Segoe UI", 10),
            )
            return
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        left, right, top, bottom = 22, 22, 14, 25
        baseline = height - bottom
        chart_height = max(baseline - top, 1)
        maximum = max(max(counts, default=0), 1)
        for fraction in (0.33, 0.66, 1):
            y = baseline - chart_height * fraction
            canvas.create_line(left, y, width - right, y, fill=COLORS["line"], dash=(2, 4))
        step = (width - left - right) / max(len(days) - 1, 1)
        bar_width = min(32, step * 0.48)
        for index, (day, count, late) in enumerate(zip(days, counts, late_counts)):
            x = left + index * step
            total_height = (count / maximum) * chart_height
            on_time_height = ((count - late) / maximum) * chart_height
            late_height = (late / maximum) * chart_height
            if on_time_height > 0:
                canvas.create_rectangle(
                    x - bar_width / 2, baseline - on_time_height,
                    x + bar_width / 2, baseline,
                    fill=COLORS["green"], outline="",
                )
            if late_height > 0:
                canvas.create_rectangle(
                    x - bar_width / 2, baseline - total_height,
                    x + bar_width / 2, baseline - on_time_height,
                    fill=COLORS["coral"], outline="",
                )
            if count:
                canvas.create_text(
                    x, baseline - total_height - 9, text=str(count),
                    fill=COLORS["ink"], font=("Segoe UI", 8, "bold"),
                )
            canvas.create_text(
                x, height - 8, text=day.strftime("%a"),
                fill=COLORS["muted"], font=("Segoe UI", 8),
            )

    def _profile_photo(self, person_id, size):
        cache_key = (str(person_id), size)
        if cache_key in self.profile_photo_cache:
            return self.profile_photo_cache[cache_key]
        samples = self._profile_samples_by_id().get(str(person_id), [])
        if not samples:
            self.profile_photo_cache[cache_key] = None
            return None
        try:
            with Image.open(samples[0]) as source:
                image = ImageOps.fit(
                    source.convert("RGB"),
                    (size, size),
                    method=Image.Resampling.LANCZOS,
                )
            photo = ctk.CTkImage(
                light_image=image,
                dark_image=image,
                size=(size, size),
            )
        except (OSError, ValueError):
            photo = None
        self.profile_photo_cache[cache_key] = photo
        return photo

    def _profile_samples_by_id(self):
        if self.profile_sample_index is None:
            index = {}
            for path in TRAINING_DIR.rglob("face.*.*.jpg"):
                pieces = path.stem.split(".")
                if len(pieces) == 3 and pieces[1].isdigit():
                    index.setdefault(str(int(pieces[1])), []).append(path)
            for paths in index.values():
                paths.sort()
            self.profile_sample_index = index
        return self.profile_sample_index

    def show_users(self):
        if self._show_cached_page("people", "People") is not None:
            return
        page = self._new_page()
        self._set_active_nav("People")
        self._label(page, "People", font=("Segoe UI", 22, "bold")).pack(anchor="w")
        self._label(
            page, "Select a person to view their profile and enrollment photo.",
            fg=COLORS["muted"],
        ).pack(anchor="w", pady=(5, 16))

        panel = tk.Frame(page, bg=COLORS["paper"])
        panel.pack(fill="both", expand=True)
        panel.columnconfigure(0, weight=2, uniform="people")
        panel.columnconfigure(1, weight=3, uniform="people")
        panel.rowconfigure(0, weight=1)

        roster_panel = tk.Frame(
            panel, bg=COLORS["white"], highlightbackground=COLORS["line"],
            highlightthickness=1, corner_radius=10,
        )
        roster_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        tk.Label(
            roster_panel, text="REGISTERED PEOPLE", bg=COLORS["white"],
            fg=COLORS["muted"], font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", padx=16, pady=(14, 8))
        self.people_roster = ctk.CTkScrollableFrame(
            roster_panel, fg_color=COLORS["white"], corner_radius=0,
        )
        self.people_roster.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        detail_panel = tk.Frame(
            panel, bg=COLORS["white"], highlightbackground=COLORS["line"],
            highlightthickness=1, corner_radius=10,
        )
        detail_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.people_detail = ctk.CTkScrollableFrame(
            detail_panel, fg_color=COLORS["white"], corner_radius=0,
        )
        self.people_detail.pack(fill="both", expand=True, padx=8, pady=8)

        self.user_rows = list(load_users().items())
        self.people_row_widgets = {}
        for person_id, name in self.user_rows:
            photo = self._profile_photo(person_id, 46)
            row = tk.Frame(
                self.people_roster, bg=COLORS["white"], height=64,
                corner_radius=8,
            )
            row.pack(fill="x", padx=4, pady=3)
            row.pack_propagate(False)
            if photo is not None:
                avatar = tk.Label(row, text="", image=photo, bg=COLORS["white"])
                avatar.image = photo
            else:
                avatar = tk.Label(
                    row, text=name[:1].upper(), bg="#263746", fg=COLORS["green"],
                    font=("Segoe UI", 13, "bold"), width=3, height=2,
                    corner_radius=8,
                )
            avatar.pack(side="left", padx=(8, 10), pady=8)
            identity = tk.Frame(row, bg=COLORS["white"])
            identity.pack(side="left", fill="both", expand=True, pady=8)
            name_label = tk.Label(
                identity, text=name, bg=COLORS["white"], fg=COLORS["ink"],
                font=("Segoe UI", 10, "bold"), anchor="w",
            )
            name_label.pack(anchor="w")
            id_label = tk.Label(
                identity, text=f"ID {person_id}", bg=COLORS["white"],
                fg=COLORS["muted"], font=("Segoe UI", 9), anchor="w",
            )
            id_label.pack(anchor="w", pady=(2, 0))
            self.people_row_widgets[person_id] = row
            select_person = lambda _event=None, selected_id=person_id: self._show_user_details(selected_id)
            for widget in (row, avatar, identity, name_label, id_label):
                widget.bind("<Button-1>", select_person)

        self.selected_user_id = self.user_rows[0][0] if self.user_rows else None
        if self.selected_user_id:
            self._show_user_details(self.selected_user_id)
        else:
            tk.Label(
                self.people_detail, text="No people are registered yet.",
                bg=COLORS["white"], fg=COLORS["muted"], font=("Segoe UI", 11),
            ).pack(anchor="center", expand=True, pady=48)

        controls = tk.Frame(page, bg=COLORS["paper"])
        controls.pack(fill="x", pady=(12, 0))
        self._button(controls, "Back", self.show_home).pack(side="left")
        self._button(
            controls, "Remove selected profile", self.remove_selected_user,
            primary=True, bg=COLORS["red"], activebackground="#873A33",
        ).pack(side="right")
        self._cache_current_page("people")

    def _show_user_details(self, person_id):
        if person_id not in dict(self.user_rows):
            return
        self.selected_user_id = person_id
        for row_id, row in self.people_row_widgets.items():
            row.configure(
                border_width=1 if row_id == person_id else 0,
                border_color=COLORS["green"],
            )

        for child in self.people_detail.winfo_children():
            child.destroy()
        name = dict(self.user_rows)[person_id]
        photo = self._profile_photo(person_id, 132)
        if photo is not None:
            image_label = tk.Label(self.people_detail, image=photo, text="", bg=COLORS["white"])
            image_label.image = photo
            image_label.pack(anchor="w", padx=18, pady=(14, 8))
        else:
            tk.Label(
                self.people_detail, text="No enrollment photo found",
                bg=COLORS["white"], fg=COLORS["muted"], font=("Segoe UI", 10),
            ).pack(anchor="w", padx=18, pady=(22, 12))
        tk.Label(
            self.people_detail, text=name, bg=COLORS["white"],
            fg=COLORS["ink"], font=("Segoe UI", 20, "bold"),
        ).pack(anchor="w", padx=18)
        tk.Label(
            self.people_detail,
            text=f"ID {person_id}",
            bg=COLORS["white"], fg=COLORS["muted"], font=("Segoe UI", 10),
        ).pack(anchor="w", padx=18, pady=(3, 16))

    def remove_selected_user(self):
        person_id = self.selected_user_id
        if not person_id:
            messagebox.showinfo("Select a user", "Choose a profile to remove.", parent=self.root)
            return
        name = dict(self.user_rows).get(person_id, "Unknown")
        confirmed = messagebox.askyesno(
            "Remove user",
            f"Remove {name} (ID {person_id}), their saved face samples, and their attendance entries? This cannot be undone.",
            parent=self.root,
        )
        if not confirmed:
            return
        if not self._authorize("remove this profile"):
            return

        try:
            user_dir = student_training_dir(person_id, name)
            if user_dir.exists():
                shutil.rmtree(user_dir)
            for image_path in TRAINING_DIR.glob(f"*.{person_id}.*.jpg"):
                image_path.unlink(missing_ok=True)
            for proof_path in ATTENDANCE_PROOF_DIR.rglob(f"{person_id}_*.jpg"):
                proof_path.unlink(missing_ok=True)
            self.profile_photo_cache = {
                key: photo for key, photo in self.profile_photo_cache.items()
                if key[0] != person_id
            }
            self.profile_sample_index = None

            remaining_users = load_users()
            remaining_users.pop(person_id, None)
            with STUDENT_FILE.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                for registered_id, registered_name in remaining_users.items():
                    writer.writerow((registered_id, registered_name))

            for attendance_file in ATTENDANCE_DIR.glob("Attendance_*.csv"):
                rows = read_csv_rows(attendance_file)
                remaining_rows = [
                    row for row in rows
                    if row.get("Id", "").strip().lstrip("0") != person_id.lstrip("0")
                ]
                with attendance_file.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=ATTENDANCE_COLUMNS)
                    writer.writeheader()
                    writer.writerows(remaining_rows)

            faces, ids = load_training_images()
            if faces:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.train(faces, np.asarray(ids, dtype=np.int32))
                recognizer.save(str(MODEL_FILE))
            else:
                MODEL_FILE.unlink(missing_ok=True)
        except Exception as error:
            messagebox.showerror("Could not remove user", str(error), parent=self.root)
            return

        messagebox.showinfo("User removed", f"{name} and their stored attendance data were removed.", parent=self.root)
        self._invalidate_page_cache("home", "people", "reports")
        self.show_users()

    def show_registration(self):
        page = self._new_page()
        self._set_active_nav("Register face")
        self.capture_samples = []
        self._label(page, "Register a user", font=("Segoe UI", 22, "bold")).pack(anchor="w")
        self._label(
            page, "Enter an ID and name. Missing profiles can be restored with their current details; faces are checked before capture.",
            fg=COLORS["muted"],
        ).pack(anchor="w", pady=(5, 18))

        form = tk.Frame(page, bg=COLORS["white"], padx=24, pady=22,
                        highlightbackground=COLORS["line"], highlightthickness=1)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        tk.Label(form, text="USER ID", bg=COLORS["white"], fg=COLORS["muted"],
                 font=("Segoe UI", 8, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 7))
        tk.Label(form, text="FULL NAME", bg=COLORS["white"], fg=COLORS["muted"],
                 font=("Segoe UI", 8, "bold")).grid(row=0, column=1, sticky="w", padx=(18, 0), pady=(0, 7))

        self.user_id_entry = tk.Entry(form, font=("Segoe UI", 11), relief="solid", bd=1)
        self.user_id_entry.grid(row=1, column=0, sticky="ew", ipady=9)
        self.name_entry = tk.Entry(form, font=("Segoe UI", 11), relief="solid", bd=1)
        self.name_entry.grid(row=1, column=1, sticky="ew", padx=(18, 0), ipady=9)

        self.consent = tk.BooleanVar(value=False)
        tk.Checkbutton(
            form, text="I have permission to collect and use this person’s face samples.",
            variable=self.consent, bg=COLORS["white"], fg=COLORS["ink"],
            activebackground=COLORS["white"], selectcolor=COLORS["white"],
            font=("Segoe UI", 9),
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(17, 5))

        camera_panel = tk.Frame(page, bg=COLORS["ink"], padx=16, pady=14)
        camera_panel.pack(fill="both", expand=True, pady=(18, 14))
        camera_panel.columnconfigure(0, weight=1)
        camera_panel.rowconfigure(1, weight=1)
        self.capture_status = tk.Label(
            camera_panel, text="Camera preview will appear here.", bg=COLORS["ink"],
            fg=COLORS["white"], font=("Segoe UI", 10),
        )
        self.capture_status.grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.video_label = tk.Label(camera_panel, bg="#102321", text="LIVE PREVIEW",
                                    fg="#B8C8BF", font=("Segoe UI", 10, "bold"))
        self.video_label.grid(row=1, column=0, sticky="nsew")

        controls = tk.Frame(page, bg=COLORS["paper"])
        controls.pack(fill="x")
        self._button(controls, "Back", self.show_home).pack(side="left")
        self.capture_button = self._button(
            controls, "Start face capture", self.start_registration_capture, primary=True,
        )
        self.capture_button.pack(side="right")

    def start_registration_capture(self):
        user_id = self.user_id_entry.get().strip()
        name = self.name_entry.get().strip()
        if not user_id.isdigit():
            messagebox.showwarning("Check user ID", "Enter a numeric user ID.", parent=self.root)
            return
        user_id = str(int(user_id))
        if not name or not any(character.isalpha() for character in name):
            messagebox.showwarning("Check name", "Enter the user’s name.", parent=self.root)
            return
        registered_users = load_users()
        if not self.consent.get():
            messagebox.showwarning("Consent required", "Confirm permission before collecting face samples.", parent=self.root)
            return
        if not callable(getattr(getattr(cv2, "face", None), "LBPHFaceRecognizer_create", None)):
            messagebox.showerror(
                "OpenCV face recognition unavailable",
                "This Python environment has a conflicting OpenCV installation.\n\n"
                f"Python: {sys.executable}\n\n"
                "In VS Code, select the project's .venv interpreter, then run train.py again.",
                parent=self.root,
            )
            return
        organize_training_images()
        existing_faces, existing_ids = load_training_images()
        sampled_ids = {str(int(person_id)) for person_id in existing_ids}
        profile_needs_samples = user_id in registered_users and user_id not in sampled_ids
        if user_id in registered_users and not profile_needs_samples:
            messagebox.showwarning("ID already registered", "Choose a different user ID.", parent=self.root)
            return
        if profile_needs_samples and name.casefold() != registered_users[user_id].casefold():
            messagebox.showwarning(
                "Use the registered name",
                f"ID {user_id} is already registered as {registered_users[user_id]}. "
                "Enter that name to restore this profile.",
                parent=self.root,
            )
            return
        profiles_without_samples = set(registered_users) - sampled_ids
        if profiles_without_samples and not profile_needs_samples:
            self.registration_recognizer = None
            messagebox.showerror(
                "Existing face samples unavailable",
                f"{len(profiles_without_samples)} registered profile(s) have no saved face samples.\n\n"
                "Restore a missing profile using its existing ID and name before adding a new ID. "
                "This keeps duplicate checks reliable and preserves existing attendance history.",
                parent=self.root,
            )
            return
        if existing_faces:
            self.registration_recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.registration_recognizer.train(
                existing_faces, np.asarray(existing_ids, dtype=np.int32),
            )
        else:
            self.registration_recognizer = None
        if not self._open_camera():
            return
        self.registration = (user_id, name)
        self.capture_samples = []
        self.last_sample_time = 0.0
        self.last_duplicate_check = 0.0
        self.duplicate_check_samples = 0
        self.duplicate_candidate_votes = {}
        self.camera_mode = "registration"
        self.capture_button.configure(state="disabled")
        if self.registration_recognizer is None:
            self.capture_status.configure(
                text="No existing face samples to compare. Center one face to start registration."
            )
        else:
            self.capture_status.configure(
                text="Checking this face against enrolled users before registration…"
            )
        self._schedule_camera(self._registration_frame)

    def _registration_frame(self, frame):
        sample_count = int(self.settings.get("sample_count", SAMPLE_COUNT))
        duplicate_threshold = float(self.settings.get("duplicate_threshold", DUPLICATE_MATCH_THRESHOLD))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.2, 5, minSize=(90, 90))
        self.preview_overlay = []
        if len(faces) == 1:
            x, y, width, height = faces[0]
            self.preview_overlay = [{"x": x, "y": y, "width": width, "height": height, "label": "Face", "color": (93, 206, 164)}]
            face = gray[y:y + height, x:x + width]
            now = time.monotonic()
            if (
                self.registration_recognizer is not None
                and self.duplicate_check_samples < DUPLICATE_CHECK_SAMPLES
            ):
                if now - self.last_duplicate_check >= 0.3:
                    self.last_duplicate_check = now
                    predicted_id, confidence = self.registration_recognizer.predict(
                        cv2.resize(face, (200, 200))
                    )
                    person_id = str(int(predicted_id))
                    self.duplicate_check_samples += 1
                    if _record_duplicate_vote(
                        self.duplicate_candidate_votes,
                        person_id,
                        confidence,
                        duplicate_threshold,
                    ):
                        self._reject_duplicate_registration(person_id)
                        return

                    if self.duplicate_check_samples >= DUPLICATE_CHECK_SAMPLES:
                        self.capture_status.configure(
                            text="No existing match found. Keep facing the camera to capture samples."
                        )
                    else:
                        self.capture_status.configure(
                            text=f"Checking face against enrolled users… {self.duplicate_check_samples} / {DUPLICATE_CHECK_SAMPLES}"
                        )
                return

            if now - self.last_sample_time >= 0.12:
                self.capture_samples.append(cv2.resize(face, (200, 200)))
                self.last_sample_time = now
            self.capture_status.configure(
                text=f"Capturing face samples  /  {len(self.capture_samples):02d} of {sample_count}"
            )
            if len(self.capture_samples) >= sample_count:
                self._stop_camera()
                self._finish_registration()
        elif len(faces) > 1:
            self.capture_status.configure(text="More than one face found. Please capture one person at a time.")
        else:
            self.capture_status.configure(text="No face found. Move into the frame and face the camera.")

    def _reject_duplicate_registration(self, person_id):
        self._stop_camera()
        self.capture_button.configure(state="normal")
        existing_name = load_users().get(person_id, "an enrolled user")
        self.capture_status.configure(text="An enrolled face matched. No new samples were saved.")
        self.capture_samples = []
        self.registration_recognizer = None
        messagebox.showwarning(
            "Face already registered",
            f"This face matches {existing_name} (ID {person_id}).\n\n"
            "Registration was stopped before saving any face samples. Use the existing profile instead.",
            parent=self.root,
        )

    def _finish_registration(self):
        user_id, name = self.registration
        saved_images = []
        try:
            user_dir = student_training_dir(user_id, name)
            user_dir.mkdir(parents=True, exist_ok=True)
            existing = list(user_dir.glob(f"*.{user_id}.*.jpg"))
            first_number = len(existing) + 1
            for offset, face in enumerate(self.capture_samples):
                image_path = user_dir / f"face.{user_id}.{first_number + offset}.jpg"
                if not cv2.imwrite(str(image_path), face):
                    raise OSError("A captured face image could not be saved.")
                saved_images.append(image_path)

            train_recognizer()
            STUDENT_FILE.parent.mkdir(parents=True, exist_ok=True)
            if user_id not in load_users():
                with STUDENT_FILE.open("a", newline="", encoding="utf-8") as file:
                    csv.writer(file).writerow((user_id, name))
        except Exception as error:
            for image_path in saved_images:
                image_path.unlink(missing_ok=True)
            self.capture_button.configure(state="normal")
            self.capture_status.configure(text="Registration could not be completed.")
            messagebox.showerror("Registration failed", str(error), parent=self.root)
            return

        self.capture_samples = []
        self.profile_sample_index = None
        self.profile_photo_cache = {
            key: photo for key, photo in self.profile_photo_cache.items()
            if key[0] != user_id
        }
        self._invalidate_page_cache("home", "people", "reports")
        messagebox.showinfo(
            "User registered", f"{name} is registered and the face model is ready.", parent=self.root,
        )
        self.show_home()

    def show_attendance(self):
        page = self._new_page()
        self._set_active_nav("Live attendance")
        self.recent_capture_person_id = None
        self._label(page, "Take attendance", font=("Segoe UI", 22, "bold")).pack(anchor="w")
        self._label(
            page,
            f"Recognized people are saved once per day. Arrivals after "
            f"{self.settings.get('late_cutoff', '09:15')} are marked Late. "
            "One proof photo is saved only when attendance is recorded.",
            fg=COLORS["muted"],
        ).pack(anchor="w", pady=(5, 18))

        layout = tk.Frame(page, bg=COLORS["paper"])
        layout.pack(fill="both", expand=True)
        layout.columnconfigure(0, weight=3)
        layout.columnconfigure(1, weight=2)
        layout.rowconfigure(0, weight=1)

        video_panel = tk.Frame(layout, bg=COLORS["ink"], padx=14, pady=14)
        video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        video_panel.rowconfigure(2, weight=1)
        video_panel.columnconfigure(0, weight=1)
        self.attendance_status = tk.Label(
            video_panel, text="Preparing camera…", bg=COLORS["ink"], fg=COLORS["white"],
            font=("Segoe UI", 10),
        )
        self.attendance_status.grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.camera_progress = ctk.CTkProgressBar(
            video_panel, mode="indeterminate", height=8,
            progress_color=COLORS["green"], fg_color=COLORS["line"],
        )
        self.camera_progress.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.video_label = tk.Label(video_panel, bg="#102321", text="LIVE PREVIEW",
                                    fg="#B8C8BF", font=("Segoe UI", 10, "bold"))
        self.video_label.grid(row=2, column=0, sticky="nsew")

        roster = tk.Frame(layout, bg=COLORS["white"], padx=18, pady=16,
                          highlightbackground=COLORS["line"], highlightthickness=1)
        roster.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        roster.rowconfigure(3, weight=1)
        roster.columnconfigure(0, weight=1)
        tk.Label(roster, text="LATEST RECOGNITION", bg=COLORS["white"], fg=COLORS["green"],
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))
        recent_capture = tk.Frame(roster, bg="#121923", corner_radius=8)
        recent_capture.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        recent_capture.columnconfigure(1, weight=1)
        self.recent_capture_image_label = tk.Label(
            recent_capture, text="LIVE", bg="#202C38", fg=COLORS["muted"],
            font=("Segoe UI", 8, "bold"), width=96, height=96, corner_radius=7,
        )
        self.recent_capture_image_label.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
        self.recent_capture_name_label = tk.Label(
            recent_capture, text="Waiting for a face", bg="#121923",
            fg=COLORS["ink"], font=("Segoe UI", 11, "bold"), anchor="w",
        )
        self.recent_capture_name_label.grid(row=0, column=1, sticky="sw", padx=(0, 10), pady=(10, 2))
        self.recent_capture_time_label = tk.Label(
            recent_capture, text="The latest recognized face appears here.",
            bg="#121923", fg=COLORS["muted"], font=("Segoe UI", 8), anchor="w",
        )
        self.recent_capture_time_label.grid(row=1, column=1, sticky="nw", padx=(0, 10), pady=(0, 10))
        tk.Label(roster, text="TODAY’S LOG", bg=COLORS["white"], fg=COLORS["green"],
                 font=("Segoe UI", 9, "bold")).grid(row=2, column=0, sticky="w", pady=(0, 8))
        self.log_list = tk.Listbox(
            roster, bg=COLORS["white"], fg=COLORS["ink"], relief="flat",
            highlightthickness=0, font=("Segoe UI", 10), activestyle="none",
        )
        self.log_list.grid(row=3, column=0, sticky="nsew")
        self.attendance_users = load_users()
        self.camera_mode = "attendance"
        today = dt.date.today().isoformat()
        self.attendance_seen = self._today_attendance_ids(today)
        today_rows = registered_attendance_rows(
            read_csv_rows(ATTENDANCE_DIR / f"Attendance_{today}.csv"),
            self.attendance_users,
        )
        for row in today_rows:
            status = row.get("Status", "").strip()
            suffix = f"   ({status})" if status else ""
            self.log_list.insert("end", f"{row.get('Time', '')}   {row.get('Name', '')}{suffix}")

        controls = tk.Frame(page, bg=COLORS["paper"])
        controls.pack(fill="x", pady=(14, 0))
        self._button(controls, "Back", self.show_home).pack(side="left")
        self.stop_button = self._button(controls, "Stop camera", self.stop_attendance, primary=True)
        self.stop_button.pack(side="right")

        if not self.attendance_users:
            self.attendance_status.configure(text="No registered users found. Register a user first.")
            self.video_label.configure(text="Register a person before starting attendance.")
            self.camera_progress.grid_remove()
            self.stop_button.configure(state="disabled")
            return
        if not MODEL_FILE.exists():
            self.attendance_status.configure(text="No trained face model. Register a user first.")
            self.video_label.configure(text="A trained face model is required.")
            self.camera_progress.grid_remove()
            self.stop_button.configure(state="disabled")
            return
        self._start_attendance_initialization()

    def _start_attendance_initialization(self):
        cancel_event = threading.Event()
        startup = {
            "cancel": cancel_event,
            "lock": threading.Lock(),
            "results": queue.Queue(maxsize=1),
            "thread": None,
        }
        self.camera_startup = startup
        self.camera_progress.start()
        self.attendance_status.configure(text="Opening camera and loading face model…")
        thread = threading.Thread(
            target=self._load_attendance_resources,
            args=(startup,),
            name="attendance-camera-startup",
            daemon=True,
        )
        startup["thread"] = thread
        thread.start()
        self.root.after(50, lambda: self._poll_attendance_initialization(startup))

    def _load_attendance_resources(self, startup):
        camera = None
        result = None
        try:
            detector, camera = open_camera_resources(
                int(self.settings.get("camera_index", 0)),
            )
            recognizer_factory = getattr(
                getattr(cv2, "face", None), "LBPHFaceRecognizer_create", None,
            )
            if not callable(recognizer_factory):
                raise RuntimeError(
                    "LBPH recognizer is missing from this Python environment. "
                    f"Python: {sys.executable}. Select the project's .venv interpreter."
                )
            recognizer = recognizer_factory()
            recognizer.read(str(MODEL_FILE))
            model_ids = {
                str(int(person_id))
                for person_id in recognizer.getLabels().reshape(-1)
            }
            result = (detector, camera, recognizer, model_ids, None)
        except Exception as error:
            if camera is not None:
                camera.release()
            result = (None, None, None, set(), error)

        with startup["lock"]:
            if startup["cancel"].is_set():
                if result[1] is not None:
                    result[1].release()
                return
            startup["results"].put(result)

    def _poll_attendance_initialization(self, startup):
        if self.camera_startup is not startup:
            return
        try:
            detector, camera, recognizer, model_ids, error = startup["results"].get_nowait()
        except queue.Empty:
            self.root.after(50, lambda: self._poll_attendance_initialization(startup))
            return

        self.camera_startup = None
        self.camera_progress.stop()
        self.camera_progress.grid_remove()
        if error is not None:
            self.attendance_status.configure(text=f"Could not start attendance: {error}")
            self.video_label.configure(text="Camera unavailable")
            self.stop_button.configure(state="disabled")
            return

        self.detector = detector
        self.camera = camera
        self.recognizer = recognizer
        profiles_without_samples = set(self.attendance_users) - model_ids
        if profiles_without_samples:
            if len(profiles_without_samples) == len(self.attendance_users):
                self.attendance_status.configure(
                    text="This model has no matching registered profiles. Remove and re-register profiles before taking attendance."
                )
            else:
                self.attendance_status.configure(
                    text=f"{len(profiles_without_samples)} profile(s) have no matching face samples. Remove and re-register them."
                )
        else:
            self.attendance_status.configure(text="Camera ready. Looking for a registered face…")
        self.candidate_id = None
        self.candidate_frames = 0
        self._schedule_camera(self._attendance_frame)

    def _cancel_camera_startup(self):
        startup = self.camera_startup
        if startup is None:
            return
        self.camera_startup = None
        with startup["lock"]:
            startup["cancel"].set()
            try:
                result = startup["results"].get_nowait()
            except queue.Empty:
                result = None
            if result is not None and result[1] is not None:
                result[1].release()
        if self.camera_progress is not None:
            self.camera_progress.stop()
            self.camera_progress.grid_remove()
            self.camera_progress = None

    def _attendance_frame(self, frame):
        match_threshold = float(self.settings.get("match_threshold", MATCH_THRESHOLD))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        self.preview_overlay = []
        if not len(faces):
            self.candidate_id = None
            self.candidate_frames = 0
            self.attendance_status.configure(text="Looking for a registered face…")

        for x, y, width, height in faces:
            predicted_id, confidence = self.recognizer.predict(
                cv2.resize(gray[y:y + height, x:x + width], (200, 200))
            )
            person_id = str(predicted_id)
            known = confidence <= match_threshold and person_id in self.attendance_users
            color = (93, 206, 164) if known else (105, 123, 119)
            label = self.attendance_users.get(person_id, "Unknown") if known else "Unknown"
            self.preview_overlay.append({"x": x, "y": y, "width": width, "height": height, "label": label, "color": color})

            if known:
                self._confirm_attendance(
                    person_id, frame[y:y + height, x:x + width],
                )
            else:
                self.candidate_id = None
                self.candidate_frames = 0
                self.attendance_status.configure(text="Face not recognized. No attendance recorded.")

    def _update_recent_capture(self, person_id, face_crop, captured_at):
        resized = cv2.resize(face_crop, (96, 96))
        image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        self.recent_capture_photo = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=(96, 96),
        )
        self.recent_capture_image_label.configure(
            image=self.recent_capture_photo, text="",
        )
        self.recent_capture_name_label.configure(
            text=self.attendance_users.get(person_id, "Unknown"),
        )
        self.recent_capture_time_label.configure(
            text=captured_at.strftime("Captured %I:%M:%S %p"),
        )
        self.recent_capture_person_id = person_id

    def _confirm_attendance(self, person_id, face_crop):
        if person_id == self.candidate_id:
            self.candidate_frames += 1
        else:
            self.candidate_id = person_id
            self.candidate_frames = 1
        name = self.attendance_users[person_id]
        if person_id in self.attendance_seen:
            self.attendance_status.configure(text=f"Already recorded today  /  {name}")
            return
        self.attendance_status.configure(text=f"Confirming {name}… {min(self.candidate_frames, 3)} / 3")
        if self.candidate_frames < 3:
            return

        now = dt.datetime.now()
        cutoff = parse_cutoff_time(self.settings.get("late_cutoff", "09:15"))
        status = "Late" if now.time() > cutoff else "On time"
        row = (person_id, name, now.date().isoformat(), now.strftime("%H:%M:%S"), status)
        daily_file = ATTENDANCE_DIR / f"Attendance_{row[2]}.csv"
        proof_file = ATTENDANCE_PROOF_DIR / row[2] / (
            f"{person_id}_{now.strftime('%H%M%S_%f')}.jpg"
        )
        ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)
        needs_header = not daily_file.exists() or daily_file.stat().st_size == 0
        try:
            proof_file.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(proof_file), face_crop):
                raise OSError("The attendance proof photo could not be saved.")
            with daily_file.open("a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                if needs_header:
                    writer.writerow(ATTENDANCE_COLUMNS)
                writer.writerow(row)
        except (OSError, cv2.error) as error:
            proof_file.unlink(missing_ok=True)
            self.attendance_status.configure(text="Could not save attendance and proof photo.")
            messagebox.showerror("Save failed", str(error), parent=self.root)
            return

        self.attendance_seen.add(person_id)
        self._invalidate_page_cache("home", "people", "reports")
        self._update_recent_capture(person_id, face_crop, now)
        self.log_list.insert("end", f"{row[3]}   {name}   ({status})")
        self.attendance_status.configure(text=f"Attendance recorded  /  {name}  /  {row[3]}  /  {status}")

    def _today_attendance_ids(self, date_text):
        attendance_ids = set()
        for path in ATTENDANCE_DIR.glob(f"Attendance_{date_text}*.csv"):
            for row in read_csv_rows(path):
                person_id = row.get("Id", "").strip()
                if person_id.isdigit():
                    person_id = str(int(person_id))
                if person_id:
                    attendance_ids.add(person_id)
        return attendance_ids

    def stop_attendance(self):
        self._cancel_camera_startup()
        self._stop_camera()
        if self.page is not None:
            self.attendance_status.configure(text="Camera stopped. Today’s log has been saved.")
            self.stop_button.configure(state="disabled")

    def _open_camera(self):
        camera_index = int(self.settings.get("camera_index", 0))
        try:
            self.detector, self.camera = open_camera_resources(camera_index)
        except Exception as error:
            self.camera = None
            messagebox.showerror(
                "Camera unavailable",
                str(error),
                parent=self.root,
            )
            return False
        return True

    def _schedule_camera(self, callback):
        self.camera_job = self.root.after(40, lambda: self._update_camera(callback))

    def _render_preview_overlay(self, frame):
        width = frame.shape[1]
        for overlay in self.preview_overlay:
            x = overlay["x"]
            y = overlay["y"]
            w = overlay["width"]
            h = overlay["height"]
            mirrored_x = width - (x + w)
            mirrored_box = (mirrored_x, y), (mirrored_x + w, y + h)
            cv2.rectangle(frame, mirrored_box[0], mirrored_box[1], overlay["color"], 2)
            cv2.putText(
                frame,
                overlay["label"],
                (mirrored_x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                overlay["color"],
                2,
                cv2.LINE_AA,
            )
        return frame

    def _update_camera(self, callback):
        self.camera_job = None
        if self.camera is None:
            return
        success, frame = self.camera.read()
        if not success or frame is None:
            status = getattr(self, "attendance_status", None) if self.camera_mode == "attendance" else getattr(self, "capture_status", None)
            if status is not None:
                status.configure(text="Camera frame unavailable. Check the camera connection.")
            self._stop_camera()
            return
        callback(frame)
        if self.camera is None:
            return
        display_frame = cv2.flip(frame.copy(), 1)
        display_frame = self._render_preview_overlay(display_frame)
        preview = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(preview)
        image.thumbnail((560, 350))
        self.current_photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.current_photo, text="")
        self.video_label.image = self.current_photo
        self._schedule_camera(callback)

    def _stop_camera(self):
        if self.camera_job is not None:
            try:
                self.root.after_cancel(self.camera_job)
            except tk.TclError:
                pass
            self.camera_job = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def close(self):
        self._cancel_camera_startup()
        self._stop_camera()
        self.root.destroy()


def main():
    for directory in (TRAINING_DIR, MODEL_DIR, ATTENDANCE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    organize_training_images()
    root = ctk.CTk()
    AttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()