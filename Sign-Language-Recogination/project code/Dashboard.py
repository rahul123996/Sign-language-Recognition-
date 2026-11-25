"""
Sign Language Recognition Studio – Premium "Modern AI Studio" UI
Redesigned visual layer. ALL logic methods are byte-for-byte identical to
the original; only the stylesheet, layout, and cosmetic widget construction
have been updated. Import of ui_theme provides the centralised token system.
"""
__author__ = 'Mohsin Raza, Muhammad Awais Sarwar, Upgraded Studio Edition'

import os
import sys
import shutil
import tempfile
import cv2
import numpy as np

# HiDPI support (must be set before QApplication)
import PyQt5.QtWidgets as _qtw_pre
_app_pre = _qtw_pre.QApplication.instance()
if _app_pre is None:
    from PyQt5.QtCore import Qt as _Qt
    _qtw_pre.QApplication.setAttribute(_Qt.AA_EnableHighDpiScaling, True)
    _qtw_pre.QApplication.setAttribute(_Qt.AA_UseHighDpiPixmaps, True)

# Ensure working directory is set to script's directory
_current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_current_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

if 'MPLCONFIGDIR' not in os.environ:
    os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'matplotlib')

# Model loader shim
import model_loader
from keras.models import load_model

# PyQt5 GUI
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QColor, QPainter, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSlider, QTextEdit, QLineEdit,
    QStackedWidget, QFrame, QScrollArea, QSplitter, QProgressBar,
    QMessageBox, QFileDialog, QSizePolicy, QTabWidget, QGraphicsDropShadowEffect,
    QComboBox, QCheckBox
)

import pyttsx3

# UI theme system
from ui_theme import (
    DARK_TOKENS, LIGHT_TOKENS, build_qss,
    ToggleSwitch, PillBadge, Toast,
    load_theme_pref, save_theme_pref
)

# Model initialization
image_x, image_y = 64, 64
classifier = load_model('ASLModel.h5')

# TTS Engine
try:
    engine = pyttsx3.init()
except Exception:
    engine = None


# Gesture reference descriptions
GESTURE_CATALOG = [
    {"char": "A", "name": "Letter A", "desc": "Closed fist with thumb resting upright against side of index finger.", "type": "Alphabet"},
    {"char": "B", "name": "Letter B", "desc": "Flat open hand, 4 fingers upright together, thumb folded across palm.", "type": "Alphabet"},
    {"char": "C", "name": "Letter C", "desc": "All fingers and thumb curved in a semicircular 'C' cup shape.", "type": "Alphabet"},
    {"char": "D", "name": "Letter D", "desc": "Index finger pointing up; thumb touches middle, ring & pinky to form 'O'.", "type": "Alphabet"},
    {"char": "E", "name": "Letter E", "desc": "Fingers curled tightly down, fingertips resting on thumb tucked below.", "type": "Alphabet"},
    {"char": "F", "name": "Letter F", "desc": "Index finger and thumb touch tips forming circle ('OK'), 3 fingers upright.", "type": "Alphabet"},
    {"char": "G", "name": "Letter G", "desc": "Index finger and thumb pointing forward horizontally, parallel.", "type": "Alphabet"},
    {"char": "H", "name": "Letter H", "desc": "Index and middle fingers extended together horizontally forward.", "type": "Alphabet"},
    {"char": "I", "name": "Letter I", "desc": "Pinky finger pointing straight up, remaining fingers folded with thumb across.", "type": "Alphabet"},
    {"char": "J", "name": "Letter J", "desc": "Pinky extended upright, hand traces a 'J' hook curve in the air.", "type": "Alphabet"},
    {"char": "K", "name": "Letter K", "desc": "Index finger up, middle finger tilted forward, thumb between them.", "type": "Alphabet"},
    {"char": "L", "name": "Letter L", "desc": "Thumb and index finger extended at 90-degree angle forming an 'L'.", "type": "Alphabet"},
    {"char": "M", "name": "Letter M", "desc": "Fist closed with thumb tucked beneath index, middle, and ring fingers.", "type": "Alphabet"},
    {"char": "N", "name": "Letter N", "desc": "Fist closed with thumb tucked beneath index and middle fingers.", "type": "Alphabet"},
    {"char": "O", "name": "Letter O", "desc": "All five fingers curved together touching tips to form an 'O' circle.", "type": "Alphabet"},
    {"char": "P", "name": "Letter P", "desc": "Like 'K' pointing downward: index forward, middle down, thumb between.", "type": "Alphabet"},
    {"char": "Q", "name": "Letter Q", "desc": "Like 'G' pointing downward: index and thumb pointing downwards parallel.", "type": "Alphabet"},
    {"char": "R", "name": "Letter R", "desc": "Index and middle fingers crossed tightly over each other.", "type": "Alphabet"},
    {"char": "S", "name": "Letter S", "desc": "Tight closed fist with thumb wrapped across the front of the fingers.", "type": "Alphabet"},
    {"char": "T", "name": "Letter T", "desc": "Fist with thumb tucked between index and middle fingers poking through.", "type": "Alphabet"},
    {"char": "U", "name": "Letter U", "desc": "Index and middle fingers extended straight up together, thumb holds others.", "type": "Alphabet"},
    {"char": "V", "name": "Letter V", "desc": "'Peace' / 'Victory' sign: index and middle fingers upright separated.", "type": "Alphabet"},
    {"char": "W", "name": "Letter W", "desc": "Index, middle, and ring fingers extended upright spread apart forming 'W'.", "type": "Alphabet"},
    {"char": "X", "name": "Letter X", "desc": "Index finger bent into a hook/crook, other fingers closed in a fist.", "type": "Alphabet"},
    {"char": "Y", "name": "Letter Y", "desc": "'Hang loose' / Shaka sign: thumb and pinky extended out, middle 3 folded.", "type": "Alphabet"},
    {"char": "Z", "name": "Letter Z", "desc": "Index finger pointing forward, hand moves to trace letter 'Z' in the air.", "type": "Alphabet"},
    {"char": "␣ Space", "name": "Space Gesture", "desc": "Horizontal open palm / custom 'sp.png' gesture to insert word space.", "type": "Special"},
    {"char": ". Period", "name": "Period Gesture", "desc": "Single point / custom '..png' gesture to conclude a sentence.", "type": "Special"}
]


# ── Unchanged functional helpers ─────────────────────────────────────────────

def fileSearch():
    """Searches each file ending with .png in SampleGestures directory"""
    fileEntry = []
    if os.path.exists("SampleGestures"):
        for file in os.listdir("SampleGestures"):
            if file.endswith(".png"):
                fileEntry.append(file)
    return fileEntry


def predictor(mask_img=None):
    """Predicts gesture from mask image or 1.png"""
    try:
        if mask_img is None:
            if not os.path.exists('1.png'):
                return '', 0.0
            test_img = cv2.imread('1.png')
        else:
            test_img = mask_img

        if test_img is None or test_img.size == 0:
            return '', 0.0

        # Check custom gestures first with SIFT/Feature matching
        fileEntry = fileSearch()
        if fileEntry:
            try:
                sift = cv2.SIFT_create()
            except Exception:
                try:
                    sift = cv2.xfeatures2d.SIFT_create()
                except Exception:
                    sift = None

            if sift is not None:
                original = cv2.resize(test_img, (image_x, image_y))
                kp_1, desc_1 = sift.detectAndCompute(original, None)

                if desc_1 is not None and len(desc_1) >= 2:
                    flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict())
                    for f in fileEntry:
                        ref_path = os.path.join("SampleGestures", f)
                        ref_img = cv2.imread(ref_path)
                        if ref_img is None:
                            continue
                        kp_2, desc_2 = sift.detectAndCompute(ref_img, None)
                        if desc_2 is not None and len(desc_2) >= 2:
                            matches = flann.knnMatch(desc_1, desc_2, k=2)
                            good_points = [m for m, n in matches if m.distance < 0.6 * n.distance]
                            if (len(good_points) + len(matches)) > 20:
                                gname = f.replace('.png', '')
                                if gname == 'sp':
                                    return ' ', 0.95
                                if gname == '..':
                                    return '.', 0.95
                                return gname, 0.90

        # Use CNN classifier
        if hasattr(classifier, 'predict_detailed'):
            char, conf, _ = classifier.predict_detailed(test_img)
            return char, conf
        else:
            x_in = cv2.resize(test_img, (image_x, image_y))
            x_in = np.expand_dims(x_in, axis=0)
            res = classifier.predict(x_in)
            for i in range(26):
                if res[0][i] == 1:
                    return chr(ord('A') + i), 0.90
            return '', 0.0
    except Exception as e:
        return '', 0.0


# Backward compatibility stubs
def removeFile():
    for f in ["temp.txt"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass
    if os.path.exists("TempGest"):
        try:
            shutil.rmtree("TempGest")
        except Exception:
            pass

def checkFile():
    if os.path.isfile('temp.txt'):
        try:
            with open("temp.txt", "r", encoding='utf-8') as fr:
                return fr.read()
        except Exception:
            return "No Content Available"
    return "No Content Available"

def openimg():
    pass


# ── Dashboard ────────────────────────────────────────────────────────────────

class Dashboard(QMainWindow):
    def __init__(self):
        super(Dashboard, self).__init__()
        self.setWindowTitle("Sign Language Recognition Studio AI")
        self.resize(1280, 800)
        self.setMinimumSize(1100, 720)

        if os.path.exists('icons/windowLogo.png'):
            self.setWindowIcon(QIcon('icons/windowLogo.png'))

        # Load + apply theme
        self._theme = load_theme_pref()
        self._tokens = DARK_TOKENS if self._theme == "dark" else LIGHT_TOKENS
        self.setStyleSheet(build_qss(self._tokens))

        # State (unchanged)
        self.cam = None
        self.cam_active = True
        self.hsv_slider_val = 65
        self.invert_mask = False
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.sentence_buffer = []
        self.stable_counter = 0
        self.last_predicted = ""
        self.last_mask = None

        # Setup Main UI
        self.init_ui()

        # Camera frame timer (30 FPS) – unchanged
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_frame)
        self.start_camera()

    # ── Theme toggle ─────────────────────────────────────────────────────────

    def _toggle_theme(self):
        """Switch between dark and light themes and persist the choice."""
        self._theme = "light" if self._theme == "dark" else "dark"
        self._tokens = DARK_TOKENS if self._theme == "dark" else LIGHT_TOKENS
        save_theme_pref(self._theme)
        self.setStyleSheet(build_qss(self._tokens))
        icon = "☀" if self._theme == "dark" else "🌙"
        self._btn_theme.setText(icon)
        self._btn_theme.setToolTip(
            "Switch to Light theme" if self._theme == "dark" else "Switch to Dark theme"
        )

    # ── Sidebar helper ───────────────────────────────────────────────────────

    def _make_nav_btn(self, label: str, tooltip: str = "") -> QPushButton:
        """Create a styled sidebar nav button."""
        btn = QPushButton(label)
        btn.setProperty("class", "nav-btn")
        btn.setCheckable(True)
        btn.setToolTip(tooltip)
        btn.setMinimumHeight(40)
        return btn

    # ── init_ui ──────────────────────────────────────────────────────────────

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Sidebar ──────────────────────────────────────────────────────────
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(230)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 16)
        sidebar_layout.setSpacing(4)

        # Logo area
        logo_row = QHBoxLayout()
        logo_row.setContentsMargins(16, 20, 16, 4)
        title = QLabel("Sign Studio")
        title.setObjectName("AppTitle")
        ai_badge = QLabel("  AI  ")
        ai_badge.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 #6366F1, stop:1 #22D3EE);"
            f"color:white; font-size:9px; font-weight:800;"
            f"border-radius:5px; padding:2px 5px; letter-spacing:1px;"
        )
        logo_row.addWidget(title)
        logo_row.addWidget(ai_badge)
        logo_row.addStretch()
        sidebar_layout.addLayout(logo_row)

        subtitle = QLabel("Recognition & Synthesis Studio")
        subtitle.setObjectName("AppSubtitle")
        sidebar_layout.addWidget(subtitle)

        # Thin divider
        div = QFrame()
        div.setObjectName("SidebarDivider")
        div.setFixedHeight(1)
        div.setStyleSheet("background:rgba(255,255,255,0.07); margin:0 16px;")
        sidebar_layout.addWidget(div)
        sidebar_layout.addSpacing(8)

        # Nav buttons
        self.btn_nav_detect = self._make_nav_btn("  Live Recognition", "Live ASL gesture detection (Tab 1)")
        self.btn_nav_detect.setChecked(True)
        self.btn_nav_detect.clicked.connect(lambda: self.switch_tab(0))

        self.btn_nav_catalog = self._make_nav_btn("  Gesture Reference", "Browse all ASL signs A-Z (Tab 2)")
        self.btn_nav_catalog.clicked.connect(lambda: self.switch_tab(1))

        self.btn_nav_custom = self._make_nav_btn("  Custom Gestures", "Create & manage custom signs (Tab 3)")
        self.btn_nav_custom.clicked.connect(lambda: self.switch_tab(2))

        self.btn_nav_export = self._make_nav_btn("  Export & Speech", "Export sentences and play audio (Tab 4)")
        self.btn_nav_export.clicked.connect(lambda: self.switch_tab(3))

        for btn in [self.btn_nav_detect, self.btn_nav_catalog,
                    self.btn_nav_custom, self.btn_nav_export]:
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()

        # Footer divider
        div2 = QFrame()
        div2.setFixedHeight(1)
        div2.setStyleSheet("background:rgba(255,255,255,0.07); margin:0 16px;")
        sidebar_layout.addWidget(div2)
        sidebar_layout.addSpacing(8)

        # Camera status pill
        self.cam_status_lbl = PillBadge("  Camera Active", state="live")
        self.cam_status_lbl.setContentsMargins(16, 0, 16, 0)
        sidebar_layout.addWidget(self.cam_status_lbl)
        sidebar_layout.addSpacing(6)

        # Theme toggle button
        theme_row = QHBoxLayout()
        theme_row.setContentsMargins(10, 0, 10, 0)
        self._btn_theme = QPushButton("☀" if self._theme == "dark" else "🌙")
        self._btn_theme.setProperty("class", "btn-mini")
        self._btn_theme.setFixedWidth(50)
        self._btn_theme.setToolTip("Switch to Light theme" if self._theme == "dark" else "Switch to Dark theme")
        self._btn_theme.clicked.connect(self._toggle_theme)
        theme_lbl = QLabel("Theme")
        theme_lbl.setStyleSheet("color:#64748B; font-size:11px; font-weight:600;")
        theme_row.addWidget(theme_lbl)
        theme_row.addStretch()
        theme_row.addWidget(self._btn_theme)
        sidebar_layout.addLayout(theme_row)
        sidebar_layout.addSpacing(6)

        # Quit button
        btn_exit = QPushButton("  Quit Application")
        btn_exit.setProperty("class", "btn-quit")
        btn_exit.setToolTip("Close the application")
        btn_exit.clicked.connect(self.close)
        sidebar_layout.addWidget(btn_exit)

        main_layout.addWidget(sidebar)

        # ── Main Stacked Content Pages ────────────────────────────────────
        self.stack = QStackedWidget()
        self.page_detect = self.build_detect_page()
        self.page_catalog = self.build_catalog_page()
        self.page_custom = self.build_custom_page()
        self.page_export = self.build_export_page()

        self.stack.addWidget(self.page_detect)
        self.stack.addWidget(self.page_catalog)
        self.stack.addWidget(self.page_custom)
        self.stack.addWidget(self.page_export)

        main_layout.addWidget(self.stack, stretch=1)

    # ── switch_tab (unchanged logic, visual nav update added) ───────────────

    def switch_tab(self, idx):
        self.stack.setCurrentIndex(idx)
        buttons = [self.btn_nav_detect, self.btn_nav_catalog,
                   self.btn_nav_custom, self.btn_nav_export]
        for i, b in enumerate(buttons):
            b.setChecked(i == idx)
        if idx == 3:
            self.refresh_export_page()
        elif idx == 2:
            self.refresh_custom_gallery()

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE BUILDERS
    # ─────────────────────────────────────────────────────────────────────────

    # ── TAB 1: LIVE RECOGNITION ──────────────────────────────────────────────

    def build_detect_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # ── Page header bar ───────────────────────────────────────────────
        header = QHBoxLayout()

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        h_title = QLabel("Live Recognition")
        h_title.setProperty("role", "page_title")
        h_desc = QLabel("Position your hand in the target box — gestures are detected in real time.")
        h_desc.setProperty("role", "body")
        title_col.addWidget(h_title)
        title_col.addWidget(h_desc)
        header.addLayout(title_col)
        header.addStretch()

        # Header right: camera controls
        self.btn_cam_toggle = QPushButton("  Pause Camera")
        self.btn_cam_toggle.setProperty("class", "btn-secondary")
        self.btn_cam_toggle.setToolTip("Pause or resume the live webcam feed  [Esc]")
        self.btn_cam_toggle.clicked.connect(self.toggle_camera)

        btn_view_ref = QPushButton("  ASL Chart")
        btn_view_ref.setProperty("class", "btn-secondary")
        btn_view_ref.setToolTip("Open the Visual ASL reference poster")
        btn_view_ref.clicked.connect(lambda: self.switch_tab(1))

        header.addWidget(self.btn_cam_toggle)
        header.addSpacing(8)
        header.addWidget(btn_view_ref)
        layout.addLayout(header)

        # ── Two-column bento layout ───────────────────────────────────────
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)

        # ── LEFT COLUMN (62%) ─────────────────────────────────────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        # Camera Feed Card
        cam_card = QFrame()
        cam_card.setProperty("class", "card")
        cam_card_layout = QVBoxLayout(cam_card)
        cam_card_layout.setContentsMargins(12, 10, 12, 12)
        cam_card_layout.setSpacing(8)

        feed_header = QHBoxLayout()
        feed_lbl = QLabel("LIVE WEBCAM FEED")
        feed_lbl.setProperty("role", "caption")
        feed_header.addWidget(feed_lbl)
        feed_header.addStretch()
        fps_lbl = QLabel("30 FPS")
        fps_lbl.setStyleSheet("font-size:11px; color:#64748B; font-weight:600;")
        feed_header.addWidget(fps_lbl)
        cam_card_layout.addLayout(feed_header)

        self.lbl_cam_feed = QLabel()
        self.lbl_cam_feed.setFixedSize(460, 320)
        self.lbl_cam_feed.setProperty("role", "cam_feed")
        self.lbl_cam_feed.setAlignment(Qt.AlignCenter)
        self.lbl_cam_feed.setText("Waiting for camera feed…")
        self.lbl_cam_feed.setStyleSheet(
            "background:#000710; border-radius:10px; color:#64748B;"
            "font-size:13px;"
        )
        cam_card_layout.addWidget(self.lbl_cam_feed)
        left_col.addWidget(cam_card)

        # Mask & Tuning Card
        mask_card = QFrame()
        mask_card.setProperty("class", "mask-card")
        mask_layout = QVBoxLayout(mask_card)
        mask_layout.setContentsMargins(14, 12, 14, 12)
        mask_layout.setSpacing(10)

        mask_hdr = QHBoxLayout()
        mask_lbl = QLabel("BINARY MASK & SEGMENTATION")
        mask_lbl.setProperty("role", "caption")
        mask_hdr.addWidget(mask_lbl)
        mask_hdr.addStretch()

        self.btn_invert = QPushButton("  Invert Mask")
        self.btn_invert.setProperty("class", "btn-mini")
        self.btn_invert.setCheckable(True)
        self.btn_invert.setToolTip("Invert the binary hand mask")
        self.btn_invert.clicked.connect(self.toggle_invert_mask)
        mask_hdr.addWidget(self.btn_invert)
        mask_layout.addLayout(mask_hdr)

        # Mode Selector
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Threshold:")
        mode_lbl.setProperty("role", "body")
        mode_lbl.setFixedWidth(80)
        mode_row.addWidget(mode_lbl)

        self.combo_thresh_mode = QComboBox()
        self.combo_thresh_mode.addItems([
            "Smart Skin (YCrCb + HSV) – Auto",
            "Adaptive Otsu Auto-Contrast",
            "Manual Value / HSV Slider",
        ])
        self.combo_thresh_mode.setToolTip("Select the skin segmentation algorithm")
        self.combo_thresh_mode.currentIndexChanged.connect(self.on_thresh_mode_changed)
        mode_row.addWidget(self.combo_thresh_mode, stretch=1)
        mask_layout.addLayout(mode_row)

        # Mask preview + controls side-by-side
        mask_sublayout = QHBoxLayout()
        mask_sublayout.setSpacing(14)

        self.lbl_mask_feed = QLabel()
        self.lbl_mask_feed.setFixedSize(130, 130)
        self.lbl_mask_feed.setProperty("role", "mask_feed")
        self.lbl_mask_feed.setAlignment(Qt.AlignCenter)
        self.lbl_mask_feed.setToolTip("64×64 binary hand mask sent to the classifier")
        mask_sublayout.addWidget(self.lbl_mask_feed)

        ctrl_col = QVBoxLayout()
        ctrl_col.setSpacing(8)

        # Sensitivity slider
        slider_hdr = QHBoxLayout()
        s_lbl = QLabel("Sensitivity:")
        s_lbl.setProperty("role", "body")
        slider_hdr.addWidget(s_lbl)
        slider_hdr.addStretch()
        self.slider_val_lbl = QLabel(f"{self.hsv_slider_val}")
        self.slider_val_lbl.setStyleSheet(
            f"color:{DARK_TOKENS['accent_end']}; font-size:12px; font-weight:700;"
        )
        slider_hdr.addWidget(self.slider_val_lbl)
        ctrl_col.addLayout(slider_hdr)

        self.trackbar = QSlider(Qt.Horizontal)
        self.trackbar.setRange(20, 255)
        self.trackbar.setValue(self.hsv_slider_val)
        self.trackbar.setToolTip("Adjust segmentation sensitivity (20–255)")
        self.trackbar.valueChanged.connect(self.on_slider_changed)
        ctrl_col.addWidget(self.trackbar)

        # Toggle switches (replace plain checkboxes cosmetically)
        filter_row = QHBoxLayout()
        self.chk_denoise = ToggleSwitch("Denoise & Fill")
        self.chk_denoise.setChecked(True)
        self.chk_denoise.setToolTip("Apply morphological denoising to the binary mask")
        filter_row.addWidget(self.chk_denoise)
        filter_row.addSpacing(12)
        self.chk_isolate = ToggleSwitch("Isolate Hand")
        self.chk_isolate.setChecked(True)
        self.chk_isolate.setToolTip("Keep only the largest hand contour")
        filter_row.addWidget(self.chk_isolate)
        filter_row.addStretch()
        ctrl_col.addLayout(filter_row)

        # Lighting presets
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        p_lbl = QLabel("Preset:")
        p_lbl.setProperty("role", "body")
        preset_row.addWidget(p_lbl)

        for label, preset_key, tip in [
            ("Bright", "bright", "High-brightness room preset"),
            ("Normal", "normal", "Standard indoor lighting preset"),
            ("Dim",    "dim",    "Low-light / dim room preset"),
        ]:
            btn = QPushButton(label)
            btn.setProperty("class", "btn-mini")
            btn.setToolTip(tip)
            btn.clicked.connect(lambda checked, k=preset_key: self.apply_preset(k))
            preset_row.addWidget(btn)
        preset_row.addStretch()
        ctrl_col.addLayout(preset_row)

        mask_sublayout.addLayout(ctrl_col)
        mask_layout.addLayout(mask_sublayout)
        left_col.addWidget(mask_card)

        content_layout.addLayout(left_col, stretch=62)

        # ── RIGHT COLUMN (38%) ────────────────────────────────────────────
        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        # Hero Detection Card
        hero_card = QFrame()
        hero_card.setProperty("class", "hero-card")
        hero_layout = QVBoxLayout(hero_card)
        hero_layout.setContentsMargins(20, 18, 20, 18)
        hero_layout.setSpacing(12)

        hero_top = QHBoxLayout()
        recog_title = QLabel("DETECTED GESTURE")
        recog_title.setProperty("role", "caption")
        hero_top.addWidget(recog_title)
        hero_top.addStretch()
        self.status_badge = QLabel("Tracking Hand")
        self.status_badge.setProperty("badge_state", "searching")
        self.status_badge.setStyleSheet(
            "background:rgba(99,102,241,0.15); color:#818CF8;"
            "border:1px solid rgba(99,102,241,0.3);"
            "border-radius:10px; padding:3px 10px;"
            "font-size:11px; font-weight:600;"
        )
        hero_top.addWidget(self.status_badge)
        hero_layout.addLayout(hero_top)

        # Big prediction letter
        self.lbl_hero_char = QLabel("—")
        self.lbl_hero_char.setProperty("role", "hero_char")
        self.lbl_hero_char.setAlignment(Qt.AlignCenter)
        self.lbl_hero_char.setStyleSheet(
            f"font-size:88px; font-weight:900;"
            f"color:{DARK_TOKENS['accent_end']};"
            f"letter-spacing:-2px;"
        )
        hero_layout.addWidget(self.lbl_hero_char)

        # Confidence bar
        conf_row = QHBoxLayout()
        conf_lbl = QLabel("Confidence:")
        conf_lbl.setProperty("role", "body")
        conf_row.addWidget(conf_lbl)
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        self.conf_bar.setFormat("0%")
        self.conf_bar.setFixedHeight(20)
        self.conf_bar.setToolTip("CNN classification confidence for the current gesture")
        conf_row.addWidget(self.conf_bar)
        hero_layout.addLayout(conf_row)

        right_col.addWidget(hero_card)

        # Sentence Builder Card
        sent_card = QFrame()
        sent_card.setProperty("class", "card")
        sent_layout = QVBoxLayout(sent_card)
        sent_layout.setContentsMargins(16, 16, 16, 16)
        sent_layout.setSpacing(12)

        sent_hdr = QHBoxLayout()
        sent_title = QLabel("Constructed Sentence")
        sent_title.setProperty("role", "section_title")
        sent_hdr.addWidget(sent_title)
        sent_hdr.addStretch()
        # Character counter label
        self._char_counter = QLabel("0 chars")
        self._char_counter.setStyleSheet("color:#64748B; font-size:11px;")
        sent_hdr.addWidget(self._char_counter)
        sent_layout.addLayout(sent_hdr)

        self.text_sentence = QTextEdit()
        self.text_sentence.setPlaceholderText(
            "Recognized characters will appear here…\n\n"
            "[C] Append   [Space] Space   [⌫] Backspace   [S] Speak   [Esc] Pause"
        )
        self.text_sentence.setMinimumHeight(130)
        self.text_sentence.textChanged.connect(self._update_char_counter)
        sent_layout.addWidget(self.text_sentence)

        # Row 1: edit actions
        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(8)
        self.btn_append = QPushButton("+ Append")
        self.btn_append.setProperty("class", "btn-success")
        self.btn_append.setToolTip("Append the current predicted character  [C]")
        self.btn_append.clicked.connect(self.append_current_prediction)

        self.btn_space = QPushButton("␣ Space")
        self.btn_space.setProperty("class", "btn-secondary")
        self.btn_space.setToolTip("Insert a space  [Space]")
        self.btn_space.clicked.connect(self.add_space)

        self.btn_backspace = QPushButton("⌫ Back")
        self.btn_backspace.setProperty("class", "btn-secondary")
        self.btn_backspace.setToolTip("Delete last character  [Backspace]")
        self.btn_backspace.clicked.connect(self.backspace)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setProperty("class", "btn-danger")
        self.btn_clear.setToolTip("Clear the entire sentence")
        self.btn_clear.clicked.connect(self.clear_sentence)

        for b in [self.btn_append, self.btn_space, self.btn_backspace, self.btn_clear]:
            btn_row1.addWidget(b)
        sent_layout.addLayout(btn_row1)

        # Row 2: audio / export actions
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(8)
        self.btn_speak = QPushButton("Speak  [S]")
        self.btn_speak.setProperty("class", "btn-primary")
        self.btn_speak.setToolTip("Read the sentence aloud with text-to-speech  [S]")
        self.btn_speak.clicked.connect(self.speak_sentence)

        self.btn_copy = QPushButton("Copy")
        self.btn_copy.setProperty("class", "btn-secondary")
        self.btn_copy.setToolTip("Copy sentence to clipboard")
        self.btn_copy.clicked.connect(self.copy_sentence)

        self.btn_save = QPushButton("Save  [Shift+S]")
        self.btn_save.setProperty("class", "btn-secondary")
        self.btn_save.setToolTip("Save to temp buffer  [Shift+S]")
        self.btn_save.clicked.connect(self.save_buffer_file)

        for b in [self.btn_speak, self.btn_copy, self.btn_save]:
            btn_row2.addWidget(b)
        sent_layout.addLayout(btn_row2)

        right_col.addWidget(sent_card)
        content_layout.addLayout(right_col, stretch=38)
        layout.addLayout(content_layout)

        return page

    def _update_char_counter(self):
        """Update the live character counter next to the sentence builder title."""
        txt = self.text_sentence.toPlainText()
        self._char_counter.setText(f"{len(txt)} chars")

    # ── TAB 2: GESTURE CATALOG ───────────────────────────────────────────────

    def build_catalog_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # Header
        header = QHBoxLayout()
        h_title = QLabel("Gesture Reference")
        h_title.setProperty("role", "page_title")
        header.addWidget(h_title)
        header.addStretch()

        search_box = QLineEdit()
        search_box.setPlaceholderText("  Search gestures…")
        search_box.setFixedWidth(240)
        search_box.setToolTip("Filter the gesture catalog by letter, name, or description")
        header.addWidget(search_box)
        layout.addLayout(header)

        # Tab widget
        cat_tabs = QTabWidget()

        # Sub-tab 1: Grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_container = QWidget()
        self.catalog_grid = QGridLayout(grid_container)
        self.catalog_grid.setSpacing(12)
        self.catalog_grid.setContentsMargins(8, 8, 8, 8)

        cards = []
        col_count = 3
        for i, g in enumerate(GESTURE_CATALOG):
            card = QFrame()
            card.setProperty("class", "card")
            c_layout = QVBoxLayout(card)
            c_layout.setContentsMargins(14, 14, 14, 14)
            c_layout.setSpacing(8)

            top_h = QHBoxLayout()
            badge = QLabel(g["char"])
            badge.setStyleSheet(
                "background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                "stop:0 #6366F1, stop:1 #22D3EE);"
                "color:white; font-size:16px; font-weight:800;"
                "padding:4px 12px; border-radius:8px; letter-spacing:0.5px;"
            )
            top_h.addWidget(badge)
            top_h.addSpacing(8)
            name_lbl = QLabel(g["name"])
            name_lbl.setProperty("role", "section_title")
            name_lbl.setStyleSheet("font-size:13px; font-weight:700; color:#F1F5F9;")
            top_h.addWidget(name_lbl)
            top_h.addStretch()

            type_colors = {
                "Alphabet": ("#818CF8", "rgba(129,140,248,0.12)"),
                "Special":  ("#FBBF24", "rgba(251,191,36,0.12)"),
            }
            fc, bc = type_colors.get(g["type"], ("#94A3B8", "transparent"))
            type_tag = QLabel(g["type"])
            type_tag.setStyleSheet(
                f"color:{fc}; background:{bc}; border-radius:5px;"
                f"padding:2px 7px; font-size:10px; font-weight:700;"
            )
            top_h.addWidget(type_tag)
            c_layout.addLayout(top_h)

            desc = QLabel(g["desc"])
            desc.setProperty("role", "body")
            desc.setWordWrap(True)
            desc.setStyleSheet("font-size:12px; color:#94A3B8; line-height:1.5;")
            c_layout.addWidget(desc)

            row_i = i // col_count
            col_i = i % col_count
            self.catalog_grid.addWidget(card, row_i, col_i)
            cards.append((g, card))

        def filter_cards(txt):
            txt = txt.strip().lower()
            for g, card in cards:
                show = (not txt) or (
                    txt in g["char"].lower() or
                    txt in g["name"].lower() or
                    txt in g["desc"].lower()
                )
                card.setVisible(show)

        search_box.textChanged.connect(filter_cards)
        scroll.setWidget(grid_container)
        cat_tabs.addTab(scroll, "  Alphabet & Signs (A–Z)")

        # Sub-tab 2: Visual Poster
        chart_scroll = QScrollArea()
        chart_scroll.setWidgetResizable(True)
        chart_container = QWidget()
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setAlignment(Qt.AlignCenter)
        chart_layout.setContentsMargins(20, 20, 20, 20)

        chart_lbl = QLabel()
        chart_lbl.setAlignment(Qt.AlignCenter)
        if os.path.exists('template.png'):
            pm = QPixmap('template.png')
            chart_lbl.setPixmap(pm.scaled(500, 740, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            chart_lbl.setText("template.png not found")
        chart_lbl.setStyleSheet("border-radius:12px; border:1px solid rgba(255,255,255,0.07);")
        chart_layout.addWidget(chart_lbl)
        chart_scroll.setWidget(chart_container)
        cat_tabs.addTab(chart_scroll, "  Visual ASL Poster")

        layout.addWidget(cat_tabs)
        return page

    # ── TAB 3: CUSTOM GESTURES STUDIO ────────────────────────────────────────

    def build_custom_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # Header
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        h_title = QLabel("Custom Gesture Studio")
        h_title.setProperty("role", "page_title")
        h_desc = QLabel("Capture your own hand signs and add them to the recognition engine.")
        h_desc.setProperty("role", "body")
        title_col.addWidget(h_title)
        title_col.addWidget(h_desc)
        layout.addLayout(title_col)

        content = QHBoxLayout()
        content.setSpacing(16)

        # Left: Create card
        c_left = QFrame()
        c_left.setProperty("class", "card")
        cl_layout = QVBoxLayout(c_left)
        cl_layout.setContentsMargins(20, 20, 20, 20)
        cl_layout.setSpacing(14)

        create_title = QLabel("Create New Gesture")
        create_title.setProperty("role", "section_title")
        cl_layout.addWidget(create_title)

        # Numbered stepper instructions
        for step_num, step_text in [
            ("1", "Enter a unique name for your gesture below."),
            ("2", "Position your hand clearly inside the camera target box."),
            ("3", "Click 'Capture & Save' when the mask looks correct."),
        ]:
            step_row = QHBoxLayout()
            step_row.setSpacing(10)
            num_lbl = QLabel(step_num)
            num_lbl.setFixedSize(24, 24)
            num_lbl.setAlignment(Qt.AlignCenter)
            num_lbl.setStyleSheet(
                "background:#6366F1; color:white; border-radius:12px;"
                "font-size:11px; font-weight:800;"
            )
            step_row.addWidget(num_lbl)
            txt_lbl = QLabel(step_text)
            txt_lbl.setProperty("role", "body")
            txt_lbl.setWordWrap(True)
            step_row.addWidget(txt_lbl, stretch=1)
            cl_layout.addLayout(step_row)

        self.txt_gest_name = QLineEdit()
        self.txt_gest_name.setPlaceholderText("Gesture name (e.g. OK, ThumbsUp, Hello)…")
        self.txt_gest_name.setToolTip("Enter a unique name for your custom gesture")
        cl_layout.addWidget(self.txt_gest_name)

        btn_capture = QPushButton("  Capture & Save Gesture")
        btn_capture.setProperty("class", "btn-primary")
        btn_capture.setToolTip("Capture the current mask and save it as a custom gesture")
        btn_capture.setMinimumHeight(44)
        btn_capture.clicked.connect(self.capture_custom_gesture)
        cl_layout.addWidget(btn_capture)

        cl_layout.addStretch()
        content.addWidget(c_left, stretch=1)

        # Right: Gallery card
        c_right = QFrame()
        c_right.setProperty("class", "card")
        cr_layout = QVBoxLayout(c_right)
        cr_layout.setContentsMargins(20, 20, 20, 20)
        cr_layout.setSpacing(12)

        gallery_title = QLabel("Saved Gestures")
        gallery_title.setProperty("role", "section_title")
        cr_layout.addWidget(gallery_title)

        self.gallery_scroll = QScrollArea()
        self.gallery_scroll.setWidgetResizable(True)
        self.gallery_widget = QWidget()
        self.gallery_layout = QVBoxLayout(self.gallery_widget)
        self.gallery_layout.setSpacing(8)
        self.gallery_scroll.setWidget(self.gallery_widget)
        cr_layout.addWidget(self.gallery_scroll)

        content.addWidget(c_right, stretch=1)
        layout.addLayout(content)
        return page

    def refresh_custom_gallery(self):
        """Refresh gallery (logic unchanged; only widget styling updated)."""
        while self.gallery_layout.count():
            item = self.gallery_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        files = fileSearch()
        if not files:
            empty_lbl = QLabel("No custom gestures yet.")
            empty_lbl.setStyleSheet(
                "color:#64748B; font-size:14px; font-weight:600;"
                "padding:40px; qproperty-alignment:AlignCenter;"
            )
            empty_lbl.setAlignment(Qt.AlignCenter)
            self.gallery_layout.addWidget(empty_lbl)
            return

        for f in files:
            row = QFrame()
            row.setStyleSheet(
                "background:#0D1424; border-radius:10px; border:1px solid rgba(255,255,255,0.06);"
            )
            r_lay = QHBoxLayout(row)
            r_lay.setContentsMargins(10, 8, 10, 8)
            r_lay.setSpacing(12)

            img_p = os.path.join("SampleGestures", f)
            icon_lbl = QLabel()
            pm = QPixmap(img_p)
            icon_lbl.setPixmap(pm.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            icon_lbl.setFixedSize(48, 48)
            icon_lbl.setStyleSheet("border-radius:6px; border:1px solid rgba(255,255,255,0.08);")
            r_lay.addWidget(icon_lbl)

            name = f.replace('.png', '')
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet("font-size:13px; font-weight:700; color:#F1F5F9;")
            r_lay.addWidget(name_lbl)
            r_lay.addStretch()

            btn_del = QPushButton("Delete")
            btn_del.setProperty("class", "btn-danger")
            btn_del.setFixedSize(70, 32)
            btn_del.setToolTip(f"Delete gesture '{name}'")
            btn_del.clicked.connect(lambda checked, fn=f: self.delete_custom_gesture(fn))
            r_lay.addWidget(btn_del)

            self.gallery_layout.addWidget(row)
        self.gallery_layout.addStretch()

    # ── TAB 4: EXPORT & SPEECH ───────────────────────────────────────────────

    def build_export_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # Header
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        h_title = QLabel("Export & Speech")
        h_title.setProperty("role", "page_title")
        h_desc = QLabel("Review your sentence, play it aloud, or export it to a text file.")
        h_desc.setProperty("role", "body")
        title_col.addWidget(h_title)
        title_col.addWidget(h_desc)
        layout.addLayout(title_col)

        card = QFrame()
        card.setProperty("class", "card")
        c_layout = QVBoxLayout(card)
        c_layout.setContentsMargins(20, 20, 20, 20)
        c_layout.setSpacing(14)

        # Editor header with word/char count
        ed_hdr = QHBoxLayout()
        ed_title = QLabel("Sentence Editor")
        ed_title.setProperty("role", "section_title")
        ed_hdr.addWidget(ed_title)
        ed_hdr.addStretch()
        self._export_counter = QLabel("0 words · 0 chars")
        self._export_counter.setStyleSheet("color:#64748B; font-size:11px;")
        ed_hdr.addWidget(self._export_counter)
        c_layout.addLayout(ed_hdr)

        self.export_editor = QTextEdit()
        self.export_editor.setMinimumHeight(260)
        self.export_editor.setStyleSheet(
            "font-family:'Courier New', 'Consolas', monospace;"
            "font-size:15px; line-height:1.6;"
        )
        self.export_editor.textChanged.connect(self._update_export_counter)
        c_layout.addWidget(self.export_editor)

        # Action bar
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        btn_speak = QPushButton("  Speak Aloud")
        btn_speak.setProperty("class", "btn-primary")
        btn_speak.setToolTip("Read the text aloud using text-to-speech")
        btn_speak.setMinimumHeight(40)
        btn_speak.clicked.connect(self.speak_export_text)

        btn_save_file = QPushButton("  Save to .txt")
        btn_save_file.setProperty("class", "btn-success")
        btn_save_file.setToolTip("Export the text to a .txt file on disk")
        btn_save_file.setMinimumHeight(40)
        btn_save_file.clicked.connect(self.save_export_to_file)

        btn_clear = QPushButton("  Clear")
        btn_clear.setProperty("class", "btn-secondary")
        btn_clear.setToolTip("Clear the editor")
        btn_clear.setMinimumHeight(40)
        btn_clear.clicked.connect(self.export_editor.clear)

        btn_row.addWidget(btn_speak)
        btn_row.addWidget(btn_save_file)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch()
        c_layout.addLayout(btn_row)

        layout.addWidget(card)
        return page

    def _update_export_counter(self):
        """Update the live word/character counter on the export page."""
        txt = self.export_editor.toPlainText()
        words = len(txt.split()) if txt.strip() else 0
        self._export_counter.setText(f"{words} words · {len(txt)} chars")

    # ─────────────────────────────────────────────────────────────────────────
    # ALL LOGIC METHODS BELOW ARE UNCHANGED FROM THE ORIGINAL
    # ─────────────────────────────────────────────────────────────────────────

    def capture_custom_gesture(self):
        name = self.txt_gest_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing Name", "Please enter a name for the gesture before capturing.")
            return

        if not hasattr(self, 'last_mask') or self.last_mask is None:
            QMessageBox.warning(self, "Camera Error", "No active camera frame to capture.")
            return

        if not os.path.exists('./SampleGestures'):
            os.makedirs('./SampleGestures')

        img_path = os.path.join("SampleGestures", f"{name}.png")
        save_img = cv2.resize(self.last_mask, (image_x, image_y))
        cv2.imwrite(img_path, save_img)
        QMessageBox.information(self, "Success", f"Custom gesture '{name}' successfully saved!")
        self.txt_gest_name.clear()
        self.refresh_custom_gallery()

    def delete_custom_gesture(self, fname):
        path = os.path.join("SampleGestures", fname)
        if os.path.exists(path):
            reply = QMessageBox.question(self, "Confirm Delete",
                                         f"Are you sure you want to delete '{fname}'?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                os.remove(path)
                self.refresh_custom_gallery()

    def refresh_export_page(self):
        content = self.text_sentence.toPlainText().strip()
        if not content:
            content = checkFile()
            if content == "No Content Available":
                content = ""
        self.export_editor.setText(content)

    def speak_export_text(self):
        txt = self.export_editor.toPlainText().strip()
        if txt and engine:
            try:
                engine.say(txt.lower())
                engine.runAndWait()
            except Exception:
                pass

    def save_export_to_file(self):
        txt = self.export_editor.toPlainText()
        if not txt.strip():
            QMessageBox.information(self, "Notice", "Nothing to export.")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Export File", "",
                                                   "Text Files (*.txt);;All Files (*)")
        if filename:
            if not filename.endswith('.txt'):
                filename += '.txt'
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(txt)
            QMessageBox.information(self, "Saved", f"File saved successfully to:\n{filename}")

    # ── Camera Engine & Prediction Loop (UNCHANGED) ──────────────────────────

    def start_camera(self):
        try:
            if self.cam is None or not self.cam.isOpened():
                self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not self.cam.isOpened():
                    self.cam = cv2.VideoCapture(0)
            if not self.cam.isOpened():
                self.cam_status_lbl.setText("  Camera Offline")
                self.cam_status_lbl.setProperty("cam_state", "error")
                self.cam_status_lbl.setStyleSheet(
                    "color:#F87171; background:rgba(248,113,113,0.12);"
                    "border:1px solid rgba(248,113,113,0.3);"
                    "border-radius:10px; padding:3px 10px;"
                    "font-size:11px; font-weight:600;"
                )
                self.lbl_cam_feed.setText("Unable to open camera.\nPlease connect a webcam.")
            else:
                self.cam_status_lbl.setText("  Camera Active")
                self.cam_status_lbl.setProperty("cam_state", "live")
                self.cam_status_lbl.setStyleSheet(
                    "color:#34D399; background:rgba(52,211,153,0.12);"
                    "border:1px solid rgba(52,211,153,0.3);"
                    "border-radius:10px; padding:3px 10px;"
                    "font-size:11px; font-weight:600;"
                )
                self.timer.start(33)
        except Exception as e:
            self.lbl_cam_feed.setText(f"Camera initialization error: {e}")

    def toggle_camera(self):
        self.cam_active = not self.cam_active
        if self.cam_active:
            self.btn_cam_toggle.setText("  Pause Camera")
            self.cam_status_lbl.setText("  Camera Active")
            self.cam_status_lbl.setStyleSheet(
                "color:#34D399; background:rgba(52,211,153,0.12);"
                "border:1px solid rgba(52,211,153,0.3);"
                "border-radius:10px; padding:3px 10px;"
                "font-size:11px; font-weight:600;"
            )
            self.timer.start(33)
        else:
            self.btn_cam_toggle.setText("  Resume Camera")
            self.cam_status_lbl.setText("  Camera Paused")
            self.cam_status_lbl.setStyleSheet(
                "color:#FBBF24; background:rgba(251,191,36,0.12);"
                "border:1px solid rgba(251,191,36,0.3);"
                "border-radius:10px; padding:3px 10px;"
                "font-size:11px; font-weight:600;"
            )
            self.timer.stop()

    def on_slider_changed(self, val):
        self.hsv_slider_val = val
        self.slider_val_lbl.setText(f"{val}")

    def toggle_invert_mask(self):
        self.invert_mask = self.btn_invert.isChecked()

    def on_thresh_mode_changed(self, idx):
        # Slider only relevant in manual mode
        self.trackbar.setEnabled(idx == 2)

    def apply_preset(self, preset_type):
        if preset_type == "bright":
            self.trackbar.setValue(100)
            self.combo_thresh_mode.setCurrentIndex(0)
        elif preset_type == "normal":
            self.trackbar.setValue(65)
            self.combo_thresh_mode.setCurrentIndex(0)
        elif preset_type == "dim":
            self.trackbar.setValue(35)
            self.combo_thresh_mode.setCurrentIndex(0)

    def process_binary_threshold(self, imcrop):
        if imcrop is None or imcrop.size == 0:
            return np.zeros((image_y, image_x), dtype=np.uint8)

        mode_text = self.combo_thresh_mode.currentText()
        sens = self.trackbar.value()

        blurred = cv2.GaussianBlur(imcrop, (5, 5), 0)

        if "Smart Skin" in mode_text:
            ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
            delta = int((sens - 65) * 0.25)
            cr_min = max(115, 133 - delta)
            cr_max = min(190, 175 + delta)
            cb_min = max(65, 77 - delta)
            cb_max = min(140, 127 + delta)
            mask_ycrcb = cv2.inRange(ycrcb,
                                     np.array([0, cr_min, cb_min]),
                                     np.array([255, cr_max, cb_max]))
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask_hsv1 = cv2.inRange(hsv, np.array([0, 20, 30]), np.array([25, 255, 255]))
            mask_hsv2 = cv2.inRange(hsv, np.array([165, 20, 30]), np.array([180, 255, 255]))
            mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
            mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

        elif "Otsu" in mode_text:
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            norm_gray = clahe.apply(gray)
            _, mask = cv2.threshold(norm_gray, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            lower_bound = np.array([0, 0, 0])
            upper_bound = np.array([179, 255, sens])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)

        if self.invert_mask:
            mask = cv2.bitwise_not(mask)

        if hasattr(self, 'chk_denoise') and self.chk_denoise.isChecked():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.dilate(mask, kernel, iterations=1)

        if hasattr(self, 'chk_isolate') and self.chk_isolate.isChecked():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(max_c) > 500:
                    clean_mask = np.zeros_like(mask)
                    cv2.drawContours(clean_mask, [max_c], -1, 255, -1)
                    mask = clean_mask
                else:
                    mask = np.zeros_like(mask)

        return mask

    def update_camera_frame(self):
        if not self.cam or not self.cam.isOpened():
            return

        ret, frame = self.cam.read()
        if not ret or frame is None:
            return

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        x1, y1 = int(w * 0.45), int(h * 0.15)
        x2, y2 = int(w * 0.85), int(h * 0.65)

        imcrop = frame[y1:y2, x1:x2]
        if imcrop.size == 0:
            return

        mask = self.process_binary_threshold(imcrop)
        self.last_mask = mask

        save_img = cv2.resize(mask, (image_x, image_y))
        cv2.imwrite("1.png", save_img)

        pred_char, conf = predictor(mask)
        self.current_prediction = pred_char
        self.current_confidence = conf

        # Draw HUD bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (56, 189, 248), 2)
        corner_len = 16
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (16, 185, 129), 4)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (16, 185, 129), 4)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (16, 185, 129), 4)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (16, 185, 129), 4)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (16, 185, 129), 4)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), (16, 185, 129), 4)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (16, 185, 129), 4)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (16, 185, 129), 4)
        cv2.putText(frame, "HAND TARGET", (x1 + 6, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1, cv2.LINE_AA)

        # Render to Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fh, fw, fc = rgb_frame.shape
        qimg = QImage(rgb_frame.data, fw, fh, fc * fw, QImage.Format_RGB888)
        self.lbl_cam_feed.setPixmap(
            QPixmap.fromImage(qimg).scaled(460, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        mh, mw = mask.shape
        qmask = QImage(mask.data, mw, mh, mw, QImage.Format_Grayscale8)
        self.lbl_mask_feed.setPixmap(
            QPixmap.fromImage(qmask).scaled(130, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        # Update hero card
        if pred_char:
            self.lbl_hero_char.setText(pred_char)
            conf_pct = int(conf * 100)
            self.conf_bar.setValue(conf_pct)
            self.conf_bar.setFormat(f"{conf_pct}%")

            # Confidence colour via property (drives QSS [conf_level] selector)
            level = "high" if conf_pct >= 80 else ("mid" if conf_pct >= 50 else "low")
            self.conf_bar.setProperty("conf_level", level)
            self.conf_bar.style().unpolish(self.conf_bar)
            self.conf_bar.style().polish(self.conf_bar)

            self.status_badge.setText(f"  Recognized: {pred_char}")
            self.status_badge.setStyleSheet(
                "background:rgba(52,211,153,0.15); color:#34D399;"
                "border:1px solid rgba(52,211,153,0.4);"
                "border-radius:10px; padding:3px 10px;"
                "font-size:11px; font-weight:600;"
            )
        else:
            self.lbl_hero_char.setText("—")
            self.conf_bar.setValue(0)
            self.conf_bar.setFormat("Searching…")
            self.conf_bar.setProperty("conf_level", "")
            self.conf_bar.style().unpolish(self.conf_bar)
            self.conf_bar.style().polish(self.conf_bar)
            self.status_badge.setText("  Tracking Hand")
            self.status_badge.setStyleSheet(
                "background:rgba(99,102,241,0.15); color:#818CF8;"
                "border:1px solid rgba(99,102,241,0.3);"
                "border-radius:10px; padding:3px 10px;"
                "font-size:11px; font-weight:600;"
            )

    # ── Keyboard Shortcuts (UNCHANGED) ───────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_C:
            self.append_current_prediction()
        elif key == Qt.Key_Space:
            self.add_space()
        elif key == Qt.Key_Backspace:
            self.backspace()
        elif key == Qt.Key_S and (event.modifiers() & Qt.ShiftModifier):
            self.save_buffer_file()
        elif key == Qt.Key_S:
            self.speak_sentence()
        elif key == Qt.Key_Escape:
            self.toggle_camera()
        else:
            super(Dashboard, self).keyPressEvent(event)

    # ── Sentence actions (UNCHANGED) ─────────────────────────────────────────

    def append_current_prediction(self):
        if self.current_prediction:
            current_txt = self.text_sentence.toPlainText()
            self.text_sentence.setText(current_txt + self.current_prediction)
            self.sentence_buffer.append(self.current_prediction)

    def add_space(self):
        current_txt = self.text_sentence.toPlainText()
        self.text_sentence.setText(current_txt + " ")
        self.sentence_buffer.append(" ")

    def backspace(self):
        current_txt = self.text_sentence.toPlainText()
        if current_txt:
            self.text_sentence.setText(current_txt[:-1])
            if self.sentence_buffer:
                self.sentence_buffer.pop()

    def clear_sentence(self):
        self.text_sentence.clear()
        self.sentence_buffer = []

    def speak_sentence(self):
        txt = self.text_sentence.toPlainText().strip()
        if txt and engine:
            try:
                engine.say(txt.lower())
                engine.runAndWait()
            except Exception:
                pass

    def copy_sentence(self):
        txt = self.text_sentence.toPlainText()
        if txt:
            QApplication.clipboard().setText(txt)
            Toast.show_in(self, "Copied to clipboard!")

    def save_buffer_file(self):
        txt = self.text_sentence.toPlainText()
        if txt.strip():
            with open("temp.txt", "w", encoding='utf-8') as f:
                f.write(txt)
            Toast.show_in(self, "Sentence saved to buffer.")

    # ── Close (UNCHANGED) ────────────────────────────────────────────────────

    def closeEvent(self, event):
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        if hasattr(self, 'cam') and self.cam is not None:
            self.cam.release()
        cv2.destroyAllWindows()
        event.accept()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # HiDPI flags (set before QApplication if not already done)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    win = Dashboard()
    win.show()
    sys.exit(app.exec_())
