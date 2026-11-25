"""
Sign Language Recognition Studio - Upgraded Modern UI
Includes Real-time ASL detection, Sentence Builder, Gesture Catalog, Custom Gesture Studio, and Audio Speech
"""
__author__ = 'Mohsin Raza, Muhammad Awais Sarwar, Upgraded Studio Edition'

import os
import sys
import shutil
import tempfile
import cv2
import numpy as np

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
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QColor, QPainter, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSlider, QTextEdit, QLineEdit,
    QStackedWidget, QFrame, QScrollArea, QSplitter, QProgressBar,
    QMessageBox, QFileDialog, QSizePolicy, QTabWidget, QGraphicsDropShadowEffect,
    QComboBox, QCheckBox
)

import pyttsx3

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


# Modern Stylesheet (QSS)
MODERN_STYLE = """
QMainWindow {
    background-color: #0b0f19;
    color: #f1f5f9;
}
QWidget {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    color: #e2e8f0;
}
/* Sidebar */
#Sidebar {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
    min-width: 230px;
    max-width: 230px;
}
#AppTitle {
    color: #38bdf8;
    font-size: 19px;
    font-weight: 700;
    padding: 18px 16px 4px 16px;
}
#AppSubtitle {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 500;
    padding: 0px 16px 18px 16px;
}
/* Nav Buttons */
QPushButton.nav-btn {
    background-color: transparent;
    color: #94a3b8;
    font-size: 14px;
    font-weight: 600;
    text-align: left;
    padding: 12px 18px;
    border: none;
    border-radius: 8px;
    margin: 4px 10px;
}
QPushButton.nav-btn:hover {
    background-color: #1e293b;
    color: #f8fafc;
}
QPushButton.nav-btn:checked, QPushButton.nav-btn.active {
    background-color: #2563eb;
    color: #ffffff;
}

/* Card panels */
QFrame.card {
    background-color: #131d31;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 14px;
}
QFrame.hero-card {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1e1b4b, stop:1 #0f172a);
    border: 1px solid #312e81;
    border-radius: 12px;
    padding: 16px;
}

/* Action Buttons */
QPushButton.btn-primary {
    background-color: #2563eb;
    color: white;
    font-size: 13px;
    font-weight: 600;
    border-radius: 8px;
    padding: 9px 18px;
    border: none;
}
QPushButton.btn-primary:hover {
    background-color: #1d4ed8;
}
QPushButton.btn-success {
    background-color: #059669;
    color: white;
    font-size: 13px;
    font-weight: 600;
    border-radius: 8px;
    padding: 9px 18px;
    border: none;
}
QPushButton.btn-success:hover {
    background-color: #047857;
}
QPushButton.btn-danger {
    background-color: #dc2626;
    color: white;
    font-size: 13px;
    font-weight: 600;
    border-radius: 8px;
    padding: 9px 18px;
    border: none;
}
QPushButton.btn-danger:hover {
    background-color: #b91c1c;
}
QPushButton.btn-secondary {
    background-color: #1e293b;
    color: #cbd5e1;
    font-size: 13px;
    font-weight: 600;
    border-radius: 8px;
    padding: 9px 16px;
    border: 1px solid #334155;
}
QPushButton.btn-secondary:hover {
    background-color: #334155;
    color: white;
}

/* Text Editors */
QTextEdit, QLineEdit {
    background-color: #0b1120;
    border: 1px solid #1e293b;
    border-radius: 8px;
    color: #f8fafc;
    font-size: 14px;
    padding: 10px;
}
QTextEdit:focus, QLineEdit:focus {
    border: 1px solid #38bdf8;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #38bdf8;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #38bdf8;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}

/* Progress bar */
QProgressBar {
    border: 1px solid #1e293b;
    border-radius: 6px;
    text-align: center;
    background-color: #0b1120;
    color: #f8fafc;
    font-weight: 600;
    font-size: 11px;
}
QProgressBar::chunk {
    background-color: #10b981;
    border-radius: 5px;
}

/* Scroll area */
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollBar:vertical {
    border: none;
    background: #0f172a;
    width: 8px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #334155;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
/* Combo Box */
QComboBox {
    background-color: #0b1120;
    border: 1px solid #1e293b;
    border-radius: 8px;
    color: #f8fafc;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 10px;
}
QComboBox:hover {
    border: 1px solid #38bdf8;
}
QComboBox QAbstractItemView {
    background-color: #0f172a;
    color: #f8fafc;
    selection-background-color: #2563eb;
    border: 1px solid #1e293b;
    border-radius: 6px;
}
/* Checkboxes */
QCheckBox {
    color: #cbd5e1;
    font-size: 11px;
    font-weight: 500;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border-radius: 4px;
    border: 1px solid #334155;
    background: #0b1120;
}
QCheckBox::indicator:checked {
    background: #2563eb;
    border: 1px solid #38bdf8;
}
/* Mini Buttons */
QPushButton.btn-mini {
    background-color: #1e293b;
    color: #cbd5e1;
    font-size: 11px;
    font-weight: 600;
    border-radius: 6px;
    padding: 4px 8px;
    border: 1px solid #334155;
}
QPushButton.btn-mini:hover {
    background-color: #334155;
    color: #ffffff;
}
QPushButton.btn-mini.active {
    background-color: #2563eb;
    color: #ffffff;
    border: 1px solid #38bdf8;
}
"""


class Dashboard(QMainWindow):
    def __init__(self):
        super(Dashboard, self).__init__()
        self.setWindowTitle("Sign Language Recognition Studio AI")
        self.resize(1180, 800)
        self.setMinimumSize(960, 680)
        
        if os.path.exists('icons/windowLogo.png'):
            self.setWindowIcon(QIcon('icons/windowLogo.png'))

        self.setStyleSheet(MODERN_STYLE)

        # State
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

        # Camera frame timer (30 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_frame)
        self.start_camera()

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Left Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 16)
        sidebar_layout.setSpacing(6)

        title = QLabel("Sign Language AI")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Recognition & Synthesis Studio")
        subtitle.setObjectName("AppSubtitle")
        sidebar_layout.addWidget(title)
        sidebar_layout.addWidget(subtitle)

        # Navigation Buttons
        self.btn_nav_detect = QPushButton("🎥  Live Recognition")
        self.btn_nav_detect.setProperty("class", "nav-btn")
        self.btn_nav_detect.setCheckable(True)
        self.btn_nav_detect.setChecked(True)
        self.btn_nav_detect.clicked.connect(lambda: self.switch_tab(0))

        self.btn_nav_catalog = QPushButton("📖  Gesture Reference")
        self.btn_nav_catalog.setProperty("class", "nav-btn")
        self.btn_nav_catalog.setCheckable(True)
        self.btn_nav_catalog.clicked.connect(lambda: self.switch_tab(1))

        self.btn_nav_custom = QPushButton("➕  Custom Gestures")
        self.btn_nav_custom.setProperty("class", "nav-btn")
        self.btn_nav_custom.setCheckable(True)
        self.btn_nav_custom.clicked.connect(lambda: self.switch_tab(2))

        self.btn_nav_export = QPushButton("📄  Export & Speech")
        self.btn_nav_export.setProperty("class", "nav-btn")
        self.btn_nav_export.setCheckable(True)
        self.btn_nav_export.clicked.connect(lambda: self.switch_tab(3))

        sidebar_layout.addWidget(self.btn_nav_detect)
        sidebar_layout.addWidget(self.btn_nav_catalog)
        sidebar_layout.addWidget(self.btn_nav_custom)
        sidebar_layout.addWidget(self.btn_nav_export)
        sidebar_layout.addStretch()

        # Camera Status Footer
        self.cam_status_lbl = QLabel("🟢 Camera Active")
        self.cam_status_lbl.setStyleSheet("color: #10b981; font-size: 12px; font-weight: 600; padding: 0 16px;")
        sidebar_layout.addWidget(self.cam_status_lbl)

        btn_exit = QPushButton("⏻  Quit Application")
        btn_exit.setProperty("class", "nav-btn")
        btn_exit.setStyleSheet("color: #f87171;")
        btn_exit.clicked.connect(self.close)
        sidebar_layout.addWidget(btn_exit)

        main_layout.addWidget(sidebar)

        # 2. Main Stacked Content Pages
        self.stack = QStackedWidget()
        self.page_detect = self.build_detect_page()
        self.page_catalog = self.build_catalog_page()
        self.page_custom = self.build_custom_page()
        self.page_export = self.build_export_page()

        self.stack.addWidget(self.page_detect)
        self.stack.addWidget(self.page_catalog)
        self.stack.addWidget(self.page_custom)
        self.stack.addWidget(self.page_export)

        main_layout.addWidget(self.stack)

    def switch_tab(self, idx):
        self.stack.setCurrentIndex(idx)
        buttons = [self.btn_nav_detect, self.btn_nav_catalog, self.btn_nav_custom, self.btn_nav_export]
        for i, b in enumerate(buttons):
            b.setChecked(i == idx)
        if idx == 3:
            self.refresh_export_page()
        elif idx == 2:
            self.refresh_custom_gallery()

    # ---------------- TAB 1: LIVE RECOGNITION ----------------
    def build_detect_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Header bar
        header = QHBoxLayout()
        h_title = QLabel("Real-time Gesture Recognition")
        h_title.setStyleSheet("font-size: 22px; font-weight: 700; color: #f8fafc;")
        header.addWidget(h_title)
        header.addStretch()

        self.btn_cam_toggle = QPushButton("Pause Camera")
        self.btn_cam_toggle.setProperty("class", "btn-secondary")
        self.btn_cam_toggle.clicked.connect(self.toggle_camera)
        header.addWidget(self.btn_cam_toggle)

        btn_view_ref = QPushButton("Visual ASL Chart")
        btn_view_ref.setProperty("class", "btn-secondary")
        btn_view_ref.clicked.connect(lambda: self.switch_tab(1))
        header.addWidget(btn_view_ref)

        layout.addLayout(header)

        # Central grid: Left (Video Feeds) & Right (Hero & Sentence Builder)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)

        # Left Column: Video Feeds
        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        # Camera Feed Card
        cam_card = QFrame()
        cam_card.setProperty("class", "card")
        cam_card_layout = QVBoxLayout(cam_card)
        cam_card_layout.setContentsMargins(10, 10, 10, 10)
        cam_card_layout.setSpacing(6)

        feed_header = QHBoxLayout()
        feed_lbl = QLabel("Live Webcam Feed (ROI Target Box)")
        feed_lbl.setStyleSheet("font-size: 13px; font-weight: 600; color: #94a3b8;")
        feed_header.addWidget(feed_lbl)
        feed_header.addStretch()
        fps_lbl = QLabel("30 FPS")
        fps_lbl.setStyleSheet("font-size: 11px; color: #64748b;")
        feed_header.addWidget(fps_lbl)
        cam_card_layout.addLayout(feed_header)

        self.lbl_cam_feed = QLabel()
        self.lbl_cam_feed.setFixedSize(440, 310)
        self.lbl_cam_feed.setStyleSheet("background-color: #020617; border-radius: 8px;")
        self.lbl_cam_feed.setAlignment(Qt.AlignCenter)
        self.lbl_cam_feed.setText("Waiting for camera feed...")
        cam_card_layout.addWidget(self.lbl_cam_feed)
        left_col.addWidget(cam_card)

        # Mask Feed & Tuning Card
        mask_card = QFrame()
        mask_card.setProperty("class", "card")
        mask_layout = QVBoxLayout(mask_card)
        mask_layout.setContentsMargins(12, 10, 12, 10)
        mask_layout.setSpacing(8)

        mask_hdr = QHBoxLayout()
        mask_lbl = QLabel("Binary Screen Threshold & Segmentation")
        mask_lbl.setStyleSheet("font-size: 13px; font-weight: 700; color: #f8fafc;")
        mask_hdr.addWidget(mask_lbl)
        mask_hdr.addStretch()

        self.btn_invert = QPushButton("⇄ Invert Mask")
        self.btn_invert.setProperty("class", "btn-mini")
        self.btn_invert.setCheckable(True)
        self.btn_invert.clicked.connect(self.toggle_invert_mask)
        mask_hdr.addWidget(self.btn_invert)
        mask_layout.addLayout(mask_hdr)

        # Mode Selector
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Threshold Mode:")
        mode_lbl.setStyleSheet("font-size: 12px; color: #94a3b8; font-weight: 600;")
        mode_row.addWidget(mode_lbl)

        self.combo_thresh_mode = QComboBox()
        self.combo_thresh_mode.addItems([
            "🤖 Smart Skin (YCrCb + HSV) - Auto",
            "⚡ Adaptive Otsu Auto-Contrast",
            "🎛 Manual Value / HSV Slider"
        ])
        self.combo_thresh_mode.currentIndexChanged.connect(self.on_thresh_mode_changed)
        mode_row.addWidget(self.combo_thresh_mode, stretch=1)
        mask_layout.addLayout(mode_row)

        mask_sublayout = QHBoxLayout()
        mask_sublayout.setSpacing(10)

        # Live Mask Display
        self.lbl_mask_feed = QLabel()
        self.lbl_mask_feed.setFixedSize(145, 145)
        self.lbl_mask_feed.setStyleSheet("background-color: #020617; border-radius: 8px; border: 1px solid #334155;")
        self.lbl_mask_feed.setAlignment(Qt.AlignCenter)
        mask_sublayout.addWidget(self.lbl_mask_feed)

        # Controls Column
        ctrl_col = QVBoxLayout()
        ctrl_col.setSpacing(5)

        # Sensitivity Slider
        slider_hdr = QHBoxLayout()
        s_lbl = QLabel("Sensitivity Tuning:")
        s_lbl.setStyleSheet("font-size: 11px; color: #cbd5e1; font-weight: 600;")
        slider_hdr.addWidget(s_lbl)
        slider_hdr.addStretch()
        self.slider_val_lbl = QLabel(f"{self.hsv_slider_val}")
        self.slider_val_lbl.setStyleSheet("font-size: 11px; color: #38bdf8; font-weight: 700;")
        slider_hdr.addWidget(self.slider_val_lbl)
        ctrl_col.addLayout(slider_hdr)

        self.trackbar = QSlider(Qt.Horizontal)
        self.trackbar.setRange(20, 255)
        self.trackbar.setValue(self.hsv_slider_val)
        self.trackbar.valueChanged.connect(self.on_slider_changed)
        ctrl_col.addWidget(self.trackbar)

        # Filters: Denoise & Isolate Hand Contour
        filter_row = QHBoxLayout()
        self.chk_denoise = QCheckBox("Denoise & Fill Voids")
        self.chk_denoise.setChecked(True)
        filter_row.addWidget(self.chk_denoise)

        self.chk_isolate = QCheckBox("Isolate Hand")
        self.chk_isolate.setChecked(True)
        filter_row.addWidget(self.chk_isolate)
        ctrl_col.addLayout(filter_row)

        # Lighting Presets
        preset_row = QHBoxLayout()
        p_lbl = QLabel("Presets:")
        p_lbl.setStyleSheet("font-size: 11px; color: #64748b;")
        preset_row.addWidget(p_lbl)

        btn_preset_bright = QPushButton("☀️ Bright")
        btn_preset_bright.setProperty("class", "btn-mini")
        btn_preset_bright.clicked.connect(lambda: self.apply_preset("bright"))
        preset_row.addWidget(btn_preset_bright)

        btn_preset_normal = QPushButton("💡 Normal")
        btn_preset_normal.setProperty("class", "btn-mini")
        btn_preset_normal.clicked.connect(lambda: self.apply_preset("normal"))
        preset_row.addWidget(btn_preset_normal)

        btn_preset_dim = QPushButton("🌙 Dim")
        btn_preset_dim.setProperty("class", "btn-mini")
        btn_preset_dim.clicked.connect(lambda: self.apply_preset("dim"))
        preset_row.addWidget(btn_preset_dim)
        ctrl_col.addLayout(preset_row)

        mask_sublayout.addLayout(ctrl_col)
        mask_layout.addLayout(mask_sublayout)
        left_col.addWidget(mask_card)

        content_layout.addLayout(left_col, stretch=1)

        # Right Column: Prediction & Sentence Builder
        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        # Hero Detection Card
        hero_card = QFrame()
        hero_card.setProperty("class", "hero-card")
        hero_layout = QVBoxLayout(hero_card)
        hero_layout.setContentsMargins(16, 16, 16, 16)
        hero_layout.setSpacing(10)

        hero_top = QHBoxLayout()
        recog_title = QLabel("CURRENT DETECTED GESTURE")
        recog_title.setStyleSheet("font-size: 11px; font-weight: 700; color: #a5b4fc; letter-spacing: 1px;")
        hero_top.addWidget(recog_title)
        hero_top.addStretch()
        self.status_badge = QLabel("Tracking Hand")
        self.status_badge.setStyleSheet("background-color: #1e3a8a; color: #93c5fd; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;")
        hero_top.addWidget(self.status_badge)
        hero_layout.addLayout(hero_top)

        # Big Hero Character Display
        self.lbl_hero_char = QLabel("—")
        self.lbl_hero_char.setStyleSheet("font-size: 72px; font-weight: 900; color: #38bdf8; text-align: center;")
        self.lbl_hero_char.setAlignment(Qt.AlignCenter)
        hero_layout.addWidget(self.lbl_hero_char)

        # Confidence Bar
        conf_row = QHBoxLayout()
        conf_lbl = QLabel("Confidence:")
        conf_lbl.setStyleSheet("font-size: 12px; color: #94a3b8; font-weight: 500;")
        conf_row.addWidget(conf_lbl)
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        conf_row.addWidget(self.conf_bar)
        hero_layout.addLayout(conf_row)

        right_col.addWidget(hero_card)

        # Sentence Builder Card
        sent_card = QFrame()
        sent_card.setProperty("class", "card")
        sent_layout = QVBoxLayout(sent_card)
        sent_layout.setContentsMargins(16, 16, 16, 16)
        sent_layout.setSpacing(12)

        sent_title = QLabel("Constructed Sentence")
        sent_title.setStyleSheet("font-size: 14px; font-weight: 700; color: #f8fafc;")
        sent_layout.addWidget(sent_title)

        self.text_sentence = QTextEdit()
        self.text_sentence.setPlaceholderText("Recognized characters and formed words will appear here...\n\nPress 'C' or click '+ Append' to add character\nPress 'Space' to add space\nPress 'Backspace' to delete last character")
        self.text_sentence.setMinimumHeight(140)
        sent_layout.addWidget(self.text_sentence)

        # Action Buttons
        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(8)
        self.btn_append = QPushButton("+ Append (C)")
        self.btn_append.setProperty("class", "btn-success")
        self.btn_append.clicked.connect(self.append_current_prediction)
        btn_row1.addWidget(self.btn_append)

        self.btn_space = QPushButton("␣ Space")
        self.btn_space.setProperty("class", "btn-secondary")
        self.btn_space.clicked.connect(self.add_space)
        btn_row1.addWidget(self.btn_space)

        self.btn_backspace = QPushButton("⌫ Backspace")
        self.btn_backspace.setProperty("class", "btn-secondary")
        self.btn_backspace.clicked.connect(self.backspace)
        btn_row1.addWidget(self.btn_backspace)

        self.btn_clear = QPushButton("🗑 Clear")
        self.btn_clear.setProperty("class", "btn-danger")
        self.btn_clear.clicked.connect(self.clear_sentence)
        btn_row1.addWidget(self.btn_clear)
        sent_layout.addLayout(btn_row1)

        # Audio & Export Buttons
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(8)

        self.btn_speak = QPushButton("🔊 Speak (S)")
        self.btn_speak.setProperty("class", "btn-primary")
        self.btn_speak.clicked.connect(self.speak_sentence)
        btn_row2.addWidget(self.btn_speak)

        self.btn_copy = QPushButton("📋 Copy Text")
        self.btn_copy.setProperty("class", "btn-secondary")
        self.btn_copy.clicked.connect(self.copy_sentence)
        btn_row2.addWidget(self.btn_copy)

        self.btn_save = QPushButton("💾 Save (Shift+S)")
        self.btn_save.setProperty("class", "btn-secondary")
        self.btn_save.clicked.connect(self.save_buffer_file)
        btn_row2.addWidget(self.btn_save)
        sent_layout.addLayout(btn_row2)

        right_col.addWidget(sent_card)
        content_layout.addLayout(right_col, stretch=1)
        layout.addLayout(content_layout)

        return page

    # ---------------- TAB 2: GESTURE CATALOG ----------------
    def build_catalog_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        header = QHBoxLayout()
        h_title = QLabel("ASL Gesture Catalog & Reference Guide")
        h_title.setStyleSheet("font-size: 22px; font-weight: 700; color: #f8fafc;")
        header.addWidget(h_title)
        header.addStretch()

        search_box = QLineEdit()
        search_box.setPlaceholderText("🔍 Filter gestures...")
        search_box.setFixedWidth(220)
        header.addWidget(search_box)
        layout.addLayout(header)

        # Tab widget inside Catalog: 1. Alphabet Grid, 2. Visual Poster
        cat_tabs = QTabWidget()
        cat_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #1e293b; background: #0f172a; border-radius: 8px; }
            QTabBar::tab { background: #1e293b; color: #94a3b8; padding: 8px 16px; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 4px; font-weight: 600; }
            QTabBar::tab:selected { background: #2563eb; color: white; }
        """)

        # Sub-tab 1: Grid list of all gestures
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_container = QWidget()
        self.catalog_grid = QGridLayout(grid_container)
        self.catalog_grid.setSpacing(12)

        cards = []
        col_count = 3
        for i, g in enumerate(GESTURE_CATALOG):
            card = QFrame()
            card.setProperty("class", "card")
            c_layout = QVBoxLayout(card)
            c_layout.setContentsMargins(12, 12, 12, 12)
            c_layout.setSpacing(6)

            top_h = QHBoxLayout()
            badge = QLabel(g["char"])
            badge.setStyleSheet("background-color: #2563eb; color: white; font-size: 18px; font-weight: 800; padding: 4px 12px; border-radius: 6px;")
            top_h.addWidget(badge)
            name_lbl = QLabel(g["name"])
            name_lbl.setStyleSheet("font-size: 14px; font-weight: 700; color: #f1f5f9;")
            top_h.addWidget(name_lbl)
            top_h.addStretch()
            type_tag = QLabel(g["type"])
            type_tag.setStyleSheet("color: #64748b; font-size: 11px;")
            top_h.addWidget(type_tag)
            c_layout.addLayout(top_h)

            desc = QLabel(g["desc"])
            desc.setStyleSheet("color: #94a3b8; font-size: 12px; line-height: 1.4;")
            desc.setWordWrap(True)
            c_layout.addWidget(desc)

            row = i // col_count
            col = i % col_count
            self.catalog_grid.addWidget(card, row, col)
            cards.append((g, card))

        def filter_cards(txt):
            txt = txt.strip().lower()
            for g, card in cards:
                matches = (txt in g["char"].lower()) or (txt in g["name"].lower()) or (txt in g["desc"].lower())
                card.setVisible(matches)

        search_box.textChanged.connect(filter_cards)

        scroll.setWidget(grid_container)
        cat_tabs.addTab(scroll, "Alphabet & Signs (A-Z)")

        # Sub-tab 2: Visual Chart (template.png)
        chart_scroll = QScrollArea()
        chart_scroll.setWidgetResizable(True)
        chart_container = QWidget()
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setAlignment(Qt.AlignCenter)

        chart_lbl = QLabel()
        if os.path.exists('template.png'):
            pm = QPixmap('template.png')
            chart_lbl.setPixmap(pm.scaled(480, 720, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            chart_lbl.setText("template.png not found")
        chart_layout.addWidget(chart_lbl)
        chart_scroll.setWidget(chart_container)
        cat_tabs.addTab(chart_scroll, "Visual ASL Poster (Template)")

        layout.addWidget(cat_tabs)
        return page

    # ---------------- TAB 3: CUSTOM GESTURES STUDIO ----------------
    def build_custom_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        h_title = QLabel("Custom Gesture Studio")
        h_title.setStyleSheet("font-size: 22px; font-weight: 700; color: #f8fafc;")
        layout.addWidget(h_title)

        content = QHBoxLayout()
        content.setSpacing(16)

        # Left: Capture tools
        c_left = QFrame()
        c_left.setProperty("class", "card")
        cl_layout = QVBoxLayout(c_left)
        cl_layout.setSpacing(12)

        cl_title = QLabel("Record New Gesture")
        cl_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #f8fafc;")
        cl_layout.addWidget(cl_title)

        lbl_inst = QLabel("1. Enter a unique gesture name below.\n2. Position your hand inside the camera target box.\n3. Click 'Capture Gesture' (or press Shift+S).")
        lbl_inst.setStyleSheet("font-size: 12px; color: #94a3b8; line-height: 1.5;")
        cl_layout.addWidget(lbl_inst)

        self.txt_gest_name = QLineEdit()
        self.txt_gest_name.setPlaceholderText("Enter gesture name (e.g. OK, ThumbsUp, Hello)...")
        cl_layout.addWidget(self.txt_gest_name)

        btn_capture = QPushButton("📸  Capture & Save Gesture")
        btn_capture.setProperty("class", "btn-primary")
        btn_capture.clicked.connect(self.capture_custom_gesture)
        cl_layout.addWidget(btn_capture)

        cl_layout.addStretch()
        content.addWidget(c_left, stretch=1)

        # Right: Saved Gestures Gallery
        c_right = QFrame()
        c_right.setProperty("class", "card")
        cr_layout = QVBoxLayout(c_right)
        cr_layout.setSpacing(12)

        cr_title = QLabel("Saved Gestures Gallery")
        cr_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #f8fafc;")
        cr_layout.addWidget(cr_title)

        self.gallery_scroll = QScrollArea()
        self.gallery_scroll.setWidgetResizable(True)
        self.gallery_widget = QWidget()
        self.gallery_layout = QVBoxLayout(self.gallery_widget)
        self.gallery_scroll.setWidget(self.gallery_widget)
        cr_layout.addWidget(self.gallery_scroll)

        content.addWidget(c_right, stretch=1)
        layout.addLayout(content)
        return page

    def refresh_custom_gallery(self):
        # Clear existing items
        while self.gallery_layout.count():
            item = self.gallery_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        files = fileSearch()
        if not files:
            no_lbl = QLabel("No custom gestures saved yet.")
            no_lbl.setStyleSheet("color: #64748b; font-size: 13px; padding: 20px;")
            self.gallery_layout.addWidget(no_lbl)
            return

        for f in files:
            row = QFrame()
            row.setStyleSheet("background: #0b1120; border-radius: 8px; padding: 6px;")
            r_lay = QHBoxLayout(row)

            img_p = os.path.join("SampleGestures", f)
            icon_lbl = QLabel()
            pm = QPixmap(img_p)
            icon_lbl.setPixmap(pm.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            r_lay.addWidget(icon_lbl)

            name = f.replace('.png', '')
            name_lbl = QLabel(f"Gesture: '{name}'")
            name_lbl.setStyleSheet("font-size: 13px; font-weight: 600; color: #f1f5f9;")
            r_lay.addWidget(name_lbl)
            r_lay.addStretch()

            btn_del = QPushButton("Delete")
            btn_del.setProperty("class", "btn-danger")
            btn_del.setFixedSize(65, 28)
            btn_del.clicked.connect(lambda checked, fn=f: self.delete_custom_gesture(fn))
            r_lay.addWidget(btn_del)

            self.gallery_layout.addWidget(row)
        self.gallery_layout.addStretch()

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
            reply = QMessageBox.question(self, "Confirm Delete", f"Are you sure you want to delete '{fname}'?", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                os.remove(path)
                self.refresh_custom_gallery()

    # ---------------- TAB 4: EXPORT & SPEECH ----------------
    def build_export_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        h_title = QLabel("Text Export & Audio Playback")
        h_title.setStyleSheet("font-size: 22px; font-weight: 700; color: #f8fafc;")
        layout.addWidget(h_title)

        card = QFrame()
        card.setProperty("class", "card")
        c_layout = QVBoxLayout(card)
        c_layout.setSpacing(12)

        lbl_desc = QLabel("Review formed sentences, play back with text-to-speech, or export directly to a text file.")
        lbl_desc.setStyleSheet("color: #94a3b8; font-size: 13px;")
        c_layout.addWidget(lbl_desc)

        self.export_editor = QTextEdit()
        self.export_editor.setMinimumHeight(240)
        c_layout.addWidget(self.export_editor)

        btn_row = QHBoxLayout()
        btn_speak = QPushButton("🔊  Speak Text Aloud")
        btn_speak.setProperty("class", "btn-primary")
        btn_speak.clicked.connect(self.speak_export_text)
        btn_row.addWidget(btn_speak)

        btn_save_file = QPushButton("💾  Save to .txt File")
        btn_save_file.setProperty("class", "btn-success")
        btn_save_file.clicked.connect(self.save_export_to_file)
        btn_row.addWidget(btn_save_file)

        btn_clear = QPushButton("🗑  Clear")
        btn_clear.setProperty("class", "btn-secondary")
        btn_clear.clicked.connect(self.export_editor.clear)
        btn_row.addWidget(btn_clear)
        c_layout.addLayout(btn_row)

        layout.addWidget(card)
        return page

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

        filename, _ = QFileDialog.getSaveFileName(self, "Export File", "", "Text Files (*.txt);;All Files (*)")
        if filename:
            if not filename.endswith('.txt'):
                filename += '.txt'
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(txt)
            QMessageBox.information(self, "Saved", f"File saved successfully to:\n{filename}")

    # ---------------- CAMERA ENGINE & PREDICTION LOOP ----------------
    def start_camera(self):
        try:
            if self.cam is None or not self.cam.isOpened():
                self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not self.cam.isOpened():
                    self.cam = cv2.VideoCapture(0)
            if not self.cam.isOpened():
                self.cam_status_lbl.setText("⚪ Camera Offline")
                self.cam_status_lbl.setStyleSheet("color: #ef4444; font-size: 12px; font-weight: 600; padding: 0 16px;")
                self.lbl_cam_feed.setText("Unable to open camera.\nPlease connect a webcam.")
            else:
                self.cam_status_lbl.setText("🟢 Camera Active")
                self.cam_status_lbl.setStyleSheet("color: #10b981; font-size: 12px; font-weight: 600; padding: 0 16px;")
                self.timer.start(33)
        except Exception as e:
            self.lbl_cam_feed.setText(f"Camera initialization error: {e}")

    def toggle_camera(self):
        self.cam_active = not self.cam_active
        if self.cam_active:
            self.btn_cam_toggle.setText("Pause Camera")
            self.cam_status_lbl.setText("🟢 Camera Active")
            self.cam_status_lbl.setStyleSheet("color: #10b981; font-size: 12px; font-weight: 600; padding: 0 16px;")
            self.timer.start(33)
        else:
            self.btn_cam_toggle.setText("Resume Camera")
            self.cam_status_lbl.setText("⚪ Camera Paused")
            self.cam_status_lbl.setStyleSheet("color: #eab308; font-size: 12px; font-weight: 600; padding: 0 16px;")
            self.timer.stop()

    def on_slider_changed(self, val):
        self.hsv_slider_val = val
        self.slider_val_lbl.setText(f"{val}")

    def toggle_invert_mask(self):
        self.invert_mask = not self.invert_mask
        if self.invert_mask:
            self.btn_invert.setText("⇄ Inverted (ON)")
            self.btn_invert.setStyleSheet("background-color: #2563eb; color: #ffffff;")
        else:
            self.btn_invert.setText("⇄ Invert Mask")
            self.btn_invert.setStyleSheet("")

    def on_thresh_mode_changed(self, idx):
        if idx == 0:  # Smart Skin
            self.trackbar.setValue(65)
        elif idx == 1:  # Otsu
            pass
        elif idx == 2:  # Manual
            self.trackbar.setValue(120)

    def apply_preset(self, preset_type):
        if preset_type == "bright":
            self.combo_thresh_mode.setCurrentIndex(0)
            self.trackbar.setValue(85)
            self.chk_denoise.setChecked(True)
            self.chk_isolate.setChecked(True)
        elif preset_type == "normal":
            self.combo_thresh_mode.setCurrentIndex(0)
            self.trackbar.setValue(65)
            self.chk_denoise.setChecked(True)
            self.chk_isolate.setChecked(True)
        elif preset_type == "dim":
            self.combo_thresh_mode.setCurrentIndex(0)
            self.trackbar.setValue(45)
            self.chk_denoise.setChecked(True)
            self.chk_isolate.setChecked(True)

    def process_binary_threshold(self, imcrop):
        if imcrop is None or imcrop.size == 0:
            return np.zeros((image_y, image_x), dtype=np.uint8)

        mode_text = self.combo_thresh_mode.currentText()
        sens = self.trackbar.value()

        # Pre-filter: Gaussian blur to smooth high frequency camera sensor grain
        blurred = cv2.GaussianBlur(imcrop, (5, 5), 0)

        if "Smart Skin" in mode_text:
            # 1. YCrCb Biometric Skin Chrominance Segmentation
            # In human skin, Cr clusters around 133-173 and Cb around 77-127
            ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
            delta = int((sens - 65) * 0.25)
            cr_min = max(115, 133 - delta)
            cr_max = min(190, 175 + delta)
            cb_min = max(65, 77 - delta)
            cb_max = min(140, 127 + delta)
            mask_ycrcb = cv2.inRange(ycrcb, np.array([0, cr_min, cb_min]), np.array([255, cr_max, cb_max]))

            # 2. HSV Skin Segmentation
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            # Normal skin hue lies in 0-25 and 165-180
            mask_hsv1 = cv2.inRange(hsv, np.array([0, 20, 30]), np.array([25, 255, 255]))
            mask_hsv2 = cv2.inRange(hsv, np.array([165, 20, 30]), np.array([180, 255, 255]))
            mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

            # Combined Skin Mask
            mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

        elif "Otsu" in mode_text:
            # Adaptive Otsu Auto-Contrast: Finds optimal threshold automatically
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            norm_gray = clahe.apply(gray)
            _, mask = cv2.threshold(norm_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        else:
            # Manual Slider Thresholding
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            lower_bound = np.array([0, 0, 0])
            upper_bound = np.array([179, 255, sens])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Invert mask if toggled
        if self.invert_mask:
            mask = cv2.bitwise_not(mask)

        # Morphological Filtering & Denoising
        if hasattr(self, 'chk_denoise') and self.chk_denoise.isChecked():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # Open: removes small noise dots in background
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            # Close: seals small holes inside the hand palm and fingers
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            # Dilate slightly to reinforce fingertips
            mask = cv2.dilate(mask, kernel, iterations=1)

        # Isolate Largest Hand Contour (removes all background clutter)
        if hasattr(self, 'chk_isolate') and self.chk_isolate.isChecked():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_c = max(contours, key=cv2.contourArea)
                # Minimum contour area filter: only keep if area is substantial (>500 pixels)
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

        # Flip horizontally for intuitive mirror view
        frame = cv2.flip(frame, 1)

        # Coordinates for hand ROI target box
        # Standard: (150, 50) to (350, 250) on 640x480 frame
        h, w = frame.shape[:2]
        x1, y1 = int(w * 0.45), int(h * 0.15)
        x2, y2 = int(w * 0.85), int(h * 0.65)

        # Extract hand region for thresholding
        imcrop = frame[y1:y2, x1:x2]
        if imcrop.size == 0:
            return

        # Process Improved Binary Threshold
        mask = self.process_binary_threshold(imcrop)
        self.last_mask = mask

        # Save temporary frame for prediction
        save_img = cv2.resize(mask, (image_x, image_y))
        cv2.imwrite("1.png", save_img)

        # Run predictor
        pred_char, conf = predictor(mask)
        self.current_prediction = pred_char
        self.current_confidence = conf

        # Draw stylish HUD Bounding Box on Frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (56, 189, 248), 2)
        # Corner brackets
        corner_len = 16
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (16, 185, 129), 4)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (16, 185, 129), 4)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (16, 185, 129), 4)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (16, 185, 129), 4)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (16, 185, 129), 4)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), (16, 185, 129), 4)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (16, 185, 129), 4)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (16, 185, 129), 4)

        # Label on bounding box
        cv2.putText(frame, "HAND TARGET", (x1 + 6, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1, cv2.LINE_AA)

        # Render Frame to PyQt Label
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fh, fw, fc = rgb_frame.shape
        qimg = QImage(rgb_frame.data, fw, fh, fc * fw, QImage.Format_RGB888)
        self.lbl_cam_feed.setPixmap(QPixmap.fromImage(qimg).scaled(440, 310, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Render Mask
        mh, mw = mask.shape
        qmask = QImage(mask.data, mw, mh, mw, QImage.Format_Grayscale8)
        self.lbl_mask_feed.setPixmap(QPixmap.fromImage(qmask).scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Update Hero Prediction Card
        if pred_char:
            self.lbl_hero_char.setText(pred_char)
            conf_pct = int(conf * 100)
            self.conf_bar.setValue(conf_pct)
            self.conf_bar.setFormat(f"{conf_pct}% Confidence")
            self.status_badge.setText(f"Recognized: {pred_char}")
            self.status_badge.setStyleSheet("background-color: #065f46; color: #6ee7b7; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;")
        else:
            self.lbl_hero_char.setText("—")
            self.conf_bar.setValue(0)
            self.conf_bar.setFormat("Searching...")
            self.status_badge.setText("Tracking Hand")
            self.status_badge.setStyleSheet("background-color: #1e3a8a; color: #93c5fd; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;")

    # ---------------- KEYBOARD ACTIONS & SHORTCUTS ----------------
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
            QMessageBox.information(self, "Copied", "Sentence copied to clipboard!")

    def save_buffer_file(self):
        txt = self.text_sentence.toPlainText()
        if txt.strip():
            with open("temp.txt", "w", encoding='utf-8') as f:
                f.write(txt)
            QMessageBox.information(self, "Saved", "Sentence temporarily saved to buffer. You can export it in the Export tab.")

    def closeEvent(self, event):
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        if hasattr(self, 'cam') and self.cam is not None:
            self.cam.release()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Dashboard()
    win.show()
    sys.exit(app.exec_())
