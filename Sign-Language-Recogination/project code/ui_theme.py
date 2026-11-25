"""
ui_theme.py – Design-token system and reusable styled widgets for
Sign Language Recognition Studio (Premium "Modern AI Studio" theme).

Usage:
    from ui_theme import DARK_TOKENS, LIGHT_TOKENS, build_qss, ToggleSwitch, Toast, PillBadge

Functionality note:
    This module is PURELY cosmetic. It never touches model inference,
    camera logic, or any signal/slot wiring defined in Dashboard.py.
"""

from PyQt5.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve,
    QRect, QSize, QPoint, pyqtProperty, QSettings
)
from PyQt5.QtGui import (
    QColor, QPainter, QPen, QBrush, QLinearGradient,
    QFont, QFontDatabase, QPainterPath
)
from PyQt5.QtWidgets import (
    QCheckBox, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame, QGraphicsOpacityEffect, QSizePolicy,
    QPushButton, QApplication
)

# ---------------------------------------------------------------------------
# Design Tokens
# ---------------------------------------------------------------------------

DARK_TOKENS = {
    # Backgrounds
    "bg_base":    "#0B0F1A",
    "bg_surface": "#111827",
    "bg_card":    "#151C2E",
    "bg_elevated":"#1A2340",
    "border":     "rgba(255,255,255,0.07)",

    # Accents
    "accent_start": "#6366F1",
    "accent_end":   "#22D3EE",
    "accent_mid":   "#818CF8",
    "violet":       "#A78BFA",

    # Semantic
    "success":  "#34D399",
    "warning":  "#FBBF24",
    "danger":   "#F87171",
    "info":     "#38BDF8",

    # Text
    "text_primary":   "#F1F5F9",
    "text_secondary": "#94A3B8",
    "text_muted":     "#64748B",

    # Sidebar
    "sidebar_bg":     "#0D1424",
    "sidebar_border": "rgba(255,255,255,0.06)",
    "nav_active_bg":  "#6366F1",
    "nav_hover_bg":   "rgba(99,102,241,0.12)",

    # Radius (px)
    "r_input":  "8",
    "r_btn":    "10",
    "r_card":   "14",
    "r_hero":   "20",

    # Font
    "font_family": "'Inter', 'Segoe UI Variable', 'Segoe UI', 'SF Pro Display', sans-serif",

    # Misc
    "scrollbar_handle": "#334155",
    "scrollbar_track":  "transparent",
}

LIGHT_TOKENS = {
    "bg_base":    "#F8FAFC",
    "bg_surface": "#FFFFFF",
    "bg_card":    "#F1F5F9",
    "bg_elevated":"#E2E8F0",
    "border":     "rgba(0,0,0,0.08)",

    "accent_start": "#6366F1",
    "accent_end":   "#22D3EE",
    "accent_mid":   "#818CF8",
    "violet":       "#7C3AED",

    "success":  "#059669",
    "warning":  "#D97706",
    "danger":   "#DC2626",
    "info":     "#0284C7",

    "text_primary":   "#0F172A",
    "text_secondary": "#475569",
    "text_muted":     "#94A3B8",

    "sidebar_bg":     "#F1F5F9",
    "sidebar_border": "rgba(0,0,0,0.07)",
    "nav_active_bg":  "#6366F1",
    "nav_hover_bg":   "rgba(99,102,241,0.10)",

    "r_input":  "8",
    "r_btn":    "10",
    "r_card":   "14",
    "r_hero":   "20",

    "font_family": "'Inter', 'Segoe UI Variable', 'Segoe UI', 'SF Pro Display', sans-serif",

    "scrollbar_handle": "#CBD5E1",
    "scrollbar_track":  "transparent",
}


# ---------------------------------------------------------------------------
# QSS Builder
# ---------------------------------------------------------------------------

def build_qss(tokens: dict) -> str:
    """
    Generate the full application QSS stylesheet from a token dict.
    All colours, radii and font families come exclusively from tokens.
    """
    t = tokens
    return f"""
/* ── Global ─────────────────────────────────────────────── */
QMainWindow, QDialog {{
    background-color: {t['bg_base']};
    color: {t['text_primary']};
}}
QWidget {{
    font-family: {t['font_family']};
    color: {t['text_primary']};
    font-size: 13px;
}}

/* ── Sidebar ─────────────────────────────────────────────── */
#Sidebar {{
    background-color: {t['sidebar_bg']};
    border-right: 1px solid {t['sidebar_border']};
}}
#AppTitle {{
    color: {t['accent_start']};
    font-size: 18px;
    font-weight: 700;
    padding: 20px 16px 2px 16px;
    letter-spacing: 0.5px;
}}
#AppSubtitle {{
    color: {t['text_muted']};
    font-size: 11px;
    font-weight: 500;
    padding: 0px 16px 20px 16px;
    letter-spacing: 0.3px;
}}
#SidebarDivider {{
    background: {t['sidebar_border']};
    max-height: 1px;
    margin: 4px 16px;
}}

/* Nav Buttons */
QPushButton[class="nav-btn"] {{
    background-color: transparent;
    color: {t['text_secondary']};
    font-size: 13px;
    font-weight: 600;
    text-align: left;
    padding: 11px 16px;
    border: none;
    border-radius: {t['r_btn']}px;
    margin: 2px 10px;
}}
QPushButton[class="nav-btn"]:hover {{
    background-color: {t['nav_hover_bg']};
    color: {t['text_primary']};
}}
QPushButton[class="nav-btn"]:checked {{
    background-color: {t['nav_active_bg']};
    color: #FFFFFF;
}}

/* ── Cards ───────────────────────────────────────────────── */
QFrame[class="card"] {{
    background-color: {t['bg_card']};
    border: 1px solid {t['border']};
    border-radius: {t['r_card']}px;
}}
QFrame[class="hero-card"] {{
    background-color: {t['bg_elevated']};
    border: 1px solid {t['accent_start']};
    border-radius: {t['r_hero']}px;
}}
QFrame[class="mask-card"] {{
    background-color: {t['bg_card']};
    border: 1px solid {t['border']};
    border-radius: {t['r_card']}px;
}}

/* ── Buttons ─────────────────────────────────────────────── */
QPushButton[class="btn-primary"] {{
    background-color: {t['accent_start']};
    color: white;
    font-size: 13px;
    font-weight: 600;
    border-radius: {t['r_btn']}px;
    padding: 9px 20px;
    border: none;
    min-height: 36px;
}}
QPushButton[class="btn-primary"]:hover {{
    background-color: {t['accent_mid']};
}}
QPushButton[class="btn-primary"]:pressed {{
    background-color: {t['violet']};
}}
QPushButton[class="btn-primary"]:disabled {{
    background-color: {t['text_muted']};
    color: {t['bg_card']};
}}

QPushButton[class="btn-success"] {{
    background-color: {t['success']};
    color: {t['bg_base']};
    font-size: 13px;
    font-weight: 700;
    border-radius: {t['r_btn']}px;
    padding: 9px 20px;
    border: none;
    min-height: 36px;
}}
QPushButton[class="btn-success"]:hover {{
    background-color: #6EE7B7;
}}
QPushButton[class="btn-success"]:pressed {{
    background-color: #059669;
    color: white;
}}

QPushButton[class="btn-danger"] {{
    background-color: {t['danger']};
    color: white;
    font-size: 13px;
    font-weight: 600;
    border-radius: {t['r_btn']}px;
    padding: 9px 20px;
    border: none;
    min-height: 36px;
}}
QPushButton[class="btn-danger"]:hover {{
    background-color: #FCA5A5;
    color: #7F1D1D;
}}

QPushButton[class="btn-secondary"] {{
    background-color: transparent;
    color: {t['text_secondary']};
    font-size: 13px;
    font-weight: 600;
    border-radius: {t['r_btn']}px;
    padding: 9px 18px;
    border: 1px solid {t['border']};
    min-height: 36px;
}}
QPushButton[class="btn-secondary"]:hover {{
    background-color: {t['bg_elevated']};
    color: {t['text_primary']};
    border-color: {t['accent_start']};
}}
QPushButton[class="btn-secondary"]:pressed {{
    background-color: {t['bg_card']};
}}

QPushButton[class="btn-mini"] {{
    background-color: {t['bg_elevated']};
    color: {t['text_secondary']};
    font-size: 11px;
    font-weight: 600;
    border-radius: 8px;
    padding: 4px 10px;
    border: 1px solid {t['border']};
    min-height: 28px;
}}
QPushButton[class="btn-mini"]:hover {{
    background-color: {t['nav_hover_bg']};
    color: {t['text_primary']};
    border-color: {t['accent_start']};
}}
QPushButton[class="btn-mini"]:checked {{
    background-color: {t['accent_start']};
    color: white;
    border-color: {t['accent_end']};
}}

QPushButton[class="btn-quit"] {{
    background-color: transparent;
    color: {t['danger']};
    font-size: 13px;
    font-weight: 600;
    border-radius: {t['r_btn']}px;
    padding: 11px 16px;
    text-align: left;
    border: 1px solid transparent;
    margin: 2px 10px;
    min-height: 36px;
}}
QPushButton[class="btn-quit"]:hover {{
    background-color: rgba(248,113,113,0.12);
    border-color: {t['danger']};
}}

/* ── Inputs ──────────────────────────────────────────────── */
QTextEdit, QLineEdit {{
    background-color: {t['bg_surface']};
    border: 1px solid {t['border']};
    border-radius: {t['r_input']}px;
    color: {t['text_primary']};
    font-size: 14px;
    padding: 10px 12px;
    selection-background-color: {t['accent_start']};
}}
QTextEdit:focus, QLineEdit:focus {{
    border: 1.5px solid {t['accent_start']};
    background-color: {t['bg_card']};
}}
QTextEdit:hover, QLineEdit:hover {{
    border-color: {t['text_muted']};
}}

/* ── Slider ──────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 6px;
    background: {t['bg_elevated']};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {t['accent_start']}, stop:1 {t['accent_end']});
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: white;
    border: 2px solid {t['accent_start']};
    width: 18px;
    height: 18px;
    margin-top: -6px;
    margin-bottom: -6px;
    border-radius: 9px;
}}
QSlider::handle:horizontal:hover {{
    border-color: {t['accent_end']};
    background: {t['accent_end']};
}}

/* ── Progress Bar ────────────────────────────────────────── */
QProgressBar {{
    border: 1px solid {t['border']};
    border-radius: 6px;
    text-align: center;
    background-color: {t['bg_elevated']};
    color: {t['text_primary']};
    font-weight: 700;
    font-size: 11px;
    min-height: 20px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {t['accent_start']}, stop:1 {t['success']});
    border-radius: 5px;
}}
QProgressBar[conf_level="low"]::chunk {{
    background: {t['danger']};
}}
QProgressBar[conf_level="mid"]::chunk {{
    background: {t['warning']};
}}
QProgressBar[conf_level="high"]::chunk {{
    background: {t['success']};
}}

/* ── ScrollArea / ScrollBar ──────────────────────────────── */
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollBar:vertical {{
    border: none;
    background: {t['scrollbar_track']};
    width: 8px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {t['scrollbar_handle']};
    min-height: 24px;
    border-radius: 4px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    border: none;
    background: {t['scrollbar_track']};
    height: 8px;
}}
QScrollBar::handle:horizontal {{
    background: {t['scrollbar_handle']};
    min-width: 24px;
    border-radius: 4px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* ── ComboBox ────────────────────────────────────────────── */
QComboBox {{
    background-color: {t['bg_surface']};
    border: 1px solid {t['border']};
    border-radius: {t['r_input']}px;
    color: {t['text_primary']};
    font-size: 12px;
    font-weight: 600;
    padding: 7px 12px;
    min-height: 34px;
}}
QComboBox:hover {{
    border-color: {t['accent_start']};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {t['text_muted']};
    width: 0;
    height: 0;
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {t['bg_surface']};
    color: {t['text_primary']};
    selection-background-color: {t['accent_start']};
    border: 1px solid {t['border']};
    border-radius: 8px;
    padding: 4px;
    outline: none;
}}

/* ── CheckBox ────────────────────────────────────────────── */
QCheckBox {{
    color: {t['text_secondary']};
    font-size: 12px;
    font-weight: 500;
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1.5px solid {t['text_muted']};
    background: {t['bg_surface']};
}}
QCheckBox::indicator:checked {{
    background: {t['accent_start']};
    border-color: {t['accent_end']};
    image: none;
}}
QCheckBox::indicator:hover {{
    border-color: {t['accent_start']};
}}

/* ── TabWidget ───────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {t['border']};
    background: {t['bg_surface']};
    border-radius: {t['r_card']}px;
    top: -1px;
}}
QTabBar::tab {{
    background: {t['bg_card']};
    color: {t['text_muted']};
    padding: 9px 20px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 4px;
    font-weight: 600;
    font-size: 12px;
    border: 1px solid {t['border']};
    border-bottom: none;
}}
QTabBar::tab:selected {{
    background: {t['accent_start']};
    color: white;
    border-color: {t['accent_start']};
}}
QTabBar::tab:hover:!selected {{
    background: {t['bg_elevated']};
    color: {t['text_primary']};
}}

/* ── Tooltip ─────────────────────────────────────────────── */
QToolTip {{
    background-color: {t['bg_elevated']};
    color: {t['text_primary']};
    border: 1px solid {t['border']};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}}

/* ── Status badge states (dynamic property) ──────────────── */
QLabel[cam_state="live"] {{
    color: {t['success']};
}}
QLabel[cam_state="paused"] {{
    color: {t['warning']};
}}
QLabel[cam_state="error"] {{
    color: {t['danger']};
}}
QLabel[badge_state="detected"] {{
    background-color: rgba(52,211,153,0.15);
    color: {t['success']};
    border: 1px solid rgba(52,211,153,0.4);
    border-radius: 10px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
}}
QLabel[badge_state="searching"] {{
    background-color: rgba(99,102,241,0.15);
    color: {t['accent_mid']};
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 10px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
}}

/* ── Camera feed widget ──────────────────────────────────── */
QLabel[role="cam_feed"] {{
    background-color: #000710;
    border-radius: 10px;
    color: {t['text_muted']};
}}
QLabel[role="mask_feed"] {{
    background-color: #000710;
    border: 1px solid {t['border']};
    border-radius: 8px;
}}

/* ── Hero char label ─────────────────────────────────────── */
QLabel[role="hero_char"] {{
    color: {t['accent_end']};
    font-size: 80px;
    font-weight: 900;
}}

/* ── Page title / section headers ────────────────────────── */
QLabel[role="page_title"] {{
    font-size: 22px;
    font-weight: 700;
    color: {t['text_primary']};
    letter-spacing: 0.3px;
}}
QLabel[role="section_title"] {{
    font-size: 15px;
    font-weight: 700;
    color: {t['text_primary']};
}}
QLabel[role="caption"] {{
    font-size: 11px;
    font-weight: 600;
    color: {t['text_muted']};
    letter-spacing: 1px;
    text-transform: uppercase;
}}
QLabel[role="body"] {{
    font-size: 12px;
    color: {t['text_secondary']};
}}
"""


# ---------------------------------------------------------------------------
# ToggleSwitch – animated pill checkbox replacement
# ---------------------------------------------------------------------------

class ToggleSwitch(QCheckBox):
    """
    A pill-style animated toggle that is a drop-in replacement for QCheckBox.
    The checked state, text, and all signals are identical to QCheckBox.
    """
    _ANIM_MS = 150

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._offset = 0.0          # 0.0 = off, 1.0 = on
        self._anim = QPropertyAnimation(self, b"toggle_offset", self)
        self._anim.setDuration(self._ANIM_MS)
        self._anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.stateChanged.connect(self._run_anim)
        # Initial state
        self._offset = 1.0 if self.isChecked() else 0.0

    # ── pyqtProperty for QPropertyAnimation ──
    def _get_offset(self):
        return self._offset

    def _set_offset(self, v):
        self._offset = v
        self.update()

    toggle_offset = pyqtProperty(float, _get_offset, _set_offset)

    def _run_anim(self, state):
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(1.0 if state else 0.0)
        self._anim.start()

    def sizeHint(self):
        base = super().sizeHint()
        return QSize(base.width() + 42, max(base.height(), 24))

    def paintEvent(self, e):
        """Draw pill track + circle handle, then text."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        track_w, track_h = 38, 20
        track_x = 0
        track_y = (self.height() - track_h) // 2

        # Track
        track_color = QColor("#6366F1") if self._offset > 0.5 else QColor("#334155")
        painter.setBrush(QBrush(track_color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(track_x, track_y, track_w, track_h,
                                track_h // 2, track_h // 2)

        # Circle handle
        handle_d = track_h - 4
        travel = track_w - handle_d - 4
        handle_x = int(track_x + 2 + self._offset * travel)
        handle_y = track_y + 2
        painter.setBrush(QBrush(QColor("white")))
        painter.drawEllipse(handle_x, handle_y, handle_d, handle_d)

        # Text
        painter.setPen(QColor("#94A3B8"))
        text_x = track_x + track_w + 8
        painter.drawText(text_x, 0, self.width() - text_x, self.height(),
                         Qt.AlignVCenter | Qt.AlignLeft, self.text())

        painter.end()


# ---------------------------------------------------------------------------
# PillBadge – coloured status pill label
# ---------------------------------------------------------------------------

class PillBadge(QLabel):
    """
    A compact rounded pill that shows a status text.
    Call set_state("live"|"paused"|"error") to restyle.
    """
    _STATES = {
        "live":   ("#34D399", "rgba(52,211,153,0.15)"),
        "paused": ("#FBBF24", "rgba(251,191,36,0.15)"),
        "error":  ("#F87171", "rgba(248,113,113,0.15)"),
        "info":   ("#818CF8", "rgba(129,140,248,0.15)"),
    }

    def __init__(self, text: str = "", state: str = "live", parent=None):
        super().__init__(text, parent)
        self.set_state(state)

    def set_state(self, state: str):
        fg, bg = self._STATES.get(state, self._STATES["info"])
        self.setStyleSheet(
            f"color:{fg}; background:{bg}; border:1px solid {fg}40;"
            f"border-radius:10px; padding:3px 10px;"
            f"font-size:11px; font-weight:600;"
        )
        self.setProperty("cam_state", state)


# ---------------------------------------------------------------------------
# Toast – slide-in non-blocking notification (bottom-right)
# ---------------------------------------------------------------------------

class Toast(QFrame):
    """
    Shows a small transient notification anchored to the bottom-right of
    its parent widget.  Auto-dismisses after `duration_ms` milliseconds.
    Usage:
        Toast.show_in(parent_widget, "Copied to clipboard!")
    """
    def __init__(self, parent: QWidget, message: str, duration_ms: int = 2000):
        super().__init__(parent)
        self.setObjectName("Toast")
        self.setStyleSheet(
            "QFrame#Toast {"
            "  background:#1A2340; border:1px solid rgba(99,102,241,0.4);"
            "  border-radius:10px; padding:0px;"
            "}"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lbl = QLabel(message)
        lbl.setStyleSheet("color:#F1F5F9; font-size:13px; font-weight:600; background:transparent; border:none;")
        lay.addWidget(lbl)
        self.adjustSize()
        self.setFixedHeight(44)

        # Opacity animation (fade in)
        self._eff = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._eff)
        self._eff.setOpacity(0.0)
        self._fade_in = QPropertyAnimation(self._eff, b"opacity", self)
        self._fade_in.setDuration(200)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.start()

        # Auto-dismiss timer
        QTimer.singleShot(duration_ms, self._dismiss)
        self._reposition()
        self.show()
        self.raise_()

    def _reposition(self):
        p = self.parent()
        if p:
            pw, ph = p.width(), p.height()
            self.setFixedWidth(min(320, pw - 32))
            self.move(pw - self.width() - 16, ph - 60)

    def _dismiss(self):
        self._fade_out = QPropertyAnimation(self._eff, b"opacity", self)
        self._fade_out.setDuration(300)
        self._fade_out.setStartValue(1.0)
        self._fade_out.setEndValue(0.0)
        self._fade_out.finished.connect(self.deleteLater)
        self._fade_out.start()

    @staticmethod
    def show_in(parent: QWidget, message: str, duration_ms: int = 2000):
        """Factory helper – call this instead of constructing directly."""
        t = Toast(parent, message, duration_ms)
        return t


# ---------------------------------------------------------------------------
# Theme persistence helper
# ---------------------------------------------------------------------------

def load_theme_pref() -> str:
    """Return 'dark' or 'light' from QSettings (default: 'dark')."""
    s = QSettings("SignStudio", "UITheme")
    return s.value("theme", "dark")


def save_theme_pref(theme: str):
    """Persist the user's theme choice."""
    s = QSettings("SignStudio", "UITheme")
    s.setValue("theme", theme)


def apply_theme(app_or_widget, theme: str):
    """
    Apply the QSS for `theme` ('dark'/'light') to `app_or_widget`.
    Returns the token dict used.
    """
    tokens = DARK_TOKENS if theme == "dark" else LIGHT_TOKENS
    qss = build_qss(tokens)
    app_or_widget.setStyleSheet(qss)
    return tokens
