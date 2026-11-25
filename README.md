# Sign Language Recognition Studio

A Python desktop application built with **PyQt5** and **OpenCV** for real-time **American Sign Language (ASL)** alphabet (`A`–`Z`) and custom gesture recognition, sentence construction, and text-to-speech synthesis.

---

## Features

- **Live Recognition**
  - Captures live webcam video (`cv2.VideoCapture(0)`) with a mirrored **HAND TARGET** ROI box.
  - Converts hand crops into `64×64` binary masks with three thresholding modes:
    - **Smart Skin (YCrCb + HSV) - Auto**
    - **Adaptive Otsu Auto-Contrast**
    - **Manual Value / HSV Slider**
  - Includes mask inversion, morphological denoising (*Denoise & Fill Voids*), largest-contour hand isolation (*Isolate Hand*), and *Bright / Normal / Dim* lighting presets.
  - Classifies `A`–`Z` gestures via `ASLModel.h5` (26-class CNN) and displays live confidence scores.
- **Custom Gestures Studio**
  - Capture and save custom gesture masks as `.png` templates inside `SampleGestures/` (including `sp.png` for Space and `..png` for Period).
  - Custom gestures are matched before CNN inference using OpenCV **SIFT + FLANN** feature matching.
- **Gesture Reference**
  - Searchable catalog of `A`–`Z`, Space, and Period signs along with the visual reference chart (`template.png`).
- **Export & Speech**
  - Build sentences from recognized gestures, play them aloud offline with `pyttsx3`, copy to clipboard, or export to a `.txt` file.
- **Lightweight Model Loader**
  - `model_loader.py` loads `ASLModel.h5` weights via `h5py` and executes inference in NumPy, automatically installing a `keras` module shim if TensorFlow/Keras is not installed.

---

## Project Structure

```text
.
├── run_app.bat
└── Sign-Language-Recogination/
    └── project code/
        ├── Dashboard.py                          # Main PyQt5 Studio application
        ├── model_loader.py                       # NumPy + h5py CNN inference & keras shim
        ├── ASLModel.h5                           # Pretrained 26-class ASL CNN weights
        ├── cnn_model.py                          # CNN training script
        ├── Capture.py                            # Standalone dataset capture script
        ├── Recognise.py                          # Standalone recognition script
        ├── recogniseAppend.py                    # Standalone sentence-builder script
        ├── Dashboard_original.py                 # Original UI version
        ├── winGuiAuto.py                         # Windows GUI automation helper
        ├── template.png                          # Visual ASL binary gesture chart
        ├── Dataset/                              # Sample train_set / test_set images
        ├── SampleGestures/                       # Saved custom gesture templates
        ├── UI_Files/                             # Qt Designer .ui files
        ├── icons/                                # UI icons and assets
        ├── model.ipynb
        └── sign-language-classification_model.ipynb
```

---

## Prerequisites

- **Python 3.7+** (with webcam access for live recognition)
- Required Python packages:
  ```bash
  pip install PyQt5 opencv-python opencv-contrib-python numpy h5py pillow pyttsx3
  ```

> **Note on `qtawesome`**: The UI redesign does **not** require `qtawesome`. All icons are rendered as Unicode/text. If you install it (`pip install qtawesome`) it is not used but will not cause errors.

---

## UI Overview (v2.0 — "Modern AI Studio" Theme)

The visual layer was completely redesigned without touching any recognition, camera, or TTS logic.

### Design Tokens (`ui_theme.py`)

| Token group | Dark value | Light value |
|---|---|---|
| Background base | `#0B0F1A` | `#F8FAFC` |
| Surface | `#111827` | `#FFFFFF` |
| Card | `#151C2E` | `#F1F5F9` |
| Primary accent | `#6366F1` → `#22D3EE` (gradient) | same |
| Success | `#34D399` | `#059669` |
| Warning | `#FBBF24` | `#D97706` |
| Danger | `#F87171` | `#DC2626` |
| Font | Inter / Segoe UI Variable / Segoe UI | same |

### Theme Toggle

A ☀ / 🌙 button in the sidebar footer switches between **Dark** and **Light** themes. The choice is persisted automatically via `QSettings("SignStudio", "UITheme")` and restored on the next launch.

### New Reusable Widgets (`ui_theme.py`)

| Widget | Description |
|---|---|
| `ToggleSwitch` | Animated pill toggle, drop-in `QCheckBox` subclass |
| `PillBadge` | Coloured rounded pill (`live` / `paused` / `error` states) |
| `Toast` | Slide-in, auto-dismiss notification (bottom-right) |
| `build_qss(tokens)` | Generates the full QSS from a token dict |

---

## How to Run

### Option 1: Launcher Script (Windows)
Double-click `run_app.bat` in the repository root.

### Option 2: Terminal
```bash
cd "Sign-Language-Recogination/project code"
python Dashboard.py
```

---

## Keyboard Shortcuts

| Shortcut | Action |
| --- | --- |
| `C` | Append current predicted character to sentence |
| `Space` | Insert space |
| `Backspace` | Delete last character |
| `S` | Speak constructed sentence aloud |
| `Shift + S` | Save sentence buffer to `temp.txt` |
| `Esc` | Pause / resume camera feed |
