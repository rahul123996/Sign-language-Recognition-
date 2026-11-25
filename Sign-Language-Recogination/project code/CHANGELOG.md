# CHANGELOG

## [2.0.0] – 2026-11-25 — Premium "Modern AI Studio" UI Redesign

### Visual Changes Only
> **No functional logic was modified.**  
> All inference, camera, keyboard-shortcut, sentence-builder, TTS,
> custom-gesture, and export behaviours are byte-for-byte identical to v1.x.

---

### New File: `ui_theme.py`
- Centralised design-token system: `DARK_TOKENS` and `LIGHT_TOKENS` dicts.
- `build_qss(tokens)` – generates the full application QSS from tokens (colours, radii, font family all sourced from the dict, never hard-coded inline).
- `ToggleSwitch` – animated pill toggle that is a drop-in `QCheckBox` subclass (identical signal API, used for *Denoise & Fill* and *Isolate Hand* checkboxes).
- `PillBadge` – coloured rounded pill label with `set_state("live"|"paused"|"error")`.
- `Toast` – slide-in, auto-dismiss notification widget anchored to bottom-right; replaces `QMessageBox.information` for *Copied* and *Saved* actions only.
- `load_theme_pref()` / `save_theme_pref()` – QSettings persistence for the user's theme choice.
- `apply_theme()` – convenience wrapper.

### Updated File: `Dashboard.py`
- **Imports**: Added `from ui_theme import …`; added HiDPI flags (`AA_EnableHighDpiScaling`, `AA_UseHighDpiPixmaps`) before QApplication.
- **Theme system**: Replaced monolithic `MODERN_STYLE` string with `build_qss(DARK_TOKENS)`. Added `_toggle_theme()` and a sun/moon button in the sidebar footer that persists to `QSettings`.
- **Window**: Min size bumped to 1100 × 720; default launch size 1280 × 800.
- **Sidebar** (`init_ui`):
  - Gradient "AI" badge next to logo text.
  - Navigation buttons restyled with new QSS classes; hover/active accent glow.
  - Camera status now uses `PillBadge` with `set_state()` calls.
  - Thin dividers between logo, nav, and footer sections.
  - Theme toggle (☀ / 🌙) button with tooltip.
  - Quit button uses `btn-quit` class with red outline hover.
- **Live Recognition page** (`build_detect_page`):
  - Two-line page header (title + subtitle).
  - Camera card: enlarged to 460 × 320, caption-style label, `role="cam_feed"` property.
  - Mask card: caption-style header; `ToggleSwitch` for *Denoise* and *Isolate Hand*; gradient slider track.
  - Confidence bar: `conf_level` dynamic property drives red/amber/green QSS chunk colour via `unpolish/polish`.
  - Hero letter: 88 px, accent-cyan colour.
  - Status badge: styled inline; states: *Tracking Hand* (indigo), *Recognized: X* (green).
  - Sentence builder: live character counter; `Toast` replaces `QMessageBox` for *Copy* and *Save*.
  - All keyboard shortcuts listed in tooltips and placeholder text.
- **Gesture Reference page** (`build_catalog_page`):
  - Gradient letter badges (indigo → cyan) on each card.
  - Type tags (*Alphabet* = violet pill, *Special* = amber pill).
  - Search box always shows all cards when empty (fixed blank-search edge case).
- **Custom Gestures page** (`build_custom_page`):
  - Numbered stepper instructions (circles 1 / 2 / 3).
  - Gallery rows: rounded dark cards with thumbnail image.
  - Empty-state text centered and larger.
- **Export & Speech page** (`build_export_page`):
  - Monospace editor; live word + character counter.
  - Action bar with consistent button sizing.
- **QSS improvements** (all in `ui_theme.py`):
  - Hover / pressed / disabled states for every button class.
  - Gradient slider sub-page track (indigo → cyan).
  - `conf_level` property selector for progress-bar chunk colour.
  - Custom scrollbars (8 px, rounded, transparent track).
  - Dark rounded tooltips via `QToolTip`.
  - `border-radius` on `QComboBox` drop-down arrow; item-view padding.
  - `QCheckBox::indicator` radius and hover border.
  - `QTabBar` restyled with rounded segmented look.

### No Changes To
- `predictor()`, `process_binary_threshold()`, `update_camera_frame()`, `start_camera()`, `toggle_camera()`, `apply_preset()`, `on_slider_changed()`, `on_thresh_mode_changed()`, `toggle_invert_mask()`, `append_current_prediction()`, `add_space()`, `backspace()`, `clear_sentence()`, `speak_sentence()`, `speak_export_text()`, `save_export_to_file()`, `capture_custom_gesture()`, `delete_custom_gesture()`, `refresh_custom_gallery()`, `refresh_export_page()`, `save_buffer_file()`, `copy_sentence()` *(only the `QMessageBox` call in `copy_sentence` was replaced with `Toast.show_in` – the clipboard write logic is unchanged)*, `closeEvent()`, `keyPressEvent()`.
- `model_loader.py`, `ASLModel.h5`, `Capture.py`, `Recognise.py`, `recogniseAppend.py`, `cnn_model.py`, `winGuiAuto.py`, `Dashboard_original.py`, notebooks, Dataset/, SampleGestures/, `template.png`.
