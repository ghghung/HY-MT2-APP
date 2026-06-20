"""Translation App — Hy-MT2-1.8B | Google Translate-style UI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
from langdetect import detect, DetectorFactory
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QCloseEvent
from PySide6.QtGui import QBrush, QColor, QFont, QPalette
from PySide6.QtCore import Qt, QSize, QPoint, QTimer, QThread, Signal

from model_service import ModelService

# ── reproducibility ────────────────────────────────────────
DetectorFactory.seed = 0

# ── app data dir ───────────────────────────────────────────
APP_DIR = Path.home() / ".hy_mt2_translator"
MODELS_DIR = APP_DIR / "models"
CONFIG_FILE = APP_DIR / "config.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── styling constants ──────────────────────────────────────
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F8F9FA"
BORDER = "#E0E0E0"
TEXT_PRIMARY = "#212121"
TEXT_SECONDARY = "#757575"
BLUE = "#1976D2"
BLUE_HOVER = "#1565C0"
HOVER = "#F0F0F0"
PRESSED = "#E0E0E0"
ICON_SIZE = 16


# ── config helpers ─────────────────────────────────────────
def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {"last_model_path": "", "last_source_lang": "vie", "last_target_lang": "eng"}


def save_config(data: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(data, ensure_ascii=False))


# ── model variants from HuggingFace ────────────────────────
MODEL_VARIANTS = [
    ("2-bit", [
        ("UD-IQ2_M", "723 MB"),
        ("UD-Q2_K_XL", "804 MB"),
    ]),
    ("3-bit", [
        ("UD-IQ3_XXS", "787 MB"),
        ("Q3_K_S", "872 MB"),
        ("Q3_K_M", "951 MB"),
        ("UD-Q3_K_XL", "990 MB"),
    ]),
    ("4-bit", [
        ("IQ4_XS", "1.03 GB"),
        ("Q4_K_S", "1.08 GB"),
        ("IQ4_NL", "1.08 GB"),
        ("Q4_0", "1.08 GB"),
        ("Q4_1", "1.17 GB"),
        ("Q4_K_M", "1.13 GB"),
        ("UD-Q4_K_XL", "1.17 GB"),
    ]),
    ("5-bit", [
        ("Q5_K_S", "1.27 GB"),
        ("Q5_K_M", "1.30 GB"),
        ("UD-Q5_K_XL", "1.30 GB"),
    ]),
    ("6-bit", [
        ("Q6_K", "1.47 GB"),
        ("UD-Q6_K_XL", "1.64 GB"),
    ]),
    ("8-bit", [
        ("Q8_0", "1.91 GB"),
        ("UD-Q8_K_XL", "2.40 GB"),
    ]),
    ("16-bit", [
        ("BF16", "3.59 GB"),
    ]),
]


# ── HuggingFace helpers ────────────────────────────────────
HF_REPO = "unsloth/Hy-MT2-1.8B-GGUF"
HF_URL_BASE = "https://huggingface.co"


def get_gguf_filename(variant: str) -> str:
    return f"Hy-MT2-1.8B-{variant}.gguf"


# ── Language selector ──────────────────────────────────────
class _LangPopup(QWidget):
    """Dropdown popup for language selector with search."""

    selected = Signal(str)  # language code

    def __init__(self, codes: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._languages = ModelService.LANG_CODE_TO_LABEL
        self._all_codes = codes

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Search line edit
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Find")
        self._search_edit.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._search_edit.setClearButtonEnabled(False)
        self._search_edit.setStyleSheet(f"""
            QLineEdit {{
                background-color: {LIGHT_GRAY};
                border: 1px solid {BORDER};
                border-radius: 8px;
                padding: 4px 8px;
                color: {TEXT_PRIMARY};
                font-size: 12px;
            }}
        """)
        self._search_edit.textChanged.connect(self._filter_languages)
        layout.addWidget(self._search_edit)

        # Language list
        self._list = QListWidget()
        self._list.setStyleSheet(f"""
            QListWidget {{
                background-color: {WHITE};
                border: none;
                padding: 2px 0;
            }}
            QListWidget::item {{
                padding: 3px 8px;
                border-radius: 4px;
                font-size: 13px;
            }}
            QListWidget::item:hover {{
                background-color: {HOVER};
            }}
            QListWidget::item:selected {{
                background-color: {BLUE};
                color: {WHITE};
            }}
        """)
        self._list.setMaximumHeight(250)
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)

        # Populate
        self._populate_list("")

    def _populate_list(self, filter_text: str) -> None:
        self._list.clear()
        filter_lower = filter_text.lower().strip()
        for code in self._all_codes:
            label = self._languages[code]
            display = f"{label}  ({code})"
            if filter_lower and filter_lower not in label.lower() and filter_lower not in code.lower():
                continue
            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, code)
            self._list.addItem(item)

    def _filter_languages(self, text: str) -> None:
        self._populate_list(text)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:  # type: ignore[override]
        code = item.data(Qt.ItemDataRole.UserRole)  # type: ignore[arg-type]
        if code:
            self.selected.emit(code)
            self.close()


class LanguageSelector(QWidget):
    """Language selector with search dropdown — QPushButton + popup."""

    selected = Signal(str)  # emitted with language code

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._languages = ModelService.LANG_CODE_TO_LABEL
        self._all_codes: list[str] = []
        self._selected_code: str = "eng"
        self._popup: _LangPopup | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._btn = QPushButton()
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_style()
        self._btn.clicked.connect(self._on_button_clicked)
        layout.addWidget(self._btn)

    def _apply_style(self) -> None:
        self._btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {LIGHT_GRAY};
                border: 1px solid {BORDER};
                border-radius: 16px;
                padding: 4px 12px;
                color: {TEXT_PRIMARY};
                font-size: 13px;
                font-weight: 500;
                min-width: 100px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {HOVER};
            }}
        """)

    def _on_button_clicked(self) -> None:
        if self._popup is not None and self._popup.isVisible():
            self._popup.close()
            return

        self._popup = _LangPopup(self._all_codes, self)
        self._popup.selected.connect(self._on_selection)
        # Position popup below button
        geo = self.geometry()
        global_pos = self.mapToGlobal(QPoint(0, geo.height() + 2))
        self._popup.move(global_pos)
        self._popup.show()
        self._popup.raise_()
        self._popup._search_edit.setFocus()

    def _on_selection(self, code: str) -> None:
        self._selected_code = code
        self._update_display()
        self.selected.emit(code)

    def _update_display(self) -> None:
        self._btn.setText(self._languages.get(self._selected_code, self._selected_code))

    def sizeHint(self) -> QSize:
        return QSize(120, 28)

    def set_languages(self, codes: list[str]) -> None:
        self._all_codes = codes

    def set_selected(self, code: str) -> None:
        if code in self._languages:
            self._selected_code = code
            self._update_display()

    def selected_code(self) -> str:
        return self._selected_code

    def selected_label(self) -> str:
        return self._languages.get(self._selected_code, self._selected_code)


# ── Base panel ─────────────────────────────────────────────
class TextPanel(QWidget):
    """Panel: text area with corner action buttons that show on hover."""

    def __init__(
        self,
        text_edit: QTextEdit,
        action_buttons: list[QPushButton],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.text_edit = text_edit
        self._action_buttons = action_buttons

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Separator
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {BORDER};")
        layout.addWidget(sep)

        # Text area container with corner overlay
        text_container = QWidget()
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(2, 2, 2, 2)
        text_layout.addWidget(text_edit, 1)

        # Corner overlay for action buttons — hidden until hover
        overlay = QWidget()
        overlay.setStyleSheet("background-color: transparent;")
        overlay_layout = QHBoxLayout(overlay)
        overlay_layout.setContentsMargins(0, 0, 6, 6)
        overlay_layout.setSpacing(4)
        overlay_layout.addStretch()

        for btn in action_buttons:
            btn.setFixedSize(42, 42)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgba(255, 255, 255, 0);
                    border: none;
                    border-radius: 10px;
                    padding: 4px;
                    color: {TEXT_SECONDARY};
                    font-size: 27px;
                    opacity: 0;
                }}
                QPushButton:hover {{
                    background-color: rgba(255, 255, 255, 0.9);
                    border: 1px solid {BORDER};
                    color: {TEXT_PRIMARY};
                    opacity: 1;
                }}
                QPushButton:pressed {{
                    background-color: {PRESSED};
                    opacity: 1;
                }}
            """)
            overlay_layout.addWidget(btn)

        overlay_container = QWidget()
        overlay_layout_outer = QVBoxLayout(overlay_container)
        overlay_layout_outer.setContentsMargins(0, 0, 0, 0)
        overlay_layout_outer.setSpacing(0)
        overlay_layout_outer.addWidget(text_container, 1)
        overlay_layout_outer.addWidget(overlay)

        layout.addWidget(overlay_container, 1)

        # Character counter
        self.char_count_label = QLabel("0")
        self.char_count_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.char_count_label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_SECONDARY};
                font-size: 11px;
                padding: 2px 8px;
            }}
        """)
        layout.addWidget(self.char_count_label)

        self._apply_style()
        self.text_edit.textChanged.connect(self._update_char_count)

    def _update_char_count(self) -> None:
        count = len(self.text_edit.toPlainText())
        self.char_count_label.setText(str(count))

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {WHITE};
                border-radius: 12px;
            }}
        """)
        self.text_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {WHITE};
                border: none;
                color: {TEXT_PRIMARY};
                font-family: 'Helvetica Neue', 'Helvetica', system-ui, sans-serif;
                font-size: 14px;
                padding: 4px;
            }}
        """)


# ── Input panel ────────────────────────────────────────────
class InputPanel(TextPanel):
    """Left panel: paste/copy buttons + text input."""

    def __init__(self, parent: QWidget | None = None) -> None:
        text_edit = QTextEdit()
        text_edit.setPlaceholderText("Nhập văn bản cần dịch...")
        paste_btn = QPushButton("⎘")
        paste_btn.setToolTip("Dán")
        copy_btn = QPushButton("⧉")
        copy_btn.setToolTip("Sao chép")
        super().__init__(text_edit, [paste_btn, copy_btn], parent)
        self.text_edit = text_edit
        self.paste_btn = paste_btn
        self.copy_btn = copy_btn

    @property
    def text(self) -> str:
        return self.text_edit.toPlainText()

    @text.setter
    def text(self, value: str) -> None:
        self.text_edit.setPlainText(value)


# ── Output panel ───────────────────────────────────────────
class OutputPanel(TextPanel):
    """Right panel: copy button + translated text."""

    def __init__(self, parent: QWidget | None = None) -> None:
        text_edit = QTextEdit()
        text_edit.setPlaceholderText("Bản dịch sẽ hiển thị ở đây...")
        text_edit.setReadOnly(True)
        copy_btn = QPushButton("⧉")
        copy_btn.setToolTip("Sao chép bản dịch")
        super().__init__(text_edit, [copy_btn], parent)
        self.text_edit = text_edit
        self.copy_btn = copy_btn

    @property
    def text(self) -> str:
        return self.text_edit.toPlainText()

    @text.setter
    def text(self, value: str) -> None:
        self.text_edit.setPlainText(value)


# ── Download thread ────────────────────────────────────────
class DownloadWorker(QThread):
    """Download a GGUF file from HuggingFace with progress."""

    progress = Signal(int, str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, url: str, dest: Path) -> None:
        super().__init__()
        self._url = url
        self._dest = dest

    def run(self) -> None:
        try:
            resp = requests.get(self._url, stream=True, timeout=30)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 8192

            self._dest.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._dest.with_suffix(".tmp")

            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if hasattr(self, "_aborted") and self._aborted:
                        tmp_path.unlink(missing_ok=True)
                        return
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(downloaded * 100 / total)
                    else:
                        pct = 0
                    mb = downloaded / (1024 * 1024)
                    self.progress.emit(pct, f"Da tai: {mb:.1f} MB")

            tmp_path.rename(self._dest)
            self.finished.emit(str(self._dest))
        except Exception as e:
            self.error.emit(str(e))

    def abort(self) -> None:
        self._aborted = True
        self.quit()


# ── Main window ────────────────────────────────────────────
class MainWindow(QMainWindow):
    """Main application window — Google Translate style."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hy-MT2 Translator")
        self.setMinimumSize(900, 550)
        self._model_service = ModelService()
        self._is_translating = False
        self._loading_state = True  # flag to prevent save during init
        self._apply_palette()
        self._build_ui()
        self._setup_auto_detect()
        self._setup_languages()
        self._loading_state = False  # enable save/swap after init
        self._load_last_model()

    # ── palette ───────────────────────────────────────────

    def _apply_palette(self) -> None:
        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window, QColor(WHITE))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_PRIMARY))
        pal.setColor(QPalette.ColorRole.Text, QColor(TEXT_PRIMARY))
        pal.setColor(QPalette.ColorRole.BrightText, QColor(WHITE))
        pal.setColor(QPalette.ColorRole.Button, QColor(WHITE))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT_PRIMARY))
        self.setPalette(pal)

    # ── UI construction ───────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 0, 8, 0)
        main_layout.setSpacing(0)

        # ── Top bar: model selector + translate button ──
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(8, 4, 8, 4)
        top_layout.setSpacing(8)

        # Model selector
        self.model_combo = QComboBox()
        self.model_combo.setFixedHeight(30)
        self.model_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.model_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {LIGHT_GRAY};
                border: 1px solid {BORDER};
                border-radius: 15px;
                color: {TEXT_PRIMARY};
                font-size: 12px;
                padding: 0 10px;
                text-align: left;
                min-width: 200px;
            }}
            QComboBox:hover {{
                background-color: {HOVER};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {TEXT_SECONDARY};
            }}
            QComboBox QAbstractItemView {{
                background-color: {WHITE};
                border: 1px solid {BORDER};
                border-radius: 8px;
                selection-background-color: {BLUE};
                selection-color: {WHITE};
                color: {TEXT_PRIMARY};
                outline: none;
                padding: 4px 8px;
            }}
        """)
        self._build_model_combo_items()
        self.model_combo.setMaxVisibleItems(14)
        self.model_combo.activated.connect(self._on_model_selected)
        top_layout.addWidget(self.model_combo)

        top_layout.addStretch()

        # Translate button — compact circle in top-right
        self.translate_btn = QPushButton("▶")
        self.translate_btn.setFixedSize(32, 32)
        self.translate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.translate_btn.setToolTip("Dich")
        self.translate_btn.setStyleSheet(f"""
            QPushButton #translate {{
                background-color: {BLUE};
                border: none;
                border-radius: 16px;
                color: {WHITE};
                font-size: 16px;
                padding: 0;
            }}
            QPushButton #translate:hover {{
                background-color: {BLUE_HOVER};
            }}
            QPushButton #translate:pressed {{
                background-color: #0D47A1;
            }}
        """)
        self.translate_btn.setObjectName("translate")
        self.translate_btn.clicked.connect(self.translate)
        top_layout.addWidget(self.translate_btn)

        main_layout.addWidget(top_bar)

        # ── Language bar ──
        self.left_language_sel = LanguageSelector()
        self.right_language_sel = LanguageSelector()

        lang_bar = QWidget()
        lang_bar_layout = QHBoxLayout(lang_bar)
        lang_bar_layout.setContentsMargins(12, 4, 12, 4)
        lang_bar_layout.setSpacing(8)
        lang_bar_layout.addWidget(self.left_language_sel, 1)

        # Swap button
        self.swap_btn = QPushButton("⟷")
        self.swap_btn.setFixedSize(30, 30)
        self.swap_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.swap_btn.setToolTip("Doi ngon ngu")
        self.swap_btn.setStyleSheet(f"""
            QPushButton {{
                background: none;
                border: none;
                color: {TEXT_SECONDARY};
                font-size: 18px;
                padding: 0;
            }}
            QPushButton:hover {{
                color: {BLUE};
            }}
            QPushButton:pressed {{
                color: {BLUE_HOVER};
            }}
        """)
        self.swap_btn.clicked.connect(self._swap_languages)
        lang_bar_layout.addWidget(self.swap_btn)

        lang_bar_layout.addWidget(self.right_language_sel, 1)
        main_layout.addWidget(lang_bar)

        # ── Panels: left input | right output ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background-color: {BORDER}; }}")
        self.left_panel = InputPanel()
        self.right_panel = OutputPanel()
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter, 1)

        # ── Status bar ──
        self.status_label = QLabel("Sang hien")
        self.status_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 10px; padding: 2px 8px;"
        )
        main_layout.addWidget(self.status_label)

        # ── Signals ──
        self.left_panel.text_edit.textChanged.connect(self._on_text_changed)
        self.left_panel.paste_btn.clicked.connect(self._paste_text)
        self.left_panel.copy_btn.clicked.connect(self._copy_input)
        self.right_panel.copy_btn.clicked.connect(self._copy_output)
        self.left_language_sel.selected.connect(self._on_language_changed)
        self.right_language_sel.selected.connect(self._on_language_changed)

    def _setup_languages(self) -> None:
        codes = list(ModelService.LANG_CODE_TO_LABEL.keys())
        self.left_language_sel.set_languages(codes)
        self.right_language_sel.set_languages(codes)
        cfg = load_config()
        src = cfg.get("last_source_lang", "vie")
        tgt = cfg.get("last_target_lang", "eng")
        self.left_language_sel.set_selected(src)
        self.right_language_sel.set_selected(tgt)

        # Handle auto-swap during init if both languages happen to match
        if src == tgt:
            for code in codes:
                if code != src:
                    self.right_language_sel.set_selected(code)
                    break
            # Save swapped config
            save_config({"last_source_lang": src, "last_target_lang": self.right_language_sel.selected_code()})

    # ── Auto detect & translate ───────────────────────────

    def _setup_auto_detect(self) -> None:
        self._auto_detect_timer = QTimer()
        self._auto_detect_timer.setSingleShot(True)
        self._auto_detect_timer.setInterval(2000)  # 2 seconds from last keystroke
        self._auto_detect_timer.timeout.connect(self._do_detect)
        # Auto-translation timer
        self._auto_translate_timer = QTimer()
        self._auto_translate_timer.setSingleShot(True)
        self._auto_translate_timer.setInterval(800)
        self._auto_translate_timer.timeout.connect(self._do_auto_translate)
        self._prev_char_count = 0
        self._prev_text = ""

    def _on_text_changed(self) -> None:
        if self._auto_detect_timer:
            self._auto_detect_timer.stop()
            self._auto_detect_timer.start()
        # Always start auto-translate timer (debounce all input)
        self._auto_translate_timer.stop()
        self._auto_translate_timer.start()

    def _do_auto_translate(self) -> None:
        if self._is_translating or not self._model_service.model:
            return
        text = self.left_panel.text
        last_char = text[-1:] if text else ""
        # Punctuation trigger
        if last_char in (".", ",", "\n", "!", "?", ":", "@", ")", "(", "*", "&"):
            self.translate()
            self._prev_char_count = len(text)
            self._prev_text = text
            return
        # Paste trigger: text grew by more than 10 chars since last check
        if len(text) - self._prev_char_count > 10 and text != self._prev_text:
            self._prev_char_count = len(text)
            self._prev_text = text
            self.translate()
            return
        # Always update counters after firing
        self._prev_char_count = len(text)
        self._prev_text = text

    def _do_detect(self) -> None:
        text = self.left_panel.text_edit.toPlainText().strip()
        if len(text) < 10:
            return
        try:
            lang = detect(text)
            code_map = {
                "vi": "vie", "en": "eng", "zh-cn": "zh-CN", "zh-tw": "zh-TW",
                "ja": "jpn", "ko": "kor", "fr": "fra", "de": "deu",
                "es": "spa", "pt": "por", "it": "ita", "nl": "nld",
                "ru": "rus", "ar": "ara", "hi": "hin", "th": "tha",
                "tr": "tur", "pl": "pol", "id": "ind", "ms": "msa",
                "uk": "ukr", "ro": "ron", "el": "ell", "sv": "swe",
                "he": "heb", "ta": "tam", "bn": "ben", "ur": "urd",
                "fil": "tgl", "cs": "ces", "da": "dan", "fi": "fin", "hrv": "hrv",
            }
            code = code_map.get(lang)
            if code and code in ModelService.LANG_CODE_TO_LABEL:
                self.left_language_sel.set_selected(code)
        except Exception:
            pass

    # ── Language auto-swap ────────────────────────────────

    def _on_language_changed(self) -> None:
        """If both panels end up with the same language, swap right panel."""
        # Skip auto-swap during init (state is loading from config)
        if getattr(self, "_loading_state", False):
            return

        left_code = self.left_language_sel.selected_code()
        right_code = self.right_language_sel.selected_code()

        if left_code != right_code:
            return

        all_codes = list(ModelService.LANG_CODE_TO_LABEL.keys())
        for code in all_codes:
            if code != left_code:
                self.right_language_sel.set_selected(code)
                break

        cfg = load_config()
        cfg["last_source_lang"] = self.left_language_sel.selected_code()
        cfg["last_target_lang"] = self.right_language_sel.selected_code()
        save_config(cfg)

    # ── Translation ───────────────────────────────────────

    def translate(self) -> None:
        text = self.left_panel.text
        if not text.strip() or self._is_translating:
            return
        if not self._model_service.model:
            self.status_label.setText("Chua chon mo hinh. Nhan MENH MO HINH de tai.")
            return

        self._is_translating = True
        source_lang = self.left_language_sel.selected_code()
        target_lang = self.right_language_sel.selected_code()

        self.right_panel.text_edit.clear()
        self.status_label.setText("Dang dich...")
        try:
            result = self._model_service.translate(text, source_lang, target_lang)
            if result:
                self.right_panel.text = result
                self.status_label.setText("Hoan thanh")
            else:
                self.status_label.setText("Khong co ket qua")
        except Exception as e:
            self.status_label.setText(f"Lai: {e}")
        finally:
            self._is_translating = False

    # ── Actions ───────────────────────────────────────────

    def _swap_languages(self) -> None:
        src_code = self.left_language_sel.selected_code()
        tgt_code = self.right_language_sel.selected_code()
        src_text = self.left_panel.text
        tgt_text = self.right_panel.text

        if hasattr(self, '_auto_translate_timer'):
            self._auto_translate_timer.stop()

        self.left_language_sel.set_selected(tgt_code)
        self.right_language_sel.set_selected(src_code)
        self.left_panel.text = tgt_text
        self.right_panel.text = src_text

        if hasattr(self, '_auto_detect_timer') and self._auto_detect_timer:
            self._auto_detect_timer.stop()
            self._auto_detect_timer.start()

        if tgt_text.strip():
            QTimer.singleShot(1500, self.translate)

        cfg = load_config()
        cfg["last_source_lang"] = src_code
        cfg["last_target_lang"] = tgt_code
        save_config(cfg)

    def _paste_text(self) -> None:
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if text:
            self.left_panel.text_edit.setPlainText(text)
            if self._auto_detect_timer:
                self._auto_detect_timer.stop()
                self._auto_detect_timer.start()

    def _copy_input(self) -> None:
        clipboard = QApplication.clipboard()
        clipboard.setText(self.left_panel.text_edit.toPlainText())

    def _copy_output(self) -> None:
        clipboard = QApplication.clipboard()
        clipboard.setText(self.right_panel.text_edit.toPlainText())

    # ── Model selector (all variants with sizes) ──────────

    def _build_model_combo_items(self) -> None:
        self.model_combo.clear()
        self.model_combo.addItem("\U0001F4C2 Chon file GGUF tu local...")
        self.model_combo.addItem("")
        local_files = {f.name for f in MODELS_DIR.glob("*.gguf")}
        for group_name, variants in MODEL_VARIANTS:
            self.model_combo.addItem(f"── {group_name} ──")
            idx = self.model_combo.count() - 1
            self.model_combo.model().item(idx).setFlags(Qt.ItemFlag.NoItemFlags)
            for variant_name, size_str in variants:
                filename = get_gguf_filename(variant_name)
                display = f"{variant_name}  ({size_str})"
                if filename in local_files:
                    display = f"● {display}"
                    item = self.model_combo.model().item(self.model_combo.count() - 1)
                    item.setForeground(QBrush(QColor("green")))
                self.model_combo.addItem(display, {"type": "variant", "variant": variant_name, "filename": filename})

    def _update_model_combo(self) -> None:
        current = self.model_combo.currentIndex()
        self._build_model_combo_items()
        if current > 0:
            self.model_combo.setCurrentIndex(current)
        else:
            path = self._model_service.model_path
            if path:
                name = Path(path).name
                for i in range(self.model_combo.count()):
                    data = self.model_combo.itemData(i)
                    if data and data.get("filename") == name:
                        self.model_combo.setCurrentIndex(i)
                        return

    def _on_model_selected(self, index: int) -> None:
        if index == 0:
            # Browse local file
            self._pick_model_file()
            self.model_combo.setCurrentIndex(0)
            return
        data = self.model_combo.itemData(index)
        if not data:
            return
        if data.get("type") == "variant":
            variant = data["variant"]
            self._download_and_load(variant)

    def _load_model_by_filename(self, filename: str) -> None:
        filepath = MODELS_DIR / filename
        if not filepath.is_file():
            return
        self._load_model_by_path(str(filepath))

    def _download_and_load(self, variant: str) -> None:
        filename = get_gguf_filename(variant)
        dest = MODELS_DIR / filename

        if dest.exists():
            self._load_model_by_path(str(dest))
            return

        url = f"{HF_URL_BASE}/{HF_REPO}/resolve/main/{filename}"
        self.status_label.setText(f"Da tai {filename}...")
        self.model_combo.setEnabled(False)

        # Abort any previous download before starting a new one
        if hasattr(self, "_download_worker") and self._download_worker.isRunning():
            self._download_worker.abort()
            self._download_worker.wait()

        self._download_worker = DownloadWorker(url, dest)
        self._download_worker.progress.connect(lambda pct, msg: self.status_label.setText(f"{msg} ({pct}%)..."))
        self._download_worker.finished.connect(lambda path: self._finish_download(path, filename))
        self._download_worker.error.connect(lambda err: self._download_failed(err, filename))
        self._download_worker.start()

    def _finish_download(self, path: str, filename: str) -> None:
        self.model_combo.setEnabled(True)
        self.status_label.setText(f"Da tai xong: {filename}")
        self._load_model_by_path(path)

    def _download_failed(self, err: str, filename: str) -> None:
        self.model_combo.setEnabled(True)
        QMessageBox.critical(self, "Loi tai mo hinh", f"Khong the tai {filename}:\n{err}")
        self.status_label.setText("Tai mo hinh that bai")

    def _pick_model_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Chon file GGUF", str(MODELS_DIR),
            "GGUF Files (*.gguf);;All Files (*)",
        )
        if path:
            self._load_model_by_path(path)

    def _load_model_by_path(self, path: str) -> None:
        self.status_label.setText("Dang tai mo hinh...")
        self.model_combo.setEnabled(False)
        try:
            self._model_service.load_model(path)
            cfg = load_config()
            cfg["last_model_path"] = path
            save_config(cfg)
            self.status_label.setText(f"Da tai: {Path(path).name}")
            self._update_model_combo()
            if self.left_panel.text.strip():
                QTimer.singleShot(500, self.translate)
        except Exception as e:
            self.status_label.setText(f"Lai: {e}")
        finally:
            self.model_combo.setEnabled(True)

    # ── Load last model ───────────────────────────────────

    def _load_last_model(self) -> None:
        cfg = load_config()
        last_path = cfg.get("last_model_path", "")
        if last_path and Path(last_path).is_file():
            self.status_label.setText("Dang tai lai mo hinh cu...")
            try:
                self._model_service.load_model(last_path)
                self.status_label.setText("Da tai mo hinh cu, san sang")
                self._update_model_combo()
                if self.left_panel.text.strip():
                    QTimer.singleShot(1000, self.translate)
            except Exception as e:
                self.status_label.setText(f"Khong tai duoc mo hinh cu: {e}")

    # ── Window close ────────────────────────────────────────

    def closeEvent(self, event: QCloseEvent) -> None:
        # Wait for download thread to finish before closing
        if hasattr(self, "_download_worker") and self._download_worker.isRunning():
            self._download_worker.abort()
            self._download_worker.wait(3000)  # wait up to 3s
        super().closeEvent(event)


# ── Entry point ────────────────────────────────────────────
def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font = QFont("Helvetica", 13)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
