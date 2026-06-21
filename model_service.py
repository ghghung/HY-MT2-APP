"""Model service for Hy-MT2-1.8B GGUF translation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from llama_cpp import Llama


class ModelService:
    """Loads GGUF model and provides translation."""

    LANG_CODE_TO_LABEL = {
        "eng": "English",
        "vie": "Vietnamese",
        "zh-CN": "Chinese (Simplified)",
        "zh-TW": "Chinese (Traditional)",
        "jpn": "Japanese",
        "kor": "Korean",
        "fra": "French",
        "deu": "German",
        "spa": "Spanish",
        "por": "Portuguese",
        "ita": "Italian",
        "nld": "Dutch",
        "rus": "Russian",
        "ara": "Arabic",
        "hin": "Hindi",
        "tha": "Thai",
        "tur": "Turkish",
        "pol": "Polish",
        "ind": "Indonesian",
        "msa": "Malay",
        "ukr": "Ukrainian",
        "ron": "Romanian",
        "ell": "Greek",
        "swe": "Swedish",
        "heb": "Hebrew",
        "tam": "Tamil",
        "ben": "Bengali",
        "urd": "Urdu",
        "tgl": "Filipino",
        "ces": "Czech",
        "dan": "Danish",
        "fin": "Finnish",
        "hrv": "Croatian",
    }

    LANG_LABEL_TO_CODE = {v: k for k, v in LANG_CODE_TO_LABEL.items()}

    def __init__(self) -> None:
        self.model: Optional[Llama] = None

    # ── load ───────────────────────────────────────────────

    def load_model(self, path: str) -> None:
        """Load a GGUF model from disk."""
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        if not path.lower().endswith(".gguf"):
            raise ValueError(f"Only .gguf files are supported: {path}")

        self.model = Llama(
            model_path=str(p),
            n_gpu_layers=-1,
            n_ctx=8192,
            n_batch=2048,
            verbose=False,
            lora_adapters=[],
        )
        # Model loads successfully (uses completion API, not chat API)

    # ── translate ──────────────────────────────────────────

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        on_token: Optional[Callable[[str], None]] = None,
        *,
        stream: bool = False,
    ) -> str | None:
        """Translate text. If *stream* is True, call *on_token(token)* for each token and return None.
        If *stream* is False, return the full result string."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if not text.strip():
            return ""

        # Convert lang code to label for the prompt
        tgt_label = self.LANG_CODE_TO_LABEL.get(target_lang, target_lang)

        instruction = (
            f"Translate the following text into {tgt_label}. "
            f"Note that you should **only output the translated result without "
            f"any additional explanation**:\n\n{text}"
        )

        if stream:
            result = self.model(
                instruction,
                max_tokens=4096,
                temperature=0.7,
                top_p=0.6,
                top_k=20,
                repeat_penalty=1.05,
                echo=False,
                stream=True,
            )
            for chunk in result:
                choices = chunk.get("choices", [])
                if choices:
                    text_delta = choices[0].get("text", "")
                    if on_token:
                        on_token(text_delta)
            return None
        else:
            result = self.model(
                instruction,
                max_tokens=4096,
                temperature=0.7,
                top_p=0.6,
                top_k=20,
                repeat_penalty=1.05,
                echo=False,
                stream=False,
            )

            choices = result.get("choices", [])
            if choices:
                return choices[0].get("text", "")
            return ""

    # ── helpers ────────────────────────────────────────────

    def is_model_loaded(self) -> bool:
        return self.model is not None

    @property
    def model_path(self) -> Optional[str]:
        return self.model.model_path if self.model else None
