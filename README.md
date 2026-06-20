Hy-MT2 Desktop Translator

Hy-MT2 Desktop Translator is a professional, open-source translation application
for Windows, powered by the Hy-MT2-1.8B Large Language Model in GGUF format. It
provides a clean, intuitive interface inspired by Google Translate, allowing for
high-quality local translation without relying on expensive cloud APIs.

✨ Key Features

  - 🚀 High Performance: Optimized for local execution using the Hy-MT2-1.8B
    model, providing fast inference even on standard consumer hardware.
  - 🎨 Modern UI: Built with PySide6, featuring a responsive and clean layout,
    character counters, and hover actions.
  - 🌍 Multilingual Support: Seamlessly translate between Vietnamese, English,
    and over 30 other languages.
  - 🔍 Auto-Detection: Integrated language detection that identifies the source
    language as you type.
  - 📦 In-App Model Management: Easily browse, download, and switch between
    different quantization variants (from 2-bit to 16-bit) directly from
    HuggingFace.
  - ☁️ Offline Mode: Once the model is downloaded, the application works
    entirely offline, ensuring your data stays private.

🛠 Installation (For Developers)

Prerequisites

  - Python 3.9 or higher
  - C++ Compiler (Required for llama-cpp-python installation)

Steps

1.  Clone the repository:

    git clone https://github.com/[your-username]/hy-mt2-translator.git
    cd hy-mt2-translator

2.  Create and activate a virtual environment:

    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate

3.  Install dependencies:

    pip install -r requirements.txt

4.  Launch the application:

    python main.py

🚀 Building the Executable (.exe)

To package the application into a single standalone executable for Windows:

pip install pyinstaller
pyinstaller --noconsole --onefile --name "Hy-MT2-Translator" --collect-all llama_cpp --collect-all langdetect main.py

The output file will be located in the dist/ folder.

📖 How to Use

1.  Select a Model: Use the dropdown menu at the top to select a GGUF variant.
    The app will automatically download it from the HuggingFace repository if
    it’s not found locally.
2.  Input Text: Type or paste your text into the left panel. The app will
    attempt to detect the language automatically after a short delay.
3.  Translate: Click the Play (▶) button or let the auto-translate trigger
    (activated by punctuation or pasting).
4.  Output: The translation appears in the right panel. You can copy it to your
    clipboard using the copy icon.
5.  Swap: Use the ⟷ button to quickly switch source and target languages.

🗄️ Model Information

The application utilizes GGUF weights provided by Unsloth.

  - Model Repo: unsloth/Hy-MT2-1.8B-GGUF
  - Supported Quantizations: IQ2_M, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, and
    BF16.

🤝 Contributing

Contributions are welcome! If you have suggestions for new features or find any
bugs:

1.  Fork the Project.
2.  Create your Feature Branch (git checkout -b feature/AmazingFeature).
3.  Commit your Changes (git commit -m 'Add some AmazingFeature').
4.  Push to the Branch (git push origin feature/AmazingFeature).
5.  Open a Pull Request.

📄 License

Distributed under the MIT License. See LICENSE for more information.

🌟 Acknowledgments

  - Unsloth for the GGUF quantization of the Hy-MT2 model.
  - The llama.cpp community for enabling LLM inference on consumer hardware.
  - The PySide/Qt team for the robust GUI framework.

Developed with ❤️ to bring local LLM translation to everyone.
