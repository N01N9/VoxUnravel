<div align="center">
  
# 🎧 VoxUnravel
**The Ultimate All-in-One Audio Processing & TTS Dataset Builder**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)](#)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/N01N9/VoxUnravel/blob/main/run_colab.ipynb)
[![License: GPL v3 (Non-Commercial)](https://img.shields.io/badge/License-GPLv3%20(Non--Commercial)-red.svg)](https://www.gnu.org/licenses/gpl-3.0)

[English](#english) • [한국어](#한국어)

</div>

---

## 📖 About VoxUnravel

**VoxUnravel** is a powerful, vertically integrated GUI application designed for comprehensive audio processing and automated **Text-to-Speech (TTS) dataset creation**. 

By uniting various state-of-the-art AI models under a single, intuitive interface, VoxUnravel simplifies complex and tedious workflows—such as vocal extraction, speaker diarization, and automatic speech recognition (ASR)—into a seamless, user-friendly experience. Whether you are an AI researcher, a TTS model trainer, or an audio engineer, VoxUnravel automates the heavy lifting.

---

## ✨ Key Features

* 🎵 **State-of-the-Art Vocal Separation**
  * Extract pristine vocals from heavily mixed audio tracks or background noise.
  * Supports cutting-edge models: `bs_roformer`, `mel_band_roformer`, `mdx23c`, `htdemucs`, and more.
* 🗣️ **Precise Speaker Diarization**
  * Identity, segment, and isolate different speakers within a single audio file.
  * Powered by industry standards: **Pyannote** and **DiariZen**.
* 📝 **High-Accuracy ASR (Speech-to-Text)**
  * Transcribe audio into high-quality text accurately and efficiently.
  * Integrates state-of-the-art models: **Whisper Large v3** and **OWSM-CTC v4**.
* ⚡ **Auto TTS Builder (One-Click Pipeline)**
  * Run **Separation ➔ Diarization ➔ ASR** sequentially with click of a button!
  * Automatically splices audio, generates a `metadata.csv` file, and sorts segments into speaker-specific folders. Your dataset is instantly ready for TTS model training.
* 🧹 **Smart Dataset Cleaner**
  * Effortlessly sanitize generated TTS datasets by automatically filtering out excessively short audio snippets or files with empty transcriptions.

---

## 🛠 Prerequisites

To get the most out of VoxUnravel, please ensure your system meets the following requirements:

- **OS:** Windows 10/11 or Linux
- **Python:** `Python 3.11` is required.
- **FFmpeg:** Must be installed and added to your system's `PATH`.
- **Hardware:** A CUDA-compatible **NVIDIA GPU with 16GB+ VRAM** is *highly recommended* for reasonable inference times.

---

## 🚀 Installation & Architecture

To prevent dependency hell and version conflicts between various distinct Deep Learning frameworks (e.g., Pyannote, Whisper, Separation models), VoxUnravel utilizes an innovative **Isolated Multi-Environment Architecture**. 

The installation scripts automatically create **four independent virtual environments** and manage them seamlessly in the background.

### 💻 Local Installation (Windows)

1. Clone or download this repository to your local machine:
   ```bash
   git clone https://github.com/N01N9/VoxUnravel.git
   cd VoxUnravel
   ```
2. Double-click on `install.bat` (or run `install.sh` on Linux).
3. Grab a coffee ☕. The script will automatically create the following environments and download necessary PyTorch packages:
   - 🟥 `.venv_main` *(Main GUI & Pipeline Manager)*
   - 🟩 `.venv_sep` *(Vocal Separation dependencies)*
   - 🟦 `.venv_dia` *(Speaker Diarization dependencies)*
   - 🟨 `.venv_asr` *(ASR / Transcription dependencies)*

### ☁️ Google Colab Installation
Don't have a high-end GPU? No problem! VoxUnravel fully supports Google Colab.
1. Open [`run_colab.ipynb`](https://colab.research.google.com/github/N01N9/VoxUnravel/blob/main/run_colab.ipynb) in Google Colab.
2. Follow the cell instructions to pull the repo, run `install_colab.sh`, and launch the Web GUI via **Gradio**.

---

## 🖱️ Usage Guide

### PyQt5 Desktop GUI (Local)
1. Double-click on `start.bat` (or run `start.sh`) to launch the application interface.
2. Select your desired processing tab: **Separation**, **Diarization**, **ASR**, **Auto TTS Builder**, or **Clean Dataset**.
3. Upload your target audio files or select a batch folder.
4. Tweak specific settings (Model type, Language, Batch size, etc.).
5. Click **"Start"** and monitor real-time progress using the built-in log panel!

### Gradio Web GUI (Colab)
1. At the bottom of the Colab notebook, run the final cell to start the Gradio app:
   ```bash
   !/content/VoxUnravel/.venv_main/bin/python gradio_app.py
   ```
2. Click the generated `https://xxxx.gradio.live` link to access the Web UI.
3. Configure your job and watch the results generate instantly.

---

## 📂 Project Structure

```text
VoxUnravel/
├── main_gui.py           # Entry point for the PyQt5-based Desktop UI
├── gradio_app.py         # Entry point for the Gradio-based Web UI (Colab)
├── runner.py             # Handles cross-environment script executions
├── pipeline.py           # Orchestrates the integrated Auto TTS Builder pipeline
├── modules/              # Core inference logic for Separation, Diarization, and ASR
├── configs/              # Model configurations (YAML/JSON)
├── data/                 # Stores generated environment mappings (environments.json)
├── run_colab.ipynb       # Google Colab deployment notebook
└── install.bat / .sh     # Automated environment setup scripts
```

---

## ⚖️ License & Acknowledgements

VoxUnravel strongly believes in the power of open-source software and is built upon the amazing work of the AI community. Huge shoutout to the following repositories that made this project possible:

### Core Frameworks & GUI
- [PySide6](https://doc.qt.io/qtforpython-6/) by Qt (LGPL/GPL)
- [Gradio](https://github.com/gradio-app/gradio) (Apache 2.0)

### 🎵 Vocal Separation
- [Music-Source-Separation-Training (MSST)](https://github.com/ZFTurbo/Music-Source-Separation-Training) by ZFTurbo (MIT)
- [nd-Mamba2-torch](https://github.com/Human9000/nd-Mamba2-torch) by Human9000 (MIT/Apache 2.0)
- [SCNet-PyTorch](https://github.com/amanteur/SCNet-PyTorch) by amanteur (MIT)

### 🗣️ Speaker Diarization
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio) by pyannote (MIT)
- [DiariZen](https://github.com/BUTSpeechFIT/DiariZen) by BUTSpeechFIT 
  - **Code License:** MIT
  - **Pre-trained Model Weights:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Non-Commercial use only)

### 📝 Speech-to-Text (ASR)
- [Whisper](https://github.com/openai/whisper) by OpenAI (MIT)
- [ESPnet (OWSM-CTC)](https://github.com/espnet/espnet) by ESPnet (Apache 2.0)

### 📌 Project License: GPL-3.0 (Non-Commercial)
To conform to the most restrictive licensing requirements among its dependencies (specifically to respect the copyleft stipulations inherited from PySide6), the **VoxUnravel** source code is officially bounded by and distributed under the **GNU General Public License v3.0 (GPL-3.0)**. 

**🚨 [MANDATORY] NO COMMERCIAL USE ALLOWED**
This project inherently relies on **DiariZen** models for speaker diarization. The pre-trained weights for these models are licensed under **CC BY-NC 4.0**, strictly prohibiting any form of commercial use. 
To prevent any subsequent legal disputes or licensing conflicts, **any commercial use of VoxUnravel—inclusive of the application code itself and any audio/text datasets generated through it—is strictly prohibited.** Please feel free to use it *exclusively* for personal projects or research purposes.

---
<div align="center">
  <i>Created with ❤️ for the AI Voice Community.</i>
</div>
