# VoxUnravel

VoxUnravel is a powerful, integrated GUI application designed for comprehensive audio processing and automated Text-to-Speech (TTS) dataset creation. By uniting various state-of-the-art AI models under a single interface, VoxUnravel simplifies complex workflows like vocal extraction, speaker diarization, and automatic speech recognition (ASR) into an intuitive, user-friendly experience.

## ✨ Features

- **Vocal Separation**: Extract pristine vocals from audio tracks. Supports advanced models such as `bs_roformer`, `mdx23c`, `mel_band_roformer`, `htdemucs`, etc.
- **Speaker Diarization**: Identify and isolate different speakers within an audio file (powered by Pyannote and DiariZen).
- **ASR (Speech-to-Text)**: Transcribe audio accurately utilizing cutting-edge models like **Whisper Large v3** and **OWSM-CTC v4**.
- **Auto TTS Builder**: A one-click integrated pipeline that automatically runs separation, diarization, and ASR sequentially. It produces a fully organized dataset (`metadata.csv` and speaker folders) ready for TTS model training.
- **Dataset Cleaner**: Clean up your generated TTS datasets by filtering out audio files that are too short or have empty text transcriptions.

## 🛠 Prerequisites

- Windows Operating System
- **Python 3.11** (must be installed and added to PATH)
- A CUDA-compatible NVIDIA GPU is highly recommended for reasonable processing times.

## 🚀 Installation

VoxUnravel uses multiple isolated virtual environments to prevent dependency conflicts between different deep learning frameworks and tools.

1. Clone or download this repository.
2. Double-click on `install.bat`.
3. The script will automatically create and configure four specific environments:
   - `.venv_main` (Main GUI & Pipeline)
   - `.venv_sep` (Separation dependencies)
   - `.venv_dia` (Diarization dependencies)
   - `.venv_asr` (ASR dependencies)

*Note: The installation process may take some time as it downloads several large PyTorch packages.*

## 💻 Usage

Once the installation is complete:
1. Double-click on `start.bat` to launch the application.
2. Select the desired tab (Separation, Diarization, ASR, Auto TTS Builder, or Clean Dataset).
3. Choose your input files or folder.
4. Configure the specific settings for your task.
5. Click the "Start" button and monitor the progress in the log panel.

## 📂 Project Structure

- `main_gui.py`: The entry point for the PyQt5-based graphical user interface.
- `runner.py`: Handles background processing and environment management.
- `pipeline.py`: Executes the integrated TTS builder pipeline.
- `configs/`: Configuration files for various separation and ASR models.
- `modules/`: Contains the core inference logic for different audio processing steps.
- `data/environments.json`: Automatically generated mapping of virtual environments.

## 📄 License & Acknowledgements

This project integrates several open-source audio processing models and tools. Please refer to their respective repositories for licensing information:
- Pyannote / DiariZen
- Whisper (OpenAI)
- OWSM-CTC
- Music Source Separation Training (MSST) models

