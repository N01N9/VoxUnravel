<div align="center">
  
# 🎧 VoxUnravel
**올인원 오디오 프로세싱 및 TTS 데이터셋 자동 구축 GUI**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)](#)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](run_colab.ipynb)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[English](README.md) • [한국어](#한국어)

</div>

---

## 📖 VoxUnravel 소개

**VoxUnravel**은 다양한 오디오 처리 작업과 **TTS(Text-to-Speech) 모델 학습용 데이터셋 구축**을 자동화하기 위해 설계된 강력하고 통합된 GUI 애플리케이션입니다.

여러 최첨단 AI 모델을 직관적인 단일 인터페이스 환경에 통합하여, 보컬 분리, 화자 분할(Diarization), 그리고 자동 음성 인식(ASR)과 같은 복잡하고 번거로운 워크플로우를 누구나 쉽게 사용할 수 있도록 단순화했습니다. AI 연구자, TTS 모델 개발자, 음향 엔지니어 등 누구나 **VoxUnravel**을 통해 무거운 작업들을 클릭 한 번으로 자동화할 수 있습니다.

---

## ✨ 핵심 기능

* 🎵 **최첨단 보컬 분리 (Vocal Separation)**
  * 노이즈나 입체적인 오디오 믹스에서 깔끔한 보컬만을 정교하게 추출합니다.
  * 지원 모델: `bs_roformer`, `mel_band_roformer`, `mdx23c`, `htdemucs` 등 최신 기술 적용.
* 🗣️ **정밀한 화자 분할 (Speaker Diarization)**
  * 단일 오디오 파일 내에 존재하는 여러 화자의 음성을 자동으로 식별하고 구간별로 분리합니다.
  * **Pyannote** 및 **DiariZen** 모델 기반으로 놀라운 정확도를 제공합니다.
* 📝 **고정밀 ASR (음성 인식)**
  * 추출된 오디오를 고품질 텍스트(전사)로 빠르고 정확하게 변환합니다.
  * 통합 모델: **Whisper Large v3**, **OWSM-CTC v4**.
* ⚡ **오토 TTS 빌더 (원클릭 파이프라인)**
  * 버튼 한 번 클릭으로 **음원 분리 ➔ 화자 분할 ➔ 음성 인식(ASR)** 과정을 순차적으로 실행합니다!
  * 오디오를 문장 단위로 자동 분할하고, `metadata.csv` 파일을 생성하며, 화자별 폴더로 데이터를 완벽하게 정리해 줍니다. 완성된 데이터셋은 즉시 TTS 모델 학습에 사용할 수 있습니다.
* 🧹 **스마트 데이터셋 클리너**
  * 생성된 TTS 데이터셋 중 너무 짧은 오디오 조각이나 텍스트 전사가 비어있는 파일을 자동으로 필터링 및 삭제하여 데이터의 품질을 높입니다.

---

## 🛠 시스템 요구 사항 (Prerequisites)

최상의 성능을 발휘하기 위해 아래의 시스템 환경을 권장합니다:

- **운영체제:** Windows 10/11 또는 Linux
- **Python:** `Python 3.11` 필수.
- **FFmpeg:** 시스템에 설치되어 있어야 하며 환경 변수(`PATH`)에 등록되어 있어야 합니다.
- **하드웨어:** 원활한 처리 속도를 위해 CUDA를 지원하고 **VRAM 16GB 이상**의 NVIDIA GPU 사용을 적극 권장합니다.

---

## 🚀 설치 및 아키텍처

서로 다른 딥러닝 프레임워크(예: Pyannote, Whisper, 음원 분리 모델) 간의 종속성 충돌 및 버전 오류를 방지하기 위해 VoxUnravel은 혁신적인 **독립 다중 환경 아키텍처(Isolated Multi-Environment Architecture)**를 사용합니다.

설치 스크립트는 **독립된 4개의 가상 환경**을 자동으로 생성하고 백그라운드에서 이를 원활하게 관리합니다.

### 💻 로컬 PC 설치 (Windows)

1. 이 저장소를 로컬 컴퓨터에 클론하거나 다운로드합니다:
   ```bash
   git clone https://github.com/N01N9/VoxUnravel.git
   cd VoxUnravel
   ```
2. `install.bat` 파일을 더블클릭합니다. (Linux 사용자는 `install.sh` 실행)
3. 커피를 한 잔 즐기며 기다려 주세요 ☕. 시스템이 자동으로 대용량 PyTorch 패키지를 다운로드하고 다음 환경을 구성합니다:
   - 🟥 `.venv_main` *(메인 GUI 및 파이프라인 관리)*
   - 🟩 `.venv_sep` *(보컬 분리 종속성)*
   - 🟦 `.venv_dia` *(화자 분할 종속성)*
   - 🟨 `.venv_asr` *(ASR 및 전사 종속성)*

### ☁️ Google Colab 환경 설치
고성능 GPU가 없으신가요? 문제 없습니다! VoxUnravel은 Google Colab 환경을 완벽하게 지원합니다.
1. Google Colab에서 [`run_colab.ipynb`](run_colab.ipynb) 파일을 엽니다.
2. 노트북 셀(Cell)의 설명에 따라 리포지토리를 가져오고 `install_colab.sh`를 실행한 후, **Gradio**를 통해 웹 GUI를 실행하세요.

---

## 🖱️ 사용 방법

### 데스크탑 GUI (PyQt5 / 로컬)
1. `start.bat`을 더블클릭하여 프로그램 인터페이스를 실행합니다. (Linux 사용자는 `start.sh`)
2. 원하는 처리 탭을 선택합니다: **보컬 분리**, **화자 분할**, **ASR**, **Auto TTS 빌더**, 또는 **데이터셋 클리너**.
3. 처리할 오디오 파일 또는 상위 폴더를 업로드합니다.
4. 특정 설정(사용할 언어 모델, 배치 사이즈 등)을 필요에 맞게 조정합니다.
5. **"Start"** 버튼을 클릭하고, 내장된 로그 패널에서 실시간 처리 진행 상황을 확인하세요!

### 웹 GUI (Gradio / Colab)
1. Colab 노트북의 마지막 셀을 실행하여 Gradio 앱을 시작합니다:
   ```bash
   !/content/VoxUnravel/.venv_main/bin/python gradio_app.py
   ```
2. 생성된 `https://xxxx.gradio.live` 링크를 클릭하여 웹 UI에 접속합니다.
3. 동일하게 옵션을 설정한 뒤, 결과를 즉시 생성해 보세요.

---

## 📂 프로젝트 구조 (Structure)

```text
VoxUnravel/
├── main_gui.py           # PyQt5 기반 데스크탑 UI의 진입점
├── gradio_app.py         # Gradio 기반 웹 UI의 진입점 (주로 Colab용)
├── runner.py             # 독립된 환경 간 크로스 스크립트 실행 관리자
├── pipeline.py           # 모든 통합 과정을 조율하는 Auto TTS Builder 클래스
├── modules/              # 보컬 분리, 화자 분할, ASR 등 핵심 추론 로직 폴더
├── configs/              # 다양한 모델을 위한 설정 파일 (YAML/JSON)
├── data/                 # 생성된 가상 환경 매핑 정보 저장 (environments.json)
├── run_colab.ipynb       # Google Colab 배포 파일
└── install.bat / .sh     # 자동화된 가상 환경 구성 스트립트
```

---

## ⚖️ 라이선스 및 감사의 글 (Acknowledgements)

이 프로젝트는 오픈 소스 커뮤니티의 놀라운 업적과 생태계를 기반으로 구축되었습니다. VoxUnravel을 탄생할 수 있게 해 준 다음 AI 관련 저장소의 기여자분들께 깊은 감사를 드립니다(Shoutout!):

### 🎯 코어 프레임워크 및 GUI 구성
- [PySide6](https://doc.qt.io/qtforpython-6/) by Qt (LGPL/GPL)
- [Gradio](https://github.com/gradio-app/gradio) (Apache 2.0)

### 🎵 보컬 및 음원 분리 (Vocal Separation)
- [Music-Source-Separation-Training (MSST)](https://github.com/ZFTurbo/Music-Source-Separation-Training) by ZFTurbo (MIT)
- [nd-Mamba2-torch](https://github.com/Human9000/nd-Mamba2-torch) by Human9000 (MIT/Apache 2.0)
- [SCNet-PyTorch](https://github.com/amanteur/SCNet-PyTorch) by amanteur (MIT)

### 🗣️ 화자 분할 (Speaker Diarization)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio) by pyannote (MIT)
- [DiariZen](https://github.com/BUTSpeechFIT/DiariZen) by BUTSpeechFIT
  - **코드 라이선스:** MIT
  - **사전 학습된 모델 가중치 (Weights):** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (비상업적 용도만 허용됨)

### 📝 음성 인식 (ASR)
- [Whisper](https://github.com/openai/whisper) by OpenAI (MIT)
- [ESPnet (OWSM-CTC)](https://github.com/espnet/espnet) by ESPnet (Apache 2.0)

### 📌 프로젝트 라이선스: GPL-3.0
이 프로젝트에 통합된 다양한 라이브러리들 중 가장 엄격한 제약 조건을 가진 라이선스(특히 코어 GUI 프레임워크인 PySide6의 카피레프트 조항)를 준수하기 위하여, **VoxUnravel** 소스 코드는 공식적으로 **GNU General Public License v3.0 (GPL-3.0)** 하에 배포됩니다.

*상업적 목적으로 VoxUnravel을 통해 생성한 데이터를 사용하려 하신다면, 위 해당 리포지토리의 AI 모델들이 가지는 개별 라이선스 조항을 반드시 준수해 주시길 바랍니다.*

---
<div align="center">
  <i>AI 음성 커뮤니티를 위해 정성껏 만들었습니다. ❤️</i>
</div>
