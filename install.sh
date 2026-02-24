#!/bin/bash
# VoxUnravel 리눅스용 자동 환경 설정 스크립트

echo "=================================================="
echo "VoxUnravel Automated Environment Setup (Linux)"
echo "=================================================="

# [span_4](start_span)Python 3.11 설치 여부 확인[span_4](end_span)
python3.11 --version >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] Python 3.11이 설치되어 있지 않거나 경로가 설정되지 않았습니다."
    echo "설치 명령: sudo apt update && sudo apt install python3.11 python3.11-venv"
    exit 1
fi

PYTHON_CMD="python3.11"

# [span_5](start_span)[1/5] 메인 환경 설정 (.venv_main)[span_5](end_span)
echo -e "\n[1/5] Setting up Main Environment (.venv_main)..."
if [ ! -d ".venv_main" ]; then
    $PYTHON_CMD -m venv .venv_main
fi
.venv_main/bin/python -m pip install --upgrade pip --no-cache-dir
.venv_main/bin/python -m pip install -r requirements_main.txt --no-cache-dir

# [span_6](start_span)[2/5] 분리 환경 설정 (.venv_sep)[span_6](end_span)
echo -e "\n[2/5] Setting up Separation Environment (.venv_sep)..."
if [ ! -d ".venv_sep" ]; then
    $PYTHON_CMD -m venv .venv_sep
fi
.venv_sep/bin/python -m pip install --upgrade pip
echo "Installing PyTorch for Separation (Linux CUDA 12.1)..."
.venv_sep/bin/python -m pip install torch==2.1.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
.venv_sep/bin/python -m pip install -r requirements_sep.txt --no-cache-dir

# [span_7](start_span)[3/5] 화자 분리 환경 설정 (.venv_dia)[span_7](end_span)
echo -e "\n[3/5] Setting up Diarization Environment (.venv_dia)..."
if [ ! -d ".venv_dia" ]; then
    $PYTHON_CMD -m venv .venv_dia
fi
.venv_dia/bin/python -m pip install --upgrade pip
echo "Installing PyTorch 2.1.1 for Diarization..."
.venv_dia/bin/python -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

echo "Fixing build tools for Github packages..."
.venv_dia/bin/python -m pip install "setuptools<70.0.0" wheel flit_core

echo "Installing pyannote.audio and diarizen without build isolation..."
.venv_dia/bin/python -m pip install "pyannote.audio @ git+https://github.com/BUTSpeechFIT/DiariZen.git#subdirectory=pyannote-audio" --no-build-isolation
.venv_dia/bin/python -m pip install "diarizen @ git+https://github.com/BUTSpeechFIT/DiariZen.git@2418425e65814cdfa5fa0ec7051b20c76bf6fa05" --no-build-isolation
.venv_dia/bin/python -m pip install -r requirements_dia.txt --no-cache-dir

# [span_8](start_span)[4/5] ASR 환경 설정 (.venv_asr)[span_8](end_span)
echo -e "\n[4/5] Setting up ASR Environment (.venv_asr)..."
if [ ! -d ".venv_asr" ]; then
    $PYTHON_CMD -m venv .venv_asr
fi
.venv_asr/bin/python -m pip install --upgrade pip
.venv_asr/bin/python -m pip install torch==2.1.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
.venv_asr/bin/python -m pip install -r requirements_asr.txt --no-cache-dir

# [span_9](start_span)[5/5] data/environments.json 설정 (리눅스 경로 적용)[span_9](end_span)
echo -e "\n[5/5] Configuring data/environments.json..."
mkdir -p data
cat <<EOF > data/environments.json
{
    "main": ".venv_main/bin/python",
    "inference": ".venv_sep/bin/python",
    "separation": ".venv_sep/bin/python",
    "diarization": ".venv_dia/bin/python",
    "asr": ".venv_asr/bin/python",
    "pipeline": ".venv_main/bin/python"
}
EOF

echo "=================================================="
echo "   Setup Completed Successfully!"
echo "   Command: ./.venv_main/bin/python main_gui.py"
echo "=================================================="
