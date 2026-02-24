#!/bin/bash
# VoxUnravel 코랩 전용 통합 환경 설정 스크립트

echo "=================================================="
echo "VoxUnravel Colab All-in-One Setup"
echo "=================================================="

# 1. 시스템 패키지 업데이트 및 필수 도구 설치
apt-get update && apt-get install -y python3.11-venv ffmpeg

# 2. pip 수동 설치용 스크립트 다운로드 (가상환경 내 pip 누락 방지)
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

BASE_PATH="/content"

# 가상환경 구축 공통 함수
setup_venv() {
    local venv_name=$1
    echo -e "\n[작업 중] $venv_name 생성 및 pip 주입..."
    
    # 가상환경 생성 (--without-pip으로 코랩의 경로 꼬임 방지)
    python3.11 -m venv $BASE_PATH/$venv_name --without-pip
    
    # 해당 가상환경의 파이썬으로 pip 강제 설치
    $BASE_PATH/$venv_name/bin/python get-pip.py
    
    # 가상환경 내부 pip 업그레이드
    $BASE_PATH/$venv_name/bin/python -m pip install --upgrade pip setuptools wheel
}

# [1/5] Main 환경 설정
setup_venv ".venv_main"
$BASE_PATH/.venv_main/bin/python -m pip install -r requirements_main.txt --no-cache-dir

# [2/5] Separation 환경 설정
setup_venv ".venv_sep"
$BASE_PATH/.venv_sep/bin/python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
$BASE_PATH/.venv_sep/bin/python -m pip install -r requirements_sep.txt --no-cache-dir

# [3/5] Diarization 환경 설정
setup_venv ".venv_dia"
$BASE_PATH/.venv_dia/bin/python -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
$BASE_PATH/.venv_dia/bin/python -m pip install "setuptools<70.0.0" wheel flit_core
$BASE_PATH/.venv_dia/bin/python -m pip install "pyannote.audio @ git+https://github.com/BUTSpeechFIT/DiariZen.git#subdirectory=pyannote-audio" --no-build-isolation
$BASE_PATH/.venv_dia/bin/python -m pip install "diarizen @ git+https://github.com/BUTSpeechFIT/DiariZen.git@2418425e65814cdfa5fa0ec7051b20c76bf6fa05" --no-build-isolation
$BASE_PATH/.venv_dia/bin/python -m pip install -r requirements_dia.txt --no-cache-dir

# [4/5] ASR 환경 설정
setup_venv ".venv_asr"
$BASE_PATH/.venv_asr/bin/python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
$BASE_PATH/.venv_asr/bin/python -m pip install wheels/pyworld-0.3.5-cp311-cp311-linux_x86_64.whl --no-cache-dir
$BASE_PATH/.venv_asr/bin/python -m pip install -r requirements_asr.txt --no-cache-dir

# [5/5] data/environments.json 생성 (코랩 절대 경로 적용)
mkdir -p data
cat <<EOF > data/environments.json
{
    "main": "$BASE_PATH/.venv_main/bin/python",
    "inference": "$BASE_PATH/.venv_sep/bin/python",
    "separation": "$BASE_PATH/.venv_sep/bin/python",
    "diarization": "$BASE_PATH/.venv_dia/bin/python",
    "asr": "$BASE_PATH/.venv_asr/bin/python",
    "pipeline": "$BASE_PATH/.venv_main/bin/python"
}
EOF

echo -e "\n=================================================="
echo "   설치가 완료되었습니다!"
echo "   실행 예시: !./.venv_main/bin/python main.py"
echo "=================================================="
