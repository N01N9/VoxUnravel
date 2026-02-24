#!/bin/bash
# VoxUnravel 코랩 전용 통합 환경 설정 스크립트 (외부 파이썬 파일 불필요)

echo "=================================================="
echo "VoxUnravel Colab All-in-One Setup"
echo "=================================================="

# 1. 필수 시스템 의존성 설치
# 코랩의 기본 환경에 가상환경 라이브러리와 오디오 처리 도구 설치
[span_2](start_span)apt-get update && apt-get install -y python3.11-venv ffmpeg[span_2](end_span)

# 2. pip 수동 설치 도구 다운로드
# 가상환경 내부에서 pip을 사용할 수 있게 해주는 도구
[span_3](start_span)curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py[span_3](end_span)

BASE_PATH="/content"

# 가상환경 구축 공통 함수 정의
setup_venv() {
    local venv_name=$1
    echo -e "\n[작업 중] $venv_name 가상환경 생성 및 pip 주입..."
    
    # 가상환경 생성 (pip 없이 생성하여 충돌 방지)
    [span_4](start_span)python3.11 -m venv $BASE_PATH/$venv_name --without-pip[span_4](end_span)
    
    # 해당 가상환경의 파이썬으로 pip 강제 설치
    [span_5](start_span)$BASE_PATH/$venv_name/bin/python get-pip.py[span_5](end_span)
    
    # 기본 도구 업그레이드
    [span_6](start_span)[span_7](start_span)$BASE_PATH/$venv_name/bin/python -m pip install --upgrade pip setuptools wheel[span_6](end_span)[span_7](end_span)
}

# [1/5] Main 환경 설정
setup_venv ".venv_main"
[span_8](start_span)$BASE_PATH/.venv_main/bin/python -m pip install -r requirements_main.txt --no-cache-dir[span_8](end_span)

# [2/5] Separation 환경 설정
setup_venv ".venv_sep"
echo "Separation용 PyTorch 설치 중..."
[span_9](start_span)$BASE_PATH/.venv_sep/bin/python -m pip install torch==2.1.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121[span_9](end_span)
[span_10](start_span)$BASE_PATH/.venv_sep/bin/python -m pip install -r requirements_sep.txt[span_10](end_span)

# [3/5] Diarization 환경 설정 (특수 빌드 포함)
setup_venv ".venv_dia"
[span_11](start_span)$BASE_PATH/.venv_dia/bin/python -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121[span_11](end_span)
[span_12](start_span)$BASE_PATH/.venv_dia/bin/python -m pip install "setuptools<70.0.0" flit_core[span_12](end_span)
[span_13](start_span)$BASE_PATH/.venv_dia/bin/python -m pip install "pyannote.audio @ git+https://github.com/BUTSpeechFIT/DiariZen.git#subdirectory=pyannote-audio" --no-build-isolation[span_13](end_span)
[span_14](start_span)$BASE_PATH/.venv_dia/bin/python -m pip install "diarizen @ git+https://github.com/BUTSpeechFIT/DiariZen.git@2418425e65814cdfa5fa0ec7051b20c76bf6fa05" --no-build-isolation[span_14](end_span)
[span_15](start_span)$BASE_PATH/.venv_dia/bin/python -m pip install -r requirements_dia.txt[span_15](end_span)

# [4/5] ASR 환경 설정
setup_venv ".venv_asr"
[span_16](start_span)$BASE_PATH/.venv_asr/bin/python -m pip install torch==2.1.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121[span_16](end_span)
[span_17](start_span)$BASE_PATH/.venv_asr/bin/python -m pip install -r requirements_asr.txt[span_17](end_span)

# [5/5] data/environments.json 생성 (Here Doc 기능을 사용하여 sh에서 직접 생성)
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
[span_18](start_span)

echo -e "\n=================================================="
echo "   설치 완료! 모든 가상환경이 구축되었습니다."
echo "   실행 예시: !./.venv_main/bin/python main.py"
echo "=================================================="
