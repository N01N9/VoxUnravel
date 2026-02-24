#!/bin/bash
echo "=================================================="
echo "Starting VoxUnravel - MSST (Linux)"
echo "=================================================="

# [span_11](start_span)메인 가상환경 존재 여부 확인[span_11](end_span)
if [ ! -f ".venv_main/bin/python" ]; then
    echo "[ERROR] 메인 환경(.venv_main)을 찾을 수 없습니다."
    echo "먼저 './install.sh'를 실행하여 환경을 구축하세요."
    exit 1
fi

echo "Launching application..."
[span_12](start_span)./.venv_main/bin/python main_gui.py[span_12](end_span)
