#!/bin/bash
echo "================================================================="
echo " HỆ THỐNG NHẬN DIỆN BIỂU CẢM KHUÔN MẶT - ĐỀ TÀI DEEP LEARNING "
echo "================================================================="

VENV_PYTHON="./.venv/bin/python"
VENV_UVICORN="./.venv/bin/uvicorn"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[!] Không tìm thấy môi trường ảo .venv. Đang tạo môi trường ảo mới..."
    python3 -m venv .venv
    $VENV_PYTHON -m pip install --upgrade pip
    $VENV_PYTHON -m pip install -r requirements.txt
fi

if [ ! -f "models/yolov8n-face.pt" ]; then
    echo "[*] Đang tải mô hình YOLOv8-Face..."
    $VENV_PYTHON download_yolo.py
fi

echo "[+] Khởi động FastAPI Server tại http://localhost:8000 ..."
$VENV_UVICORN main:app --reload --host 0.0.0.0 --port 8000
