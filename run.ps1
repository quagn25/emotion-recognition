# Script khởi động 1-click cho dự án (Windows PowerShell)
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " HỆ THỐNG NHẬN DIỆN BIỂU CẢM KHUÔN MẶT - ĐỀ TÀI DEEP LEARNING " -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Cyan

$VENV_PYTHON = ".\.venv\Scripts\python.exe"
$VENV_UVICORN = ".\.venv\Scripts\uvicorn.exe"

if (-not (Test-Path $VENV_PYTHON)) {
    Write-Host "[!] Không tìm thấy môi trường ảo .venv. Đang tạo môi trường ảo mới..." -ForegroundColor Yellow
    python -m venv .venv
    & $VENV_PYTHON -m pip install --upgrade pip
    & $VENV_PYTHON -m pip install -r requirements.txt
}

# Kiểm tra các tệp mô hình tối thiểu
if (-not (Test-Path "models\emotion_model.keras")) {
    Write-Host "[CẢNH BÁO] Chưa tìm thấy models\emotion_model.keras." -ForegroundColor Yellow
}
if (-not (Test-Path "models\yolov8n-face.pt")) {
    Write-Host "[*] Đang tải mô hình YOLOv8-Face..." -ForegroundColor Yellow
    & $VENV_PYTHON download_yolo.py
}

Write-Host "[+] Khởi động FastAPI Server tại http://localhost:8000 ..." -ForegroundColor Green
& $VENV_UVICORN main:app --reload --host 0.0.0.0 --port 8000
