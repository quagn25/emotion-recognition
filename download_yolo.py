import os
import sys
import urllib.request

url = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"
filename = "yolov8n-face.pt"
MIN_FILE_SIZE = 5 * 1024 * 1024  # At least 5 MB

if os.path.exists(filename):
    size = os.path.getsize(filename)
    if size >= MIN_FILE_SIZE:
        print(f"[download_yolo] {filename} already exists ({round(size / 1024 / 1024, 2)} MB). Skipping download.")
        sys.exit(0)
    else:
        print(f"[download_yolo] Existing {filename} is corrupted ({size} bytes). Re-downloading...")
        os.remove(filename)

print(f"[download_yolo] Downloading {filename} from {url}...")
try:
    urllib.request.urlretrieve(url, filename)
    if not os.path.exists(filename):
        raise RuntimeError("File does not exist after download completion.")
    size = os.path.getsize(filename)
    if size < MIN_FILE_SIZE:
        raise ValueError(f"Downloaded file is too small ({size} bytes). Download might be incomplete.")
    print(f"[download_yolo] Download complete and verified! Size: {round(size / 1024 / 1024, 2)} MB.")
    sys.exit(0)
except Exception as e:
    print(f"[download_yolo] ERROR: Failed to download or verify {filename}: {e}", file=sys.stderr)
    sys.exit(1)
