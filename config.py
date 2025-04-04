from os import putenv

# Cấu hình môi trường cho AMD GPU / Environment setup for AMD GPUs
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

# Đường dẫn mô hình / Model paths
PERSON_MODEL_PATH = 'yolo11n.pt'
FACE_MODEL_PATH = 'yolo8n-face-640-50epochs.pt'

# Thiết lập hiển thị / Visualization settings
PERSON_COLOR = (0, 0, 255)  # Màu đỏ / Red color
FACE_COLOR = (0, 255, 0)     # Màu xanh lá / Green color
TEXT_SCALE = 0.6             # Tỷ lệ chữ / Text scale
TEXT_THICKNESS = 2           # Độ dày chữ / Text thickness