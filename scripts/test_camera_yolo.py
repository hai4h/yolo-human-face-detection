import os
import time
import cv2
import torch
from ultralytics import YOLO

def main():
    print("=== SYSTEM CHECK ===")
    print(f"PyTorch version: {torch.__version__}")
    
    # KIỂM TRA QUAN TRỌNG: PyTorch có nhận diện được GPU AMD không?
    is_gpu_available = torch.cuda.is_available()
    print(f"GPU (ROCm/CUDA) available: {is_gpu_available}")
    if is_gpu_available:
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("\n[WARNING] PyTorch is using CPU! This is why it takes 10s per frame.")
        print("Check your ROCm installation and PyTorch ROCm version.\n")

    print("\nLoading models...")
    # Khởi tạo mô hình
    person_model = YOLO('./models/yolov8n-person-lw.pt')
    face_model = YOLO('./models/yolov8p-face-v2.pt')

    # Ép mô hình chạy trên GPU nếu có
    if is_gpu_available:
        person_model.to('cuda')
        face_model.to('cuda')

    # Khởi tạo Webcam (0 là camera mặc định, đổi thành 1, 2 nếu dùng camera ngoài)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nStarting camera. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Chạy YOLO
        # Lưu ý: Bật lại half=True nếu có GPU để tăng tốc x2
        use_half = 'fp16' if is_gpu_available else None
        
        person_results = person_model(frame, classes=[0], conf=0.3, imgsz=640, quantize=use_half, verbose=False)
        face_results = face_model(frame, conf=0.3, imgsz=640, quantize=use_half, verbose=False)

        # Vẽ khung Người (Màu Đỏ)
        if person_results and len(person_results[0].boxes) > 0:
            for box in person_results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Vẽ khung Mặt (Màu Xanh lá)
        if face_results and len(face_results[0].boxes) > 0:
            for box in face_results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tính toán thời gian & FPS
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        fps = 1000 / latency if latency > 0 else 0

        # Hiển thị FPS lên màn hình
        cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {latency:.0f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Hiển thị cửa sổ
        cv2.imshow("YOLO Camera Test", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()