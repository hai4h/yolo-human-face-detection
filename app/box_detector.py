import numpy as np
import cv2
from .models import load_models, get_face_embedding_model_details
from backend import db_utils

class Detector:
    def __init__(self):
        self.person_model, self.face_model, self.face_embedding_interpreter = load_models()
        
        self.face_embedding_input_details, self.face_embedding_output_details = get_face_embedding_model_details(self.face_embedding_interpreter)
        self.face_embedding_input_shape = self.face_embedding_input_details[0]['shape']
        self.embedding_size = self.face_embedding_input_shape[-1]
        self.emb_model = self.face_embedding_interpreter

    def process_frame(self, frame):
        try:
            # 1. Chạy YOLO
            person_results = self.person_model(frame, classes=[0], conf=0.3, imgsz=640, verbose=False)
            face_results = self.face_model(frame, conf=0.3, imgsz=640, verbose=False)
            
            person_count = 0
            face_count = 0
            person_boxes = []
            face_boxes = []
            
            # Xử lý Người
            if person_results and len(person_results[0].boxes) > 0:
                p_boxes = person_results[0].boxes.xyxy.cpu().numpy()
                p_confs = person_results[0].boxes.conf.cpu().numpy()
                person_count = len(p_boxes)
                
                for box, conf in zip(p_boxes, p_confs):
                    x1, y1, x2, y2 = map(int, box)
                    person_boxes.append(((x1, y1, x2, y2), float(conf)))

            # Xử lý Khuôn mặt
            if face_results and len(face_results[0].boxes) > 0:
                f_boxes = face_results[0].boxes.xyxy.cpu().numpy()
                f_confs = face_results[0].boxes.conf.cpu().numpy()
                
                for box, conf in zip(f_boxes, f_confs):
                    fx1, fy1, fx2, fy2 = map(int, box)
                    
                    if fx1 < fx2 and fy1 < fy2 and fx1 >= 0 and fy1 >= 0 and fx2 <= frame.shape[1] and fy2 <= frame.shape[0]:
                        curr_box = (fx1, fy1, fx2, fy2)
                        face_name = "Không xác định"
                        
                        # CẮT ẢNH VÀ CHẠY TFLITE + FAISS LIÊN TỤC MỖI FRAME
                        face_roi = frame[fy1:fy2, fx1:fx2]
                        if face_roi.size > 0:
                            # 1. Trích xuất Embedding bằng TFLite
                            embedding = self._get_face_embedding(face_roi)
                            if embedding:
                                # 2. Truy vấn thẳng vào FAISS (truyền vào file db_utils)
                                similar = db_utils.find_similar_faces(embedding)
                                if similar:
                                    face_name = similar[0]["name"]
                        
                        # Thêm kết quả khuôn mặt cùng Tên
                        face_boxes.append((curr_box, float(conf), face_name))
                        face_count += 1
            
            return person_count, face_count, person_boxes, face_boxes
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            return 0, 0, [], []
            
    def _get_face_embedding(self, face_img):
        try:
            resized_face = cv2.resize(face_img, (self.face_embedding_input_shape[1], self.face_embedding_input_shape[2]))
            input_dtype = self.face_embedding_input_details[0]['dtype']
            
            if input_dtype == np.float32:
                normalized_face = resized_face.astype(np.float32) / 255.0
            elif input_dtype == np.uint8:
                normalized_face = resized_face.astype(np.uint8)
            else:
                normalized_face = resized_face.astype(np.float32) / 255.0
                
            input_tensor = np.expand_dims(normalized_face, axis=0)
            self.emb_model.set_tensor(self.face_embedding_input_details[0]['index'], input_tensor)
            self.emb_model.invoke()
            embedding = self.emb_model.get_tensor(self.face_embedding_output_details[0]['index'])[0].tolist()
            return embedding
        except Exception as e:
            print(f"Error getting face embedding: {e}")
            return None