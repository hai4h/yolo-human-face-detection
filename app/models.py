from ultralytics import YOLO
import tensorflow as tf
from .config import PERSON_MODEL_PATH, FACE_MODEL_PATH, EMBED_MODEL_PATH

def load_models():
    """
    Tải mô hình YOLO cho nhận diện người và khuôn mặt,
    và mô hình TensorFlow/Keras cho trích xuất embedding khuôn mặt.
    """
    # Tải mô hình YOLO / Load YOLO models
    person_model = YOLO(PERSON_MODEL_PATH)   
    face_model = YOLO(FACE_MODEL_PATH)       
    
    # Tải mô hình TensorFlow Lite cho trích xuất embedding khuôn mặt
    face_embedding_interpreter = tf.lite.Interpreter(model_path='models/face_embedding_model_256.tflite') 
    face_embedding_interpreter.allocate_tensors()

    return person_model, face_model, face_embedding_interpreter

def get_face_embedding_model_details(interpreter):
    """Lấy thông tin đầu vào và đầu ra của mô hình embedding khuôn mặt"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details