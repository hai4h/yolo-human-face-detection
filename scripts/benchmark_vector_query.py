import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from os import putenv
import time
import numpy as np
import cv2
from pymongo import MongoClient
from datetime import datetime, timezone
import tensorflow as tf
from tensorflow.keras import layers, Model
from fastapi import HTTPException

# Kết nối MongoDB
global_client = None
def get_db_client(uri="mongodb://localhost:27017/"):
    global global_client
    if global_client is None:
        global_client = MongoClient(uri)
    return global_client

def load_embedding_model(input_shape=(256, 256, 3), embedding_dim=256):
    """Load mô hình embedding khuôn mặt."""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)

    embeddings = layers.Dense(embedding_dim, activation=None, name='embeddings')(x)
    embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(embeddings)
    model = Model(inputs, embeddings, name='embedding_model')
    model.load_weights("models/face_embedding_model_256.h5")
    return model

def get_face_embedding(face_img, model):
    """Trích xuất embedding từ ảnh khuôn mặt."""
    try:
        resized_face = cv2.resize(face_img, (256, 256))
        normalized_face = resized_face.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(normalized_face, axis=0)
        embedding = model.predict(input_tensor, verbose=0)[0].tolist()
        return embedding
    except Exception as e:
        print(f"Error getting face embedding: {e}")
        return None

def measure_latency_and_query(image_path, top_k=5, num_runs=100, mongo_uri="mongodb://localhost:27017/"):
    try:
        client = get_db_client(mongo_uri)
        db = client["face_recognition_db"]
        face_collection = db["faces"]

        model = load_embedding_model()
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
            
        # 1. Lấy vector truy vấn (từ ảnh test)
        q_emb = get_face_embedding(img, model)
        if q_emb is None:
            raise ValueError("Failed to extract embedding.")
        query_vec = np.array(q_emb, dtype=np.float32)

        start = time.time()
        
        # 2. Kéo toàn bộ data từ MongoDB về Python
        all_faces = list(face_collection.find({}, {"name": 1, "face_embedding": 1}))
        
        # 3. Tính Cosine Similarity bằng Numpy (Nhanh và chính xác tuyệt đối)
        results = []
        for face in all_faces:
            db_vec = np.array(face["face_embedding"], dtype=np.float32)
            
            # Kiểm tra xem chiều vector trong DB có khớp với model hiện tại không
            if len(db_vec) != len(query_vec):
                print(f"Bỏ qua '{face['name']}' do sai kích thước: DB có {len(db_vec)} chiều, Model có {len(query_vec)} chiều")
                continue
                
            # Cosine similarity = Dot product (Vì vector đã được L2 Normalized)
            cosine_sim = np.dot(query_vec, db_vec)
            results.append({"name": face.get("name", "Unknown"), "cosineSim": float(cosine_sim)})

        # 4. Sắp xếp kết quả từ cao xuống thấp
        results = sorted(results, key=lambda x: x["cosineSim"], reverse=True)
        top_results = results[:top_k]
        
        query_time = (time.time() - start) * 1000
        
        print(f"\nPython Numpy Query Time: {query_time:.2f} ms")
        if len(top_results) > 0:
            print(f"✅ Best match Name: {top_results[0]['name']}")
            print(f"✅ Cosine Sim: {top_results[0]['cosineSim']:.8f}")
            
            if top_results[0]['cosineSim'] > 0.95:
                print("MATCH SUCCESSFUL! (Similarity > 0.95)")
            else:
                print("NO MATCH. (Different person, or same person but similarity too low)")
        else:
            print("No valid faces found in database to compare.")
            
        return query_time
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    measure_latency_and_query("test_imgs/old/ng-thanh-tung.jpg")
