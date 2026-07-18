import faiss
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timezone

# Kết nối MongoDB (vẫn giữ timeout ngắn phòng khi DB sập)
client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=50)
db = client["face_recognition_db"]
face_collection = db["faces"]

# FAISS variables
face_index = None
face_names_map = {}

def init_face_recognition():
    """Tải tất cả embedding từ MongoDB vào FAISS RAM lúc khởi động server."""
    global face_index, face_names_map
    try:
        # Kiểm tra MongoDB có sống không
        client.admin.command('ping')

        # Lấy tất cả khuôn mặt từ DB
        faces = list(face_collection.find({}, {"name": 1, "face_embedding": 1}))

        if not faces:
            print("Cảnh báo: Không có khuôn mặt nào trong cơ sở dữ liệu.")
            return

        # Lấy kích thước vector
        dim = len(faces[0]["face_embedding"])

        # IndexFlatIP dùng để tính Cosine Similarity cho vector đã L2 Normalize
        face_index = faiss.IndexFlatIP(dim)

        embeddings_matrix = []
        for i, face in enumerate(faces):
            embeddings_matrix.append(face["face_embedding"])
            face_names_map[i] = face["name"] # Lưu map ID -> Tên

        # Nạp vào FAISS
        embeddings_matrix = np.array(embeddings_matrix, dtype=np.float32)
        face_index.add(embeddings_matrix)
        print(f"Đã nạp thành công {len(faces)} khuôn mặt vào FAISS index (RAM).")

    except Exception as e:
        print(f"❌ Lỗi khởi tạo FAISS (MongoDB có thể đang tắt): {e}")
        face_index = None
        face_names_map = {}

def find_similar_faces(query_embedding, top_k=1, threshold=0.9):
    """Tìm kiếm bằng FAISS trong RAM (< 1ms) thay vì gọi MongoDB."""
    global face_index, face_names_map

    # Nếu FAISS trống (DB sập hoặc chưa có data), bỏ qua luôn
    if face_index is None or face_index.ntotal == 0:
        return []

    try:
        # Chuyển query về ma trận 2D: shape (1, 256)
        query_vec = np.array([query_embedding], dtype=np.float32)

        # Search trong FAISS
        distances, indices = face_index.search(query_vec, top_k)

        results = []
        for j, i in enumerate(indices[0]):
            if i != -1 and distances[0][j] > threshold:
                results.append({"name": face_names_map[i], "cosineSim": float(distances[0][j])})

        return results
    except Exception as e:
        print(f"FAISS search error: {e}")
        return []

def store_face_data(user_id, name, face_embedding):
    try:
        face_data = {
            "user_id": user_id, "name": name, "face_embedding": face_embedding,
            "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc),
        }
        result = face_collection.insert_one(face_data)
        print(f"Stored face data for user_id: {user_id}, inserted_id: {result.inserted_id}")

        init_face_recognition()
        return True
    except Exception as e:
        print(f"Error storing face data: {e}")
        return False
