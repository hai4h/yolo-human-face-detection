import time
import numpy as np
import faiss

def generate_normalized_vectors(num_vectors, dim):
    """
    Tạo các vector ngẫu nhiên giả lập face_embedding.
    FAISS sử dụng Inner Product (IndexFlatIP) kết hợp L2 Normalize để tính Cosine Similarity.
    """
    # Dùng randn (phân bố chuẩn) thay vì rand để vector tỏa đều trong không gian nhiều chiều
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    # L2 Normalize (bắt buộc để Inner Product = Cosine Similarity)
    faiss.normalize_L2(vectors)
    return vectors

def benchmark_faiss(dim=512, num_queries=1000): # Dim thường là 128 hoặc 512 với Face Recognition
    print("="*85)
    print(f"FAISS CPU BENCHMARK - BRUTE FORCE (IndexFlatIP)")
    print(f"Vector Dimension: {dim} | Số lượng truy vấn (Queries): {num_queries}")
    print("="*85)
    print(f"{'Số lượng khuôn mặt (DB)':<25} | {'1-by-1 (Realtime)':<20} | {'Batch (Xử lý cụm)':<20} | {'QPS (Batch)'}")
    print("-" * 85)

    # Các mốc số lượng khuôn mặt trong Database (Thêm mốc 1 triệu)
    db_sizes = [100, 1000, 10_000, 100_000, 1_000_000]

    # Sinh sẵn tập các vector truy vấn khác nhau để tránh CPU Caching
    query_vectors = generate_normalized_vectors(num_queries, dim)

    for size in db_sizes:
        # 1. Sinh dữ liệu giả lập (Mock Database)
        db_vectors = generate_normalized_vectors(size, dim)

        # 2. Khởi tạo FAISS Index Brute-force
        index = faiss.IndexFlatIP(dim)
        index.add(db_vectors)

        # ==========================================
        # Bài test 1: Sequential Search (1-by-1)
        # Mô phỏng: Camera nhận dạng từng frame nối tiếp nhau
        # ==========================================
        start_time_seq = time.perf_counter()
        for i in range(num_queries):
            # Lấy từng vector ra search, giữ nguyên shape (1, dim)
            query = query_vectors[i:i+1]
            distances, indices = index.search(query, k=5)
        end_time_seq = time.perf_counter()
        
        avg_seq_latency = ((end_time_seq - start_time_seq) * 1000) / num_queries

        # ==========================================
        # Bài test 2: Batch Search
        # Mô phỏng: Đưa 1 bức ảnh chụp đám đông có nhiều khuôn mặt vào cùng lúc
        # FAISS thực hiện nhân ma trận cực nhanh ở chế độ này
        # ==========================================
        start_time_batch = time.perf_counter()
        
        # Ném toàn bộ query_vectors vào search cùng 1 lúc
        distances_batch, indices_batch = index.search(query_vectors, k=5)
        
        end_time_batch = time.perf_counter()
        
        avg_batch_latency = ((end_time_batch - start_time_batch) * 1000) / num_queries
        qps = num_queries / (end_time_batch - start_time_batch) # Queries Per Second

        # In kết quả
        print(f"{size:<25,} | {avg_seq_latency:.4f} ms/query  | {avg_batch_latency:.4f} ms/query  | {qps:,.0f} Q/s")

    print("="*85)

if __name__ == "__main__":
    # Chỉnh dim=512 (Chuẩn ArcFace/InsightFace phổ biến)
    benchmark_faiss()
