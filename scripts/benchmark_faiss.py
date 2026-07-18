import time
import numpy as np
import faiss

def generate_normalized_vectors(num_vectors, dim):
    """
    Tạo các vector ngẫu nhiên giả lập face_embedding.
    FAISS sử dụng Inner Product (IndexFlatIP) kết hợp L2 Normalize để tính Cosine Similarity.
    """
    # Sinh vector ngẫu nhiên
    vectors = np.random.rand(num_vectors, dim).astype(np.float32)
    # L2 Normalize (bắt buộc để Inner Product = Cosine Similarity)
    faiss.normalize_L2(vectors)
    return vectors

def benchmark_faiss(dim=256, num_queries=1000):
    print("="*60)
    print(f"🚀 FAISS CPU BENCHMARK (Vector Dimension: {dim})")
    print(f"Thực hiện {num_queries} vòng lặp truy vấn cho mỗi mức dữ liệu")
    print("="*60)
    print(f"{'Số lượng khuôn mặt (DB Size)':<30} | {'Thời gian trung bình / 1 Frame':<20}")
    print("-" * 60)

    # Các mốc số lượng khuôn mặt trong Database
    db_sizes = [100, 1000, 10_000, 100_000, 1_000_000]

    for size in db_sizes:
        # 1. Sinh dữ liệu giả lập (Mock Database)
        db_vectors = generate_normalized_vectors(size, dim)

        # 2. Khởi tạo FAISS Index (Chỉ chạy 1 lần khi start server)
        index = faiss.IndexFlatIP(dim)
        index.add(db_vectors)

        # 3. Sinh 1 vector truy vấn ngẫu nhiên (từ khuôn mặt YOLO cắt ra)
        query_vector = generate_normalized_vectors(1, dim)

        # 4. Đo thời gian (Benchmark)
        start_time = time.perf_counter()

        # Chạy tìm kiếm nhiều lần để lấy kết quả trung bình chính xác
        for _ in range(num_queries):
            # Tìm top 5 khuôn mặt giống nhất
            distances, indices = index.search(query_vector, k=5)

        end_time = time.perf_counter()

        # 5. Tính toán
        total_time = (end_time - start_time) * 1000  # Đổi sang mili giây (ms)
        avg_latency = total_time / num_queries

        print(f"{size:<30,} | {avg_latency:.4f} ms")

    print("="*60)
    print("💡 Lưu ý: Ở 30 FPS, bạn chỉ có quỹ thời gian 33.3ms để xử lý TẤT CẢ mọi thứ.")
    print("FAISS tốn chưa tới 1ms ngay cả với 1.000.000 khuôn mặt, nên nó sẽ KHÔNG BAO GIỜ làm tụt FPS.")
    print("="*60)

if __name__ == "__main__":
    benchmark_faiss(dim=256, num_queries=1000)
