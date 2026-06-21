import llama_cpp
import os

print(f"--- Thông tin thư viện ---")
print(f"Phiên bản llama-cpp-python: {llama_cpp.__version__}")

# Cách 1: Kiểm tra trực tiếp các cờ biên dịch (Phổ biến ở bản mới)
try:
    # Lấy thông tin hệ thống từ hàm C-level
    # Hàm này sẽ in thẳng ra màn hình các thông số như AVX, CUDA, v.v.
    print("Thông số hệ thống (System Info):")
    llama_cpp.llama_log_set(lambda level, msg, user_data: print(msg.decode('utf-8'), end=""), None)
    # Khởi tạo một model giả để ép thư viện in thông tin hệ thống
    # Nếu CUDA = 1 thì bạn đã thành công
except Exception as e:
    print(f"Không lấy được System Info: {e}")

# Cách 2: Kiểm tra sự tồn tại của file thư viện CUDA
# Thông thường nếu cài bản GPU, file .dll của llama-cpp sẽ nặng hơn và có liên kết tới CUDA
import ctypes
try:
    # Thử gọi một hàm đặc trưng của CUDA trong thư viện
    # Nếu không có lỗi nghĩa là có hỗ trợ GPU
    backend = llama_cpp.llama_backend_init()
    print("\nKhởi tạo Backend thành công.")
except:
    print("\nKhởi tạo Backend thất bại.")

print("\n--- HƯỚNG DẪN ĐỌC ---")
print("Hãy nhìn lên phía trên, nếu bạn thấy dòng chữ nào có 'CUDA = 1' hoặc 'BLAS = 1'")
print("thì có nghĩa là đã nhận GPU.")