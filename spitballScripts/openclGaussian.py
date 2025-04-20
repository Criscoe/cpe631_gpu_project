import cv2
import numpy as np
import time

def benchmark_cpu(image, num_iterations=100):
    start = time.time()
    for _ in range(num_iterations):
        _ = cv2.GaussianBlur(image, (15, 15), 0)
    end = time.time()
    return end - start

def benchmark_gpu(image, num_iterations=10000):
    if cv2.ocl.haveOpenCL():
        print("[INFO] OpenCL is available. Enabling OpenCL...")
        cv2.ocl.setUseOpenCL(True)
    else:
        print("[WARNING] OpenCL is not available on this system.")
        cv2.ocl.setUseOpenCL(False)

    u_image = cv2.UMat(image)

    # Apply Gaussian Blur using OpenCL
    start = time.time()
    for _ in range(num_iterations):
        _ = cv2.GaussianBlur(u_image, (15, 15), 0)
    end = time.time()
    
    return end - start

def main():
    # Load an image (adjust size for a more demanding task)
    image = cv2.imread('sample.jpg')
    if image is None:
        print("Failed to load image.")
        return

    # print("Starting CPU benchmark...")
    # cpu_time = benchmark_cpu(image)
    # print(f"CPU Time: {cpu_time:.4f} seconds")

    print("\nStarting GPU benchmark...")
    try:
        gpu_time = benchmark_gpu(image)
        print(f"GPU Time: {gpu_time:.4f} seconds")
    except RuntimeError as e:
        print(f"GPU Benchmark skipped: {e}")

if __name__ == "__main__":
    main()
