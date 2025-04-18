import cv2
import numpy as np
import time

def benchmark_cpu(image, num_iterations=1000):
    start = time.time()
    for _ in range(num_iterations):
        _ = cv2.GaussianBlur(image, (15, 15), 0)
    end = time.time()
    return end - start

def benchmark_gpu(image, num_iterations=1000):
    if not cv2.cuda.getCudaEnabledDeviceCount():
        raise RuntimeError("No CUDA-capable GPU detected or OpenCV not built with CUDA support.")

    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)

    start = time.time()
    for _ in range(num_iterations):
        blurred = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (15, 15), 0)
        gpu_result = blurred.apply(gpu_img)
    end = time.time()
    
    return end - start

def main():
    # Load an image (adjust size for a more demanding task)
    image = cv2.imread('sample2.jpg')
    if image is None:
        print("Failed to load image.")
        return

    print("Starting CPU benchmark...")
    cpu_time = benchmark_cpu(image)
    print(f"CPU Time: {cpu_time:.4f} seconds")

    print("\nStarting GPU benchmark...")
    try:
        gpu_time = benchmark_gpu(image)
        print(f"GPU Time: {gpu_time:.4f} seconds")
    except RuntimeError as e:
        print(f"GPU Benchmark skipped: {e}")

if __name__ == "__main__":
    main()
