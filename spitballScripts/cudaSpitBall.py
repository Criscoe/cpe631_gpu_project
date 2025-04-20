import cv2
import numpy as np
import time
import argparse
# from scalene import scalene_profiler

# @profile
def benchmark_cpu(image, num_iterations):
    upper = 50
    lower = 150
    start = time.time()
    for _ in range(num_iterations):
        _ = cv2.Canny(image, lower, upper)
    end = time.time()
    return end - start

# @profile
def benchmark_gpu(args, image, num_iterations):
    if not cv2.cuda.getCudaEnabledDeviceCount():
        raise RuntimeError("No CUDA-capable GPU detected or OpenCV not built with CUDA support.")
    
    # Set up gpu and upload image
    startGpuSetup = time.time()
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)
    endGpuSetup = time.time()
    setupTime = endGpuSetup - startGpuSetup
    print(f"Time taken for image upload: {setupTime:.4f}")
    upper = 50
    lower = 150
    if args.enable_img_result:
        canny = cv2.cuda.createCannyEdgeDetector(lower, upper)
        edged = canny.detect(gpu_img)

    start = time.time()
    for _ in range(num_iterations):
        canny = cv2.cuda.createCannyEdgeDetector(lower, upper)
        edged = canny.detect(gpu_img)
    end = time.time()

    # Download resulting image from gpu
    startGpuDownload = time.time()
    result = edged.download()
    endGpuDownload = time.time()
    downloadTime = endGpuDownload - startGpuDownload
    print(f"Time taken for image download: {downloadTime:.4f}")

    if args.enable_img_result:
        cv2.imshow("Output", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    gpu_time = end - start
    print(f"GPU Time: {gpu_time:.4f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Python benchmark for Canny edge detection using single/multi threaded CPU and GPU.")
    parser.add_argument("--enable-img-result", action="store_true", help="Enables printing of resulting image")
    parser.add_argument("--num-iterations", type=int, default=1, help="Num iterations to execute calculations on")
    parser.add_argument("--num-cpu-threads", type=int, default=1, help="Number of cpu threads to use")
    parser.add_argument("--benchmark-image", type=str, default='sample.jpg', help="Path and image name of benchmark image used")
    args = parser.parse_args()
    # Load an image (adjust size for a more demanding task)
    image = cv2.imread(args.benchmark_image)
    if image is None:
        print("Failed to load image.")
        return
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    num_iterations = args.num_iterations # number of repeat calls to be made

    cv2.setNumThreads(args.num_cpu_threads)
    cpuBenchmarkMsg="Starting CPU benchmark..."
    if args.num_cpu_threads == 1:
        cpuBenchmarkMsg="Starting single-threaded CPU benchmark..."
    print(cpuBenchmarkMsg)
    cpu_time = benchmark_cpu(grayimg, num_iterations)
    print(f"CPU Time: {cpu_time:.4f} seconds\n")

    print("Starting GPU benchmark...")
    try:
        benchmark_gpu(args, grayimg, num_iterations)
    except RuntimeError as e:
        print(f"GPU Benchmark skipped: {e}")

if __name__ == "__main__":
    main()
