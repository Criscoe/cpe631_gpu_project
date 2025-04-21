import cv2
import numpy as np
import time
import argparse
import csv
import cProfile
import re
from random import randint
# from scalene import scalene_profiler


# @profile
def benchmark_cpu(image):
    upper = 50
    lower = 150
    start = time.time()
    _ = cv2.Canny(image, lower, upper)
    end = time.time()
    return end - start

def benchmark_gpuOpenCL(image):
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
    else:
        print("[WARNING] OpenCL is not available on this system.")
        cv2.ocl.setUseOpenCL(False)

    u_image = cv2.UMat(image)

    # Apply Gaussian Blur using OpenCL
    start = time.time()

    _ = cv2.GaussianBlur(u_image, (15, 15), 0)
    end = time.time()
    cv2.ocl.setUseOpenCL(False)
    return end - start

# @profile
def benchmark_gpu(args, image):
    if not cv2.cuda.getCudaEnabledDeviceCount():
        raise RuntimeError("No CUDA-capable GPU detected or OpenCV not built with CUDA support.")
    
    upper = 50
    lower = 150
    # Set up gpu and upload image
    startGpuSetup = time.time()
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)
    endGpuSetup = time.time()
    setupTime = endGpuSetup - startGpuSetup
    # print(f"Time taken for image upload: {setupTime:.4f}")

    if args.enable_img_result:
        canny = cv2.cuda.createCannyEdgeDetector(lower, upper)
        edged = canny.detect(gpu_img)

    start = time.time()
    canny = cv2.cuda.createCannyEdgeDetector(lower, upper)
    edged = canny.detect(gpu_img)
    end = time.time()

    # Download resulting image from gpu
    startGpuDownload = time.time()
    result = edged.download()
    endGpuDownload = time.time()
    downloadTime = endGpuDownload - startGpuDownload
    # print(f"Time taken for image download: {downloadTime:.4f}")

    if args.enable_img_result:
        cv2.imshow("Output", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return end - start

def main():
    parser = argparse.ArgumentParser(description="Python benchmark for Canny edge detection using single/multi threaded CPU and GPU.")
    parser.add_argument("--enable-img-result", action="store_true", help="Enables printing of resulting image")
    parser.add_argument("--num-iterations", type=int, default=1, help="Num iterations to execute calculations on")
    parser.add_argument("--num-cpu-threads", type=int, default=1, help="Number of cpu threads to use")
    parser.add_argument("--benchmark-image", type=str, default='sample.jpg', help="Path and image name of benchmark image used")
    parser.add_argument("--type", type=int, default=0, help="0: Both, 1: CPU, 2: GPU, 3: Optimized")
    args = parser.parse_args()
    # Load an image (adjust size for a more demanding task)

    cv2.setNumThreads(args.num_cpu_threads)
    cpu_time = 0.0
    gpu_time = 0.0
    runOpt = False;

    # if args.type == 0:
    #     runCpu = True
    #     runGpu = True
    # elif args.type == 1:
    #     runCpu = True
    #     runGpu = False
    # elif args.type == 2:
    #     runCpu = False
    #     runGpu = True
    # elif args.type == 3:
    #     runCpu = True
    #     runGpu = True
    #     runOpt = True
    # else:
    #     runCpu = False
    #     runGpu = False

    imageLarge = cv2.imread("../sample_6000x4000.jpg")


    cpuBenchmarkMsg="Starting CPU benchmark..."
    if args.num_cpu_threads == 1:
        cpuBenchmarkMsg="Starting single-threaded CPU benchmark..."
    print(cpuBenchmarkMsg)
    print("Starting GPU benchmark...")

    timeData = []
    grayimg = cv2.cvtColor(imageLarge, cv2.COLOR_BGR2GRAY)
    for i in range(grayimg.shape[1]):
        height, width = grayimg.shape[:2]
        cpu_time = benchmark_cpu(grayimg)
        gpu_timeoOpenCl = benchmark_gpuOpenCL(grayimg)
        try:
            gpu_time = benchmark_gpu(args, grayimg)
        except RuntimeError as e:
            print(f"GPU Benchmark skipped: {e}")        
        
        print(f"CPU Time: {cpu_time:.6f} seconds\n")
        print(f"GPU Time: {gpu_time:.6f} seconds\n")
        print(f"GPU Time (OpenCL): {gpu_timeoOpenCl:.6f} seconds\n")
        print(f"Image Width: {width} Hight: {height}\n")
        timeData.append({'size': (str(width)+'x'+str(height)), 'cpu': cpu_time, 'cuda': gpu_time, 'openCL': gpu_timeoOpenCl})
        if height*.5 >= 1 and width*.5 >= 1: 
            grayimg = cv2.resize(grayimg, (0, 0), fx = 0.5, fy = 0.5)
        else:
            break
    with open('sizeMetrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['size', 'cpu', 'cuda', 'openCL']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timeData)


if __name__ == "__main__":
    main()
