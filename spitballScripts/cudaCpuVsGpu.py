import cv2
import numpy as np
import time
import argparse
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


    start = time.time()
    canny = cv2.cuda.createCannyEdgeDetector(lower, upper)
    gpu_result = canny.detect(gpu_img)
    end = time.time()

    # Download resulting image from gpu
    startGpuDownload = time.time()
    result = gpu_result.download()
    endGpuDownload = time.time()
    downloadTime = endGpuDownload - startGpuDownload
    # print(f"Time taken for image download: {downloadTime:.4f}")


    return endGpuDownload - startGpuSetup

def main():
    parser = argparse.ArgumentParser(description="Python benchmark for Canny edge detection using single/multi threaded CPU and GPU.")
    parser.add_argument("--num-iterations", type=int, default=1, help="Num iterations to execute calculations on")
    parser.add_argument("--num-cpu-threads", type=int, default=1, help="Number of cpu threads to use")
    parser.add_argument("--image-path", type=str, default='images', help="Path image folder used")
    parser.add_argument("--type", type=int, default=0, help="0: Both, 1: CPU, 2: GPU, 3: Optimized")
    args = parser.parse_args()
    # Load an image (adjust size for a more demanding task)

    cv2.setNumThreads(args.num_cpu_threads)
    cpu_time = 0.0
    gpu_time = 0.0
    runOpt = False;

    if args.type == 0:
        runCpu = True
        runGpu = True
    elif args.type == 1:
        runCpu = True
        runGpu = False
    elif args.type == 2:
        runCpu = False
        runGpu = True
    elif args.type == 3:
        runCpu = True
        runGpu = True
        runOpt = True
    else:
        runCpu = False
        runGpu = False

    imageSmall = cv2.imread(args.image_path + "/sample_80x45.jpg")
    imageMed = cv2.imread(args.image_path + "/sample_1920x1080.jpg")
    imageLarge = cv2.imread(args.image_path + "/sample.jpg")


    print("Starting benchmark...")
    start = time.time()
    for i in range(args.num_iterations):
        if i % 3 == 0:
            grayimg = cv2.cvtColor(imageSmall, cv2.COLOR_BGR2GRAY)
            if runOpt:
                runCpu = True
                runGpu = False                
        if i % 3 == 1:
            grayimg = cv2.cvtColor(imageMed, cv2.COLOR_BGR2GRAY)
            if runOpt:
                runCpu = False
                runGpu = True  
        if i % 3 == 2:
            grayimg = cv2.cvtColor(imageLarge, cv2.COLOR_BGR2GRAY)
            if runOpt:
                runCpu = False
                runGpu = True

        if runCpu:
            cpu_time += benchmark_cpu(grayimg)
        
        if runGpu:
            try:
                gpu_time += benchmark_gpu(args, grayimg)
            except RuntimeError as e:
                print(f"GPU Benchmark skipped: {e}")        
    end = time.time()

    print(f"CPU Time: {cpu_time:.4f} seconds\n")
    print(f"GPU Time: {gpu_time:.4f} seconds")
    print(f"Overall Time: {end - start:.4f} seconds")


if __name__ == "__main__":
    main()
