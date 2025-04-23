import cv2
import numpy as np
import time
import argparse
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scalene import scalene_profiler


@profile
def benchmark_cpu_Canny(image, num_iterations):
    upper = 50
    lower = 150
    start = time.time()
    for _ in range(num_iterations):
        _ = cv2.Canny(image, lower, upper)
    end = time.time()
    return end - start

@profile
def benchmark_cpu_Gauss(image, num_iterations):
    start = time.time()
    for _ in range(num_iterations):
        _ = cv2.GaussianBlur(image, (15, 15), 1.5)
    end = time.time()
    return end - start

@profile
def benchmark_gpuOpenCL_Canny(image, num_iterations):
    upper = 50
    lower = 150
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
    else:
        print("[WARNING] OpenCL is not available on this system.")
        cv2.ocl.setUseOpenCL(False)

    u_image = cv2.UMat(image)

    # Apply Gaussian Blur using OpenCL
    start = time.time()
    for _ in range(num_iterations):
        _ = cv2.Canny(u_image, lower, upper)
    end = time.time()

    cv2.ocl.setUseOpenCL(False)
    return end - start

@profile
def benchmark_gpuOpenCL_Gauss(image, num_iterations):
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
    else:
        print("[WARNING] OpenCL is not available on this system.")
        cv2.ocl.setUseOpenCL(False)

    u_image = cv2.UMat(image)

    # Apply Gaussian Blur using OpenCL
    start = time.time()
    for _ in range(num_iterations):
        _ = cv2.GaussianBlur(u_image, (15, 15), 1.5)
    end = time.time()

    cv2.ocl.setUseOpenCL(False)
    return end - start

@profile
def benchmark_gpu_Canny(image, num_iterations):
    if not cv2.cuda.getCudaEnabledDeviceCount():
        raise RuntimeError("No CUDA-capable GPU detected or OpenCV not built with CUDA support.")
    gpuTime = {}
    # Set up gpu and upload image
    startGpuSetup = time.time()
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)
    endGpuSetup = time.time()
    gpuTime['setup'] = endGpuSetup - startGpuSetup
    # print(f"Time taken for image upload: {setupTime:.4f}")
    upper = 50
    lower = 150

    start = time.time()
    for _ in range(num_iterations):
        canny = cv2.cuda.createCannyEdgeDetector(lower, upper)
        gpu_result = canny.detect(gpu_img)
    end = time.time()

    # Download resulting image from gpu
    startGpuDownload = time.time()
    result = gpu_result.download()
    endGpuDownload = time.time()
    gpuTime['downloadTime'] = endGpuDownload - startGpuDownload
    # print(f"Time taken for image download: {downloadTime:.4f}")
    
    # cv2.imwrite((str(int(time.time())) + ".jpg"), result)
    return (endGpuDownload - startGpuSetup) , gpuTime

@profile
def benchmark_gpu_gauss(image, num_iterations):
    if not cv2.cuda.getCudaEnabledDeviceCount():
        raise RuntimeError("No CUDA-capable GPU detected or OpenCV not built with CUDA support.")
    gpuTime = {}
    # Set up gpu and upload image
    startGpuSetup = time.time()
    gpu_gaussian = cv2.cuda.createGaussianFilter(
        cv2.CV_8UC1,  # Source image type (8-bit unsigned, single channel)
        cv2.CV_8UC1,  # Destination image type
        (15, 15),      # Kernel size (15x15)
        1.5         # Sigma value (standard deviation)
        )
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)
    endGpuSetup = time.time()
    gpuTime['setup'] = endGpuSetup - startGpuSetup
    # print(f"Time taken for image upload: {setupTime:.4f}")

    start = time.time()
    for _ in range(num_iterations):
        gpu_result = gpu_gaussian.apply(gpu_img)

    end = time.time()

    # Download resulting image from gpu
    startGpuDownload = time.time()
    result = gpu_result.download()
    endGpuDownload = time.time()
    gpuTime['downloadTime'] = endGpuDownload - startGpuDownload
    # print(f"Time taken for image download: {downloadTime:.4f}")
    
    # cv2.imwrite((str(int(time.time())) + ".jpg"), result)
    return (endGpuDownload - startGpuSetup) , gpuTime

def main():
    parser = argparse.ArgumentParser(description="Python benchmark for Canny edge detection using single/multi threaded CPU and GPU.")
    parser.add_argument("--num-iterations", type=int, default=1000, help="Num iterations to execute calculations on")
    parser.add_argument("--num-cpu-threads", type=int, default=16, help="Number of cpu threads to use")
    parser.add_argument("--image-path", type=str, default='images', help="Path image folder used")
    parser.add_argument("--gauss", action="store_true", help="Use gauss algorithm")
    args = parser.parse_args()

    cv2.setNumThreads(args.num_cpu_threads)
    cpu_time = 0.0
    gpu_time = 0.0

    imageSmall = cv2.imread(args.image_path + "/sample_80x45.jpg")
    imageMed = cv2.imread(args.image_path + "/sample_1920x1080.jpg")
    imageLarge = cv2.imread(args.image_path + "/sample.jpg")

    cpuBenchmarkMsg="Starting Iterations Test benchmark..."
    # print(cpuBenchmarkMsg)
    n = args.num_iterations
    timeData = []
    gpu_timeData = []

    images = [cv2.cvtColor(imageSmall, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageMed, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageLarge, cv2.COLOR_BGR2GRAY)]
    for gsImage in images:
        height, width = gsImage.shape[:2]
        if args.gauss:
            cpu_time = benchmark_cpu_Gauss(gsImage, n)
            gpu_timeoOpenCl = benchmark_gpuOpenCL_Gauss(gsImage, n)
            try:
                gpu_time, tmpTime = benchmark_gpu_gauss(gsImage, n)
                gpu_timeData.append(tmpTime)
            except RuntimeError as e:
                print(f"GPU Benchmark skipped: {e}")        
        else:
            cpu_time = benchmark_cpu_Canny(gsImage, n)
            gpu_timeoOpenCl = benchmark_gpuOpenCL_Canny(gsImage, n)
            try:
                gpu_time, tmpTime = benchmark_gpu_Canny(gsImage, n)
                gpu_timeData.append(tmpTime)
            except RuntimeError as e:
                print(f"GPU Benchmark skipped: {e}")        
        
        # print(f"CPU Time: {cpu_time:.6f} seconds\n")
        # print(f"GPU Time: {gpu_time:.6f} seconds\n")
        # print(f"GPU Time (OpenCL): {gpu_timeoOpenCl:.6f} seconds\n")
        timeData.append({'size': (str(width)+'x'+str(height)), 'cpu': cpu_time, 'cuda': gpu_time, 'openCL': gpu_timeoOpenCl})

    runTimestamp = str(int(time.time()))
    if args.gauss:
        testName = 'IterationsTestGauss'
    else:
        testName = 'IterationsTestCanny'
        

    with open("data/" + runTimestamp + "_" + testName + "_n" + str(n) + "_thr" + str(args.num_cpu_threads) +".csv", 'w', newline='') as csvfile:
        fieldnames = ['size', 'cpu', 'cuda', 'openCL']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timeData)

    with open("data/" + str(runTimestamp) + "_" + testName + "_n" + str(n) + "_thr" + str(args.num_cpu_threads) +"CPU.csv", 'w', newline='') as csvfile:
        fieldnames = ['setup', 'downloadTime' ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(gpu_timeData)

    df = pd.DataFrame(timeData)
    plt.figure(figsize = (20,8))
    plt.plot(df['size'], df['cpu'], label="cpu")
    plt.plot(df['size'], df["cuda"], label="cuda")
    plt.plot(df['size'], df["openCL"], label="openCL")
    plt.legend()
    plt.title("Execution time (seconds) vs Size (w x h pixes) for "+ str(n) + " iterations")
    plt.xlabel('size')
    plt.ylabel('Execution Time')
    plotName = "plots/" + str(runTimestamp) + "_" + testName + "_n" + str(n) + "_thr" + str(args.num_cpu_threads) +".jpg"
    plt.savefig(plotName)

    plt.yscale('log')
    plt.savefig("plots/" +str(runTimestamp) + "_" + testName + "_n" + str(n) + "_thr" + str(args.num_cpu_threads) +"_logScale.jpg")



if __name__ == "__main__":
    main()
