import cv2
import numpy as np
import time
import argparse
import csv
import pandas as pd
import matplotlib.pyplot as plt

# from scalene import scalene_profiler


# @profile
def benchmark_cpu(image, num_iterations, useGauss):

    start = time.time()
    for _ in range(num_iterations):
        _ = cv2.GaussianBlur(image, (15, 15), 1.5)
    end = time.time()
    return end - start

def benchmark_gpuOpenCL(image, num_iterations):

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

# @profile
def benchmark_gpu(args, image, num_iterations):
    if not cv2.cuda.getCudaEnabledDeviceCount():
        raise RuntimeError("No CUDA-capable GPU detected or OpenCV not built with CUDA support.")
    
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
    setupTime = endGpuSetup - startGpuSetup
    # print(f"Time taken for image upload: {setupTime:.4f}")

    start = time.time()
    for _ in range(num_iterations):
        gaussBlur = gpu_gaussian.apply(gpu_img)
    end = time.time()

    # Download resulting image from gpu
    startGpuDownload = time.time()
    result = gaussBlur.download()
    endGpuDownload = time.time()
    downloadTime = endGpuDownload - startGpuDownload
    # print(f"Time taken for image download: {downloadTime:.4f}")

    if args.enable_img_result:
        cv2.imshow("Output", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return endGpuDownload - startGpuSetup

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

    imageSmall = cv2.imread("../sample_80x45.jpg")
    imageMed = cv2.imread("../sample_1920x1080.jpg")
    imageLarge = cv2.imread("../sample.jpg")



    cpuBenchmarkMsg="Starting CPU benchmark..."
    if args.num_cpu_threads == 1:
        cpuBenchmarkMsg="Starting single-threaded CPU benchmark..."
    print(cpuBenchmarkMsg)
    print("Starting GPU benchmark...")
    n = 1000

    timeData = []
    images = [cv2.cvtColor(imageSmall, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageMed, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageLarge, cv2.COLOR_BGR2GRAY)]
    for i in range(3):
        height, width = images[i].shape[:2]
        cpu_time = benchmark_cpu(images[i], n)
        gpu_timeoOpenCl = benchmark_gpuOpenCL(images[i], n)
        try:
            gpu_time = benchmark_gpu(args, images[i], n)
        except RuntimeError as e:
            print(f"GPU Benchmark skipped: {e}")        
        
        print(f"CPU Time: {cpu_time:.6f} seconds\n")
        print(f"GPU Time: {gpu_time:.6f} seconds\n")
        print(f"GPU Time (OpenCL): {gpu_timeoOpenCl:.6f} seconds\n")
        timeData.append({'size': (str(width)+'x'+str(height)), 'cpu': cpu_time, 'cuda': gpu_time, 'openCL': gpu_timeoOpenCl})

    runTimestamp = str(int(time.time()))
    testName = 'IterationsTestGauss'

    with open("data/" + runTimestamp + "_" + testName + "_n" + str(n) + "_thr" + str(args.num_cpu_threads) +".csv", 'w', newline='') as csvfile:
        fieldnames = ['size', 'cpu', 'cuda', 'openCL']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timeData)

    df = pd.DataFrame(timeData)
    plt.plot(df['size'], df['cpu'], label="cpu")
    plt.plot(df['size'], df["cuda"], label="cuda")
    plt.plot(df['size'], df["openCL"], label="openCL")
    plt.legend()
    plt.title("Execution time (seconds) vs Size (w x h pixels) for "+ str(n) + " iterations")
    plt.xlabel('size')
    plt.ylabel('Execution Time')
    plotName = "plots/" + runTimestamp + "_" + testName + "_n" + str(n) + "_thr" + str(args.num_cpu_threads) +".jpg"
    plt.savefig(plotName)

    plt.yscale('log')
    plt.savefig("plots/" + runTimestamp + "_" + testName + "_n" + str(n) + "_thr" + str(args.num_cpu_threads) +"_logScale" + ".jpg")

if __name__ == "__main__":
    main()
