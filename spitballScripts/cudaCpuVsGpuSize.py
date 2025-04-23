import cv2
import numpy as np
import time
import argparse
import csv
import pandas as pd
import matplotlib.pyplot as plt
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

    edged = cv2.Canny(u_image, lower, upper)
    end = time.time()
    edged.isContinuous()
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

    start = time.time()
    canny = cv2.cuda.createCannyEdgeDetector(lower, upper)
    gpu_restlt = canny.detect(gpu_img)
    end = time.time()

    # Download resulting image from gpu
    startGpuDownload = time.time()
    result = gpu_restlt.download()
    endGpuDownload = time.time()
    downloadTime = endGpuDownload - startGpuDownload
    # print(f"Time taken for image download: {downloadTime:.4f}")
    
    return endGpuDownload - startGpuSetup

def main():
    parser = argparse.ArgumentParser(description="Python benchmark for Canny edge detection using single/multi threaded CPU and GPU.")
    parser.add_argument("--num-cpu-threads", type=int, default=16, help="Number of cpu threads to use")
    parser.add_argument("--image-path", type=str, default='images', help="Path image folder used")
    parser.add_argument("--rate", type=float, default=0.5, help="Path image folder used")
    args = parser.parse_args()

    # Load an image (adjust size for a more demanding task)

    cv2.setNumThreads(args.num_cpu_threads)
    cpu_time = 0.0
    gpu_time = 0.0

    imageLarge = cv2.imread(args.image_path + "/sample_6000x4000.jpg")

    print("Starting Size benchmark...")

    timeData = []
    grayimg = cv2.cvtColor(imageLarge, cv2.COLOR_BGR2GRAY)
    while grayimg.shape[1] > 0:
        height, width = grayimg.shape[:2]
        cpu_time = benchmark_cpu(grayimg)
        gpu_timeoOpenCl = benchmark_gpuOpenCL(grayimg)
        try:
            gpu_time = benchmark_gpu(args, grayimg)
        except RuntimeError as e:
            print(f"GPU Benchmark skipped: {e}")        
        
        # print(f"CPU Time: {cpu_time:.6f} seconds\n")
        # print(f"GPU Time: {gpu_time:.6f} seconds\n")
        # print(f"GPU Time (OpenCL): {gpu_timeoOpenCl:.6f} seconds\n")
        # print(f"Image Width: {width} Hight: {height}\n")
        timeData.append({'size': (str(width)+'x'+str(height)), 'cpu': cpu_time, 'cuda': gpu_time, 'openCL': gpu_timeoOpenCl})
        if height*args.rate >= 1 and width*args.rate >= 1: 
            grayimg = cv2.resize(grayimg, (0, 0), fx = args.rate, fy = args.rate)
        else:
            break

    runTimestamp = str(int(time.time()))
    testName = 'ImageSize'
    rate = str(args.rate).replace('.', '_')

    with open("data/" + runTimestamp + "_" + testName + "_rate" + rate + "_thr" + str(args.num_cpu_threads) +".csv", 'w', newline='') as csvfile:
        fieldnames = ['size', 'cpu', 'cuda', 'openCL']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timeData)

    df = pd.DataFrame(timeData)
    plt.figure(figsize = (20,8))
    plt.plot(df['size'], df['cpu'], label="cpu")
    plt.plot(df['size'], df["cuda"], label="cuda")
    plt.plot(df['size'], df["openCL"], label="openCL")
    plt.legend()
    plt.title("Execution time (seconds) vs Size (w x h pixels)")
    plt.xlabel('size')
    plt.ylabel('Execution Time')
    plotName = "plots/" + str(runTimestamp) + "_" + testName + "_rate" + rate + "_thr" + str(args.num_cpu_threads) +".jpg"
    plt.savefig(plotName)

    plt.yscale('log')
    plt.savefig("plots/" +str(runTimestamp) + "_" + testName + "_rate" + rate + "_thr" + str(args.num_cpu_threads) +"_logScale.jpg")

if __name__ == "__main__":
    main()
