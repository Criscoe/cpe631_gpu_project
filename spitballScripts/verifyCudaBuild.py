import cv2
print(cv2.cuda.getCudaEnabledDeviceCount()) # Check if CUDA-enabled devices are available
cv2.cuda.printCudaDeviceInfo(0) # Print information about the first CUDA device