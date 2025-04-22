/**
 * Header
 */

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/cuda.hpp>
#include <chrono>

#define CANNY_LOW 50
#define CANNY_HIGH 150
#define TimeFunction(FNC_)                                  \
    auto start = std::chrono::high_resolution_clock::now(); \
    FNC_;                                                   \
    auto end = std::chrono::high_resolution_clock::now()

#define RetVal return end - start;

template <typename T, typename... T1>
static inline auto cudaCanny(T image, T1... args)
{
    cv::cuda::GpuMat d_img;
    d_img.upload(image);

    cv::cuda::GpuMat d_edges;
    // cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(CANNY_LOW, CANNY_HIGH);
    TimeFunction(cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(CANNY_LOW, CANNY_HIGH); canny->detect(d_img, d_edges));

    cv::Mat h_edges;
    d_edges.download(h_edges);
    RetVal
}

template <typename T, typename... T1>
static inline auto openclCanny(T image, T1... args)
{
    cv::ocl::setUseOpenCL(true);
    cv::UMat ugray, edges;
    image.copyTo(ugray);
    TimeFunction(cv::Canny(ugray, edges, CANNY_LOW, CANNY_HIGH));
    cv::Mat h_edges = edges.getMat(cv::ACCESS_READ);
    cv::ocl::setUseOpenCL(false);
    RetVal
}

template <typename T, typename... T1>
static inline auto cpuCanny(T image, T1... args)
{
    cv::Mat edges;
    TimeFunction(cv::Canny(image, edges, CANNY_LOW, CANNY_HIGH));
    RetVal
}

template <typename T, typename... T1>
static inline auto singleThreadedCpuCanny(T image, T1... args)
{
    auto initThreads = cv::getNumThreads();
    cv::setNumThreads(1);
    cv::Mat edges;
    TimeFunction(cv::Canny(image, edges, CANNY_LOW, CANNY_HIGH));
    cv::setNumThreads(initThreads);
    RetVal
}