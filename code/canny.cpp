/**
 * Caleb Criscoe
 * Canny OpenCV CUDA
 */

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <chrono>

int main()
{
    cv::Mat h_img = cv::imread("sample.jpg", cv::IMREAD_GRAYSCALE);
    if (h_img.empty())
    {
        std::cerr << "Error: Could not read the image file." << std::endl;
        return -1;
    }

    cv::cuda::GpuMat d_img;
    d_img.upload(h_img);

    cv::cuda::GpuMat d_edges;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(100, 200);
        canny->detect(d_img, d_edges);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "GPU Time: " << std::setprecision(15) << elapsed.count() << " seconds" << std::endl;

    cv::Mat h_edges;
    d_edges.download(h_edges);

    cv::imwrite("output_canny_cuda.jpg", h_edges);

    return 0;
}