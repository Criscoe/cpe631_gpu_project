/**
 * Caleb Criscoe
 * Canny OpenCV CUDA
 */

#include "benchmark.hh"
#include <fstream>

#define NUM_ITERATIONS 1000

#define AvgTimeFunction(FUNCTION_, MAP_VALUE_) \
    for (int i = 0; i < NUM_ITERATIONS; ++i)   \
    {                                          \
        MAP_VALUE_ += FUNCTION_;               \
    }

template <typename T>
int printCsv(T &map)
{
    std::ofstream csvFile("output.csv");

    if (!csvFile.is_open())
    {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }
    bool completedFirstRowPrint = false;
    int arrayIter = 0;
    for (auto &pair : map)
    {
        if (!completedFirstRowPrint)
        {
            csvFile << "size";
            for (auto &innerPair : pair.second.typeToDurationMap)
            {
                csvFile << "," << innerPair.first;
            }
            completedFirstRowPrint = true;
        }
        csvFile << std::endl
                << pair.first;
        for (auto innerPair : pair.second.typeToDurationMap)
        {
            csvFile << "," << innerPair.second[arrayIter].count();
        }
        arrayIter++;
    }
    csvFile.close();
    return 0;
}

int main()
{
    struct RecordData
    {
        std::map<const char *, std::array<std::chrono::duration<double>, 20>> typeToDurationMap;
    };
    std::map<const char *, RecordData> map;

    cv::Mat h_img = cv::imread("sample.jpg", cv::IMREAD_GRAYSCALE);
    decltype(std::chrono::high_resolution_clock::now()) start, end;
    if (h_img.empty())
    {
        std::cerr << "Error: Could not read the image file." << std::endl;
        return -1;
    }

    AvgTimeFunction(cudaCanny(h_img), map["sample.jpg"].typeToDurationMap["cuda"][0]);
    AvgTimeFunction(openclCanny(h_img), map["sample.jpg"].typeToDurationMap["opencl"][0]);
    AvgTimeFunction(cpuCanny(h_img), map["sample.jpg"].typeToDurationMap["cpu"][0]);
    AvgTimeFunction(singleThreadedCpuCanny(h_img), map["sample.jpg"].typeToDurationMap["single-threaded-cpu"][0]);

    h_img.release();
    h_img = cv::imread("sample_1920x1080.jpg", cv::IMREAD_GRAYSCALE);
    AvgTimeFunction(cudaCanny(h_img), map["sample_1920x1080.jpg"].typeToDurationMap["cuda"][1]);
    AvgTimeFunction(openclCanny(h_img), map["sample_1920x1080.jpg"].typeToDurationMap["opencl"][1]);
    AvgTimeFunction(cpuCanny(h_img), map["sample_1920x1080.jpg"].typeToDurationMap["cpu"][1]);
    AvgTimeFunction(singleThreadedCpuCanny(h_img), map["sample_1920x1080.jpg"].typeToDurationMap["single-threaded-cpu"][1]);

    h_img.release();
    h_img = cv::imread("sample_80x45.jpg", cv::IMREAD_GRAYSCALE);
    AvgTimeFunction(cudaCanny(h_img), map["sample_80x45.jpg"].typeToDurationMap["cuda"][2]);
    AvgTimeFunction(openclCanny(h_img), map["sample_80x45.jpg"].typeToDurationMap["opencl"][2]);
    AvgTimeFunction(cpuCanny(h_img), map["sample_80x45.jpg"].typeToDurationMap["cpu"][2]);
    AvgTimeFunction(singleThreadedCpuCanny(h_img), map["sample_80x45.jpg"].typeToDurationMap["single-threaded-cpu"][2]);

    return printCsv(map);
}