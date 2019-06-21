#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>

class FeatureExtraction
{
public:
    FeatureExtraction();
    std::thread spawnThread(cv::Mat image);
    void extractFeatures(cv::Mat image);
    std::vector<cv::KeyPoint> getKeyPoints() const;

private:
    std::vector<cv::KeyPoint> keypoints;
    cv::Ptr<cv::BRISK> BRISK;
    cv::Ptr<cv::xfeatures2d::SURF> SURF;
};

#endif