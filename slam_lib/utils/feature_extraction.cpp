#include "feature_extraction.h"

FeatureExtraction::FeatureExtraction()
{
    BRISK = cv::BRISK::create();
    SURF = cv::xfeatures2d::SURF::create();
}

std::thread FeatureExtraction::spawnThread(cv::Mat image)
{
    return std::thread(&FeatureExtraction::extractFeatures, this, image);
}

void FeatureExtraction::extractFeatures(cv::Mat image)
{
    SURF->detect(image, keypoints);
}

std::vector<cv::KeyPoint> FeatureExtraction::getKeyPoints() const
{
    return keypoints;
}
