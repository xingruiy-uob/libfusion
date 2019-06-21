#include "feature_extraction.h"

FeatureExtraction::FeatureExtraction()
{
    BRISK = cv::BRISK::create();
    SURF = cv::xfeatures2d::SURF::create();
}

std::thread FeatureExtraction::spawnThread()
{
    return std::thread(&FeatureExtraction::extractFeatures, this);
}

void FeatureExtraction::extractFeatures()
{
    SURF->detect(image, keypoints);
}

void FeatureExtraction::setImage(cv::Mat image)
{
    this->image = image;
}

std::vector<cv::KeyPoint> FeatureExtraction::getKeyPoints() const
{
    return keypoints;
}
