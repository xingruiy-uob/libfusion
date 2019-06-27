#include "DescriptorMatcher.h"

namespace fusion
{

DescriptorMatcher::DescriptorMatcher()
{
    l2Matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_SL2);
    hammingMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
}

void DescriptorMatcher::matchHammingKNN(const cv::Mat trainDesc, const cv::Mat queryDesc, std::vector<std::vector<cv::DMatch>> &matches, const int k)
{
    hammingMatcher->knnMatch(queryDesc, trainDesc, matches, k);
}

std::thread DescriptorMatcher::matchHammingKNNAsync(const cv::Mat trainDesc, const cv::Mat queryDesc, std::vector<std::vector<cv::DMatch>> &matches, const int k)
{
    return std::thread(&DescriptorMatcher::matchHammingKNN, this, trainDesc, queryDesc, std::ref(matches), k);
}

void DescriptorMatcher::filter_matches_pair_constraint(
    const std::vector<std::shared_ptr<Point3d>> &src_pts,
    const std::vector<std::shared_ptr<Point3d>> &dst_pts,
    const std::vector<std::vector<cv::DMatch>> &knnMatches,
    std::vector<std::vector<cv::DMatch>> &candidates)
{
    std::vector<cv::DMatch> rawMatch;
    candidates.clear();
    for (const auto &match : knnMatches)
    {
        if (match[0].distance / match[1].distance < 0.6)
        {
            rawMatch.push_back(std::move(match[0]));
        }
        else
        {
            rawMatch.push_back(std::move(match[0]));
            rawMatch.push_back(std::move(match[1]));
        }
    }

    const int NUM_RAW_MATCHES = rawMatch.size();
    cv::Mat adjecencyMat = cv::Mat::zeros(NUM_RAW_MATCHES, NUM_RAW_MATCHES, CV_32FC1);

    for (int y = 0; y < adjecencyMat.rows; ++y)
    {
        float *row = adjecencyMat.ptr<float>(y);
        const auto &match_y = rawMatch[y];
        const auto &match_y_src = src_pts[match_y.queryIdx];
        const auto &match_y_dst = dst_pts[match_y.trainIdx];

        for (int x = 0; x < adjecencyMat.cols; ++x)
        {
            const auto &match_x = rawMatch[x];
            const auto &match_x_src = src_pts[match_x.queryIdx];
            const auto &match_x_dst = dst_pts[match_x.trainIdx];

            if (match_x.trainIdx == match_y.trainIdx || match_x.queryIdx == match_y.queryIdx)
                continue;

            if (x == y)
            {
                row[x] = std::exp(-cv::norm(match_x_src->descriptors, match_x_dst->descriptors, cv::NORM_HAMMING));
            }
            else if (y < x)
            {

                const float src_dist = (match_x_src->pos - match_y_src->pos).norm();
                const float src_angle = std::acos(match_x_src->vec_normal.dot(match_y_src->vec_normal));

                const float dst_dist = (match_x_dst->pos - match_y_dst->pos).norm();
                const float dst_angle = std::acos(match_x_dst->vec_normal.dot(match_y_dst->vec_normal));

                float score = std::exp(-(std::fabs(src_dist - dst_dist) + std::fabs(src_angle - dst_angle)));
                if (std::isnan(score))
                    score = 0;

                row[x] = score;
            }
            else
            {
                row[x] = adjecencyMat.ptr<float>(x)[y];
            }
        }
    }

    cv::Mat reducedAM;
    cv::reduce(adjecencyMat, reducedAM, 0, cv::ReduceTypes::REDUCE_SUM);
    cv::Mat idxMat;
    cv::sortIdx(reducedAM, idxMat, cv::SortFlags::SORT_DESCENDING);

    std::vector<int> idxList;
    for (int y = 0; y < 1; ++y)
    {
        std::vector<cv::DMatch> selectedMatches;
        int head_idx = -1;
        size_t num_selected = 0;
        for (int x = y; x < idxMat.cols; ++x)
        {
            const auto &idx = idxMat.ptr<int>(0)[x];

            if (head_idx < 0)
            {
                head_idx = idx;
                selectedMatches.push_back(rawMatch[idx]);
                num_selected += 1;
            }
            else
            {
                const float &score = adjecencyMat.ptr<float>(head_idx)[idx];
                if (score > 0.1f)
                {
                    selectedMatches.push_back(rawMatch[idx]);
                    num_selected += 1;
                }
            }

            if (num_selected >= 200)
            {
                break;
            }
        }

        candidates.push_back(selectedMatches);
    }
}

void DescriptorMatcher::filter_matches_ratio_test(
    const std::vector<std::vector<cv::DMatch>> &knnMatches,
    std::vector<cv::DMatch> &candidates)
{
    candidates.clear();
    for (const auto &match : knnMatches)
    {
        if (match[0].distance / match[1].distance < 0.8)
        {
            candidates.push_back(std::move(match[0]));
        }
    }
}

} // namespace fusion
