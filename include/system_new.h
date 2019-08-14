#ifndef SYSTEM_NEW_H
#define SYSTEM_NEW_H

#include "data_struct/rgbd_frame.h"
#include "data_struct/intrinsic_matrix.h"
#include "tracking/icp_tracker.h"
#include "voxel_hashing/voxel_hashing.h"

namespace fusion
{

class SystemNew
{
public:
    SystemNew(const IntrinsicMatrix K, const int NUM_PYR);

    void spawn_work(const cv::Mat &depth, const cv::Mat &image);
    void reset();
    bool get_rendered_scene(cv::Mat &scene);
    bool get_rendered_depth(cv::Mat &depth);
    Eigen::Matrix3f get_intrinsics() const;
    Eigen::Matrix4f get_current_pose() const;
    void write_map_to_disk(const std::string) const;
    void read_map_from_disk(const std::string);
    size_t fetch_mesh_with_normal(float *vertex, float *normal);

private:
    SystemNew(const SystemNew &) = delete;

    std::shared_ptr<RgbdFrame> current;
    std::shared_ptr<RgbdFrame> last_tracked;
    std::shared_ptr<RgbdFrame> keyframe;

    void create_keyframe();
    bool check_keyframe_critera() const;

    std::shared_ptr<DenseMapping> mapper;
    std::shared_ptr<DenseTracking> tracker;

    bool initialized;
    size_t current_frame_id;

    //! Temporary variables
    //! Do NOT reference this
    cv::cuda::GpuMat vmap_cast;
    cv::cuda::GpuMat image_cast;

    Sophus::SE3d current_pose;

    void populate_current_data(cv::Mat depth, cv::Mat image);

    Eigen::Matrix3f K;
    bool has_new_keyframe;
};

} // namespace fusion

#endif