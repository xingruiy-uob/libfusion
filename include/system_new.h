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
    ~SystemNew();

    void spawn_work(const cv::Mat &depth, const cv::Mat &image);
    void reset();
    void write_map_to_disk(const std::string) const;
    void read_map_from_disk(const std::string);

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
};

} // namespace fusion

#endif