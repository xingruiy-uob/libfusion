#ifndef FUSION_VSL_MAIN_WINDOW_H
#define FUSION_VSL_MAIN_WINDOW_H

#include "system_new.h"
#include "macros.h"
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <opencv2/opencv.hpp>

namespace fusion
{

class MainWindow
{
public:
    ~MainWindow();
    MainWindow(
        const char *name = "Untitled",
        const size_t width = 640,
        const size_t height = 480);

    //! Do not copy this class
    MainWindow(const MainWindow &) = delete;
    MainWindow &operator=(const MainWindow &) = delete;

    //! Main loop
    void render();

    enum ImageType
    {
        RGB,
        DEPTH,
        SCENE
    };

    //! External access
    bool is_paused() const;
    void set_image_src(const cv::Mat image, const ImageType type);
    void set_current_camera(const Eigen::Matrix4f &T);
    void set_system(fusion::SystemNew *const sys);
    float *get_vertex_buffer_mapped();
    float *get_normal_buffer_mapped();
    unsigned char *get_colour_buffer_mapped();

    //! Acquire Mesh Functions
    void update_vertex_and_normal();

    size_t VERTEX_COUNT;
    size_t MAX_VERTEX_COUNT;

private:
    std::string window_title;

    void setup_displays();
    void setup_gl_flags();
    void init_textures();
    void init_mesh_buffers();
    void init_glsl_programs();
    void register_key_callbacks();

    //! Check status
    void check_buttons();

    //! Drawing Functions
    void draw_image();
    void draw_depth();
    void draw_scene();
    void draw_camera();
    void draw_mesh_phong_shaded();
    void draw_mesh_colour_mapped();
    void draw_mesh_normal_mapped();

    //! Current Camera Pose
    Eigen::Matrix4f curren_pose;

    //! system ref
    fusion::SystemNew *slam;

    //! key point array
    float *keypoints;
    size_t sizeKeyPoint;
    size_t maxSizeKeyPoint;

    //========================================
    // Visualizations
    //========================================
    //! Displayed Views
    pangolin::View *right_side_view;
    pangolin::View *image_view;
    pangolin::View *depth_view;
    pangolin::View *scene_view;
    pangolin::View *mesh_view;

    //! Displayed textures
    pangolin::GlTexture image_tex;
    pangolin::GlTexture depth_tex;
    pangolin::GlTexture scene_tex;

    //! Main 3D View Camera
    std::shared_ptr<pangolin::OpenGlRenderState> camera_view;

    //! GUI buttons and checkboxes
    std::shared_ptr<pangolin::Var<bool>> BtnReset;
    std::shared_ptr<pangolin::Var<bool>> BtnSaveMap;
    std::shared_ptr<pangolin::Var<bool>> BtnSetLost;
    std::shared_ptr<pangolin::Var<bool>> BtnReadMap;
    std::shared_ptr<pangolin::Var<bool>> BoxPaused;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayImage;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayDepth;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayScene;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayMesh;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayCamera;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayKeyCameras;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayKeyPoint;

    //! Mesh Vertices
    pangolin::GlBufferCudaPtr vertex_buffer;
    pangolin::GlBufferCudaPtr normal_buffer;
    pangolin::GlBufferCudaPtr colour_buffer;

    //! Registered CUDA Ptrs
    std::shared_ptr<pangolin::CudaScopedMappedPtr> vertex_mapped;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> normal_mapped;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> colour_mapped;

    //! GL Shading program
    pangolin::GlSlProgram phong_shader;

    //! Vertex Array Objects
    //! Cannot find a replacement in Pangolin
    GLuint vao_shaded;
    GLuint vao_colour;
};

} // namespace fusion

#endif