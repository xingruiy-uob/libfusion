#ifndef GL_WINDOW_H
#define GL_WINDOW_H

#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

class MainWindow
{
public:
    ~MainWindow();
    MainWindow(const char *name, size_t width, size_t height);
    MainWindow(const MainWindow &) = delete;
    MainWindow &operator=(const MainWindow &) = delete;

    void Render();

    void ResetAllFlags();
    void SetVertexSize(size_t Size);
    void SetRGBSource(cv::Mat RgbImage);
    void SetDepthSource(cv::Mat DepthImage);
    void SetRenderScene(cv::Mat SceneImage);
    void SetCurrentCamera(Eigen::Matrix4f T);

    bool IsPaused();
    bool mbFlagRestart;
    bool mbFlagUpdateMesh;

    float3 *GetMappedVertexBuffer();
    float3 *GetMappedNormalBuffer();
    uchar3 *GetMappedColourBuffer();

    size_t VERTEX_COUNT;
    size_t MAX_VERTEX_COUNT;

private:
    std::string WindowName;

    void SetupDisplays();
    void SetupGLFlags();
    void InitTextures();
    void InitMeshBuffers();
    void InitGlSlPrograms();
    void RegisterKeyCallback();

    pangolin::View *mpViewSideBar;
    pangolin::View *mpViewRGB;
    pangolin::View *mpViewDepth;
    pangolin::View *mpViewScene;
    pangolin::View *mpViewMesh;
    pangolin::View *mpViewMenu;
    pangolin::GlTexture TextureRGB;
    pangolin::GlTexture TextureDepth;
    pangolin::GlTexture TextureScene;

    std::shared_ptr<pangolin::OpenGlRenderState> CameraView;
    std::shared_ptr<pangolin::Var<bool>> BtnReset;
    std::shared_ptr<pangolin::Var<bool>> BoxPaused;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayImage;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayDepth;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayScene;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayMesh;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayCamera;

    void DrawMeshShaded();
    void DrawMeshColoured();
    void DrawMeshNormalMapped();

    pangolin::GlBufferCudaPtr BufferVertex;
    pangolin::GlBufferCudaPtr BufferNormal;
    pangolin::GlBufferCudaPtr BufferColour;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> MappedVertex;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> MappedNormal;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> MappedColour;
    pangolin::GlSlProgram ShadingProg;
    GLuint VAOShade, VAOColour;

    Eigen::Matrix4f CameraPose;
};

#endif