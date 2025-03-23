#pragma once
#include <pangolin/pangolin.h>
#include <spdlog/spdlog.h>
#include <format>
#include <foundation/utils/thread_value.hpp>
#include <mutex>

namespace foundation {
namespace oak_slam {

class WindowLock {
   public:
    WindowLock(
        std::mutex* mutex,
        std::string& window_name
    )
        : guard(*mutex) {
        pangolin::BindToContext(window_name);
    }

    ~WindowLock() { pangolin::GetBoundWindow()->RemoveCurrent(); }

   private:
    std::lock_guard<std::mutex> guard;
};

struct ImageDisplays {
    pangolin::View* left_display = nullptr;
    pangolin::View* middle_display = nullptr;
    pangolin::View* right_display = nullptr;
    pangolin::GlTexture left_texture;
    pangolin::GlTexture middle_texture;
    pangolin::GlTexture right_texture;

    std::mutex texture_mutex;

    void render() {
        std::scoped_lock l(texture_mutex);
        // Draw the three textures
        left_display->Activate();
        left_texture.RenderToViewportFlipY();

        middle_display->Activate();
        middle_texture.RenderToViewportFlipY();

        right_display->Activate();
        right_texture.RenderToViewportFlipY();
    }

    void update(
        OakFrame frame
    ) {
        std::scoped_lock l(texture_mutex);
        left_texture.Upload(
            frame.left_mono.data, GL_LUMINANCE, GL_UNSIGNED_BYTE
        );
        middle_texture.Upload(
            frame.center_color.data, GL_RGB, GL_UNSIGNED_BYTE
        );
        right_texture.Upload(
            frame.right_mono.data, GL_LUMINANCE, GL_UNSIGNED_BYTE
        );
    }

    void initialize() {
        std::scoped_lock l(texture_mutex);
        // Initialize textures (placeholder size)
        left_texture.Reinitialise(
            640, 400, GL_LUMINANCE, false, 0, GL_RGB, GL_UNSIGNED_BYTE
        );
        middle_texture.Reinitialise(
            640, 400, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE
        );
        right_texture.Reinitialise(
            640, 400, GL_LUMINANCE, false, 0, GL_RGB, GL_UNSIGNED_BYTE
        );
    }
};

/**
 * Separate out the visualizer state to enforce that it is only run in the
 * rendering loop
 */
struct VisualizerState {
    std::mutex window_owner_mutex;

    pangolin::OpenGlRenderState camera_state;
    pangolin::View* scene_display = nullptr;
    std::unique_ptr<pangolin::Handler3D> handler;
    ImageDisplays image_displays;
    std::string window_name;

    WindowLock lock() { return WindowLock(&window_owner_mutex, window_name); }
};

/**
 * Any rendering or updating MUST be done in the rendering thread
 * This is because of some stupid opengl context thing
 */
class Visualizer {
   public:
    Visualizer()
        : should_stop(false) {
        Visualizer::initialize_state(&vis_state);

        runner = std::thread(&Visualizer::run, this);
    }

    void stop() {
        should_stop = true;
        if (runner.joinable()) {
            runner.join();
        }
    }

    void set_current_frame(
        OakFrame& frame
    ) {
        auto lock = vis_state.lock();
        vis_state.image_displays.update(frame);
    }

    ~Visualizer() { stop(); }

   private:
    std::thread runner;
    std::atomic<bool> should_stop;

    ThreadValue<OakFrame> current_frame;
    VisualizerState vis_state;

    /**
     * lock argument required to assure caller holds the lock
     */
    void update(
        const WindowLock& lock
    ) {}

    /**
     * lock argument required to assure caller holds the lock
     */
    void render(
        const WindowLock& lock
    ) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        vis_state.scene_display->Activate(vis_state.camera_state);
        pangolin::glDrawAxis(1.0);

        vis_state.image_displays.render();

        pangolin::FinishFrame();
    }

    void run() {
        spdlog::info("starting visuzlier runner");

        while (true) {
            {
                auto lock = vis_state.lock();
                if (pangolin::ShouldQuit() || should_stop.load()) {
                    break;
                }

                update(lock);
                render(lock);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        spdlog::info("visualizer runner done");
    }

    static void initialize_state(
        VisualizerState* vis_state
    ) {
        vis_state->window_name = "SLAM viz";

        pangolin::CreateWindowAndBind(vis_state->window_name, 1024, 768);
        glEnable(GL_DEPTH_TEST);

        vis_state->camera_state = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(
                1024, 768, 500, 500, 512, 389, 0.1, 1000
            ),
            pangolin::ModelViewLookAt(0, -10, -8, 0, 0, 0, pangolin::AxisY)
        );

        vis_state->handler =
            std::make_unique<pangolin::Handler3D>(vis_state->camera_state);

        // Create 3 image display views
        vis_state->image_displays.left_display =
            &pangolin::CreateDisplay()
                 .SetBounds(0.66, 1.0, 0.0, 0.33)
                 .SetLock(pangolin::LockLeft, pangolin::LockTop);

        vis_state->image_displays.middle_display =
            &pangolin::CreateDisplay()
                 .SetBounds(0.66, 1.0, 0.33, 0.66)
                 .SetLock(pangolin::LockLeft, pangolin::LockTop);

        vis_state->image_displays.right_display =
            &pangolin::CreateDisplay()
                 .SetBounds(0.66, 1.0, 0.66, 1.0)
                 .SetLock(pangolin::LockLeft, pangolin::LockTop);

        vis_state->scene_display =
            &pangolin::CreateDisplay()
                 .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                 .SetHandler(vis_state->handler.get());

        vis_state->image_displays.initialize();
        pangolin::GetBoundWindow()->RemoveCurrent();
    }
};

}  // namespace oak_slam
}  // namespace foundation