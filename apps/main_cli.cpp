#include <iostream>
#include "system.h"
#include "window.h"
#include "rgbd_camera.h"
#include "intrinsic_matrix.h"
#include <signal.h>

bool should_quit = false;

void exit_action(int s)
{
    printf("Caught signal %d\n", s);
    should_quit = true;
}

int main(int argc, char **argv)
{
    fusion::RgbdCamera camera(640, 480, 30);
    fusion::IntrinsicMatrix K(640, 480, 580, 580, 319.5, 239.5);
    fusion::System slam(K, 5);

    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = exit_action;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);

    while (!should_quit)
    {
        if (camera.get_image())
        {
            slam.process_images(camera.depth, camera.image);
        }
    }
}