#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <Windows.h>

using namespace std;
using namespace cv;

cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eye_cascade;


/**
* Detect user's face and eyes.
*/
int detectEye(cv::Mat& im, cv::Mat& tpl, cv::Rect& rect)
{
    std::vector<cv::Rect> faces, eyes;
    face_cascade.detectMultiScale(im, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    for (int i = 0; i < faces.size(); i++)
    {
        cv::Mat face = im(faces[i]);
        eye_cascade.detectMultiScale(face, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, cv::Size(20, 20));

        if (eyes.size())
        {
            rect = eyes[0] + cv::Point(faces[i].x, faces[i].y);
            tpl = im(rect);
        }
    }

    return eyes.size();
}

void trackEye(cv::Mat& im, cv::Mat& tpl, cv::Rect& rect)
{
    cv::Size size(rect.width * 2, rect.height * 2);
    cv::Rect window(rect + size - cv::Point(size.width / 2, size.height / 2));

    window &= cv::Rect(0, 0, im.cols, im.rows);

    cv::Mat dst(window.width - tpl.rows + 1, window.height - tpl.cols + 1, CV_32FC1);
    cv::matchTemplate(im(window), tpl, dst, TM_SQDIFF_NORMED);

    double minval, maxval;
    cv::Point minloc, maxloc;
    cv::minMaxLoc(dst, &minval, &maxval, &minloc, &maxloc);

    if (minval <= 0.2)
    {
        rect.x = window.x + minloc.x;
        rect.y = window.y + minloc.y;
    }
    else
        rect.x = rect.y = rect.width = rect.height = 0;
}

/**
void MouseMove(int dx, int dy)
{
    INPUT mouse = { 0 };
    mouse.type = INPUT_MOUSE;
    mouse.mi.dwFlags = MOUSEEVENTF_MOVE;
    mouse.mi.dx = dx;
    mouse.mi.dy = dy;

    ::SendInput(1, &mouse, sizeof(INPUT));
}
*/

int main(int argc, char** argv) {

    //MouseMove(100, 0);

    face_cascade.load("haarcascade_frontalface_alt2.xml");
    eye_cascade.load("haarcascade_eye.xml");

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error initializing video camera!" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CAP_PROP_FRAME_HEIGHT, 240);

    String windowName = "Webcam Feed";
    namedWindow(windowName, WINDOW_AUTOSIZE);

    cv::Mat frame, eye_tpl;
    cv::Rect eye_bb;

    while (cv::waitKey(15) != 'q') {

        cap >> frame;
        if (frame.empty())
            break;

        cv::flip(frame, frame, 1);

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if (eye_bb.width == 0 && eye_bb.height == 0) {
            detectEye(gray, eye_tpl, eye_bb);
        }
        else {
            trackEye(gray, eye_tpl, eye_bb);
            cv::rectangle(frame, eye_bb, CV_RGB(0, 255, 0));
        }

        cv::imshow(windowName, frame);
    }
    return 0;
}// end