#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace std;
using namespace cv;

cv::CascadeClassifier faceCascade;
cv::CascadeClassifier eyeCascade;

cv::Vec3f getEyeball(cv::Mat& eye, std::vector<cv::Vec3f>& circles)
{
    std::vector<int> sums(circles.size(), 0);
    for (int y = 0; y < eye.rows; y++)
    {
        uchar* ptr = eye.ptr<uchar>(y);
        for (int x = 0; x < eye.cols; x++)
        {
            int value = static_cast<int>(*ptr);
            for (int i = 0; i < circles.size(); i++)
            {
                cv::Point center((int)std::round(circles[i][0]), (int)std::round(circles[i][1]));
                int radius = (int)std::round(circles[i][2]);
                if (std::pow(x - center.x, 2) + std::pow(y - center.y, 2) < std::pow(radius, 2))
                {
                    sums[i] += value;
                }
            }
            ++ptr;
        }
    }
    int smallestSum = 9999999;
    int smallestSumIndex = -1;
    for (int i = 0; i < circles.size(); i++)
    {
        if (sums[i] < smallestSum)
        {
            smallestSum = sums[i];
            smallestSumIndex = i;
        }
    }
    return circles[smallestSumIndex];
}

cv::Rect getLeftmostEye(std::vector<cv::Rect>& eyes)
{
    int leftmost = 99999999;
    int leftmostIndex = -1;
    for (int i = 0; i < eyes.size(); i++)
    {
        if (eyes[i].tl().x < leftmost)
        {
            leftmost = eyes[i].tl().x;
            leftmostIndex = i;
        }
    }
    return eyes[leftmostIndex];
}

std::vector<cv::Point> centers;
cv::Point lastPoint;
cv::Point mousePoint;

cv::Point stabilize(std::vector<cv::Point>& points, int windowSize)
{
    float sumX = 0;
    float sumY = 0;
    int count = 0;
    for (int i = std::max(0, (int)(points.size() - windowSize)); i < points.size(); i++)
    {
        sumX += points[i].x;
        sumY += points[i].y;
        ++count;
    }
    if (count > 0)
    {
        sumX /= count;
        sumY /= count;
    }
    return cv::Point(sumX, sumY);
}

void detectEyes(cv::Mat& frame, cv::CascadeClassifier& faceCascade, cv::CascadeClassifier& eyeCascade)
{
    cv::Mat grayscale;
    cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY); // convert image to grayscale
    cv::equalizeHist(grayscale, grayscale); // enhance image contrast 
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(grayscale, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(150, 150));
    if (faces.size() == 0) return; // none face was detected
    cv::Mat face = grayscale(faces[0]); // crop the face
    std::vector<cv::Rect> eyes;
    eyeCascade.detectMultiScale(face, eyes, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30)); // same thing as above    
    rectangle(frame, faces[0].tl(), faces[0].br(), cv::Scalar(255, 0, 0), 2);
    if (eyes.size() != 2) return; // both eyes were not detected
    for (cv::Rect& eye : eyes)
    {
        rectangle(frame, faces[0].tl() + eye.tl(), faces[0].tl() + eye.br(), cv::Scalar(0, 255, 0), 2);
    }
    cv::Rect eyeRect = getLeftmostEye(eyes);
    cv::Mat eye = face(eyeRect); // crop the leftmost eye
    cv::equalizeHist(eye, eye);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(eye, circles, cv::HOUGH_GRADIENT, 1, eye.cols / 8, 250, 15, eye.rows / 8, eye.rows / 3);
    if (circles.size() > 0)
    {
        cv::Vec3f eyeball = getEyeball(eye, circles);
        cv::Point center(eyeball[0], eyeball[1]);
        centers.push_back(center);
        center = stabilize(centers, 5);
        if (centers.size() > 1)
        {
            cv::Point diff;
            diff.x = (center.x - lastPoint.x) * 20;
            diff.y = (center.y - lastPoint.y) * -30;
            mousePoint += diff;
        }
        lastPoint = center;
        int radius = (int)eyeball[2];
        cv::circle(frame, faces[0].tl() + eyeRect.tl() + center, radius, cv::Scalar(0, 0, 255), 2);
        cv::circle(eye, center, radius, cv::Scalar(255, 255, 255), 2);
    }
    cv::imshow("Eye", eye);
}


void changeMouse(cv::Mat& frame, cv::Point& location)
{
    if (location.x > frame.cols) location.x = frame.cols;
    if (location.x < 0) location.x = 0;
    if (location.y > frame.rows) location.y = frame.rows;
    if (location.y < 0) location.y = 0;
    system(("xdotool mousemove " + std::to_string(location.x) + " " + std::to_string(location.y)).c_str());
}


int main(int argc, char** argv) {

    faceCascade.load("haarcascade_frontalface_alt2.xml");
    eyeCascade.load("haarcascade_eye.xml");

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
            detectEyes(frame, faceCascade, eyeCascade);
            changeMouse(frame, mousePoint);
        }

        cv::imshow(windowName, frame);
    }
    return 0;
}
