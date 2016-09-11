#include <cstdlib>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <opencv2/opencv.hpp>
#include "LeNet.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
    {
    Mat img = imread("3.png", 0);
    
    // Scale layer set scale equal to 0.0125
    img.convertTo(img, CV_32FC1, 0.0125);
    // ---------------------------------------------
    // Set image
    // ---------------------------------------------
    vector<float> result;
    LeNet net;
    double t = (double) getTickCount();
    for (int i = 0; i < 1000; ++i)
        {
        net.predict(img, result);
        }
    t = ((double) getTickCount() - t) / getTickFrequency();
    cout << "elapsed time is: " << t * 1000 << "ms." << endl;
    
    int labels[] = {9, 6, 5, 3, 4, 8, 2, 1, 0, 7};
    cout << endl;
    for (int i = 0; i < 10; ++i)
        {
        cout << "results: " << labels[i] << ":" << result[i] << endl;
        }
    return 0;
    }
