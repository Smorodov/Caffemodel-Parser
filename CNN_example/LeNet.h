/* 
 * File:   LeNet.h
 * Author: andrey
 *
 * Created on November 19, 2015, 4:34 PM
 */

#ifndef LENET_H
#define	LENET_H

#include <cstdlib>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <opencv2/opencv.hpp>

typedef float Real;

class tensor
    {
public:
    Real* data;
    int n_, c_, h_, w_;
    // -------------------------------------------

    inline Real& operator()(int n, int c, int h, int w) const
        {
        return data[n * c_ * w_ * h_ + c * w_ * h_ + h * w_ + w];
        }
    // -------------------------------------------

    tensor();

    void create(int n, int c, int h, int w);

    // -------------------------------------------
    // Not initialized constructor
    // -------------------------------------------

    tensor(int n, int c, int h, int w);
    // -------------------------------------------
    // Constructor from opencv matrix
    // -------------------------------------------

    tensor(cv::Mat& m);

    void setImage(cv::Mat & m);


    // -------------------------------------------
    // Constructor from binary file
    // -------------------------------------------

    tensor(int n, int c, int h, int w, std::string filename);
    // -------------------------------------------
    //   Destructor
    // -------------------------------------------

    ~tensor();
    // -------------------------------------------
    //   Extract 1 channel float type matrix from tensor
    // -------------------------------------------

    void getImage(int n, int c, cv::Mat& dst);

    // -------------------------------------------
    // Convolve
    // -------------------------------------------
    void convlove(const tensor& filter, tensor& result);

    // -------------------------------------------
    // Pooling max
    // ------------------------------------------- 

    void maxPool(cv::Size pool_size, tensor& result);
    
    // -------------------------------------------
    // Local normalization layer
    // ------------------------------------------- 
    void LRN_within_cahnnel(void);
    void LRN_across_channels(void);
    // -------------------------------------------
    // ReLu
    // ------------------------------------------- 

    void ReLu(void);
    // -------------------------------------------
    // Add bias
    // ------------------------------------------- 

    void addBias(const tensor& bias);

    void softMax(void);

    void FullConnected(tensor& weights, tensor& bias, tensor& result);

    };

class LeNet
    {
private:
    tensor* conv1;
    tensor* conv1bias;
    tensor* conv2;
    tensor* conv2bias;
    tensor* fc1;
    tensor* fc1bias;
    tensor* fc2;
    tensor* fc2bias;

    tensor* data;
    tensor* data1;
    tensor* pool1;
    tensor* data2;
    tensor* pool2;
    tensor* fc1_res;
    tensor* fc2_res;
    void setImage(cv::Mat& m);

public:
    LeNet();
    ~LeNet();
    void predict(cv::Mat& m, std::vector<Real>& result);
    };

#endif	/* LENET_H */

