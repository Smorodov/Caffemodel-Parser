/* 
 * File:   LeNet.cpp
 * Author: andrey
 * 
 * Created on November 19, 2015, 4:34 PM
 */
#include "LeNet.h"
#include <cstdlib>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

tensor::tensor()
    {
    n_ = 0;
    c_ = 0;
    h_ = 0;
    w_ = 0;
    data = NULL;
    }

void tensor::create(int n, int c, int h, int w)
    {
    n_ = n;
    c_ = c;
    h_ = h;
    w_ = w;
    if (data != NULL)
        {
        delete data;
        data = NULL;
        }
    data = new Real[n * c * w * h];
    memset((char*) data, 0, n * c * w * h * sizeof (Real));
    }

// -------------------------------------------
// Not initialized constructor
// -------------------------------------------

tensor::tensor(int n, int c, int h, int w)
    {
    n_ = n;
    c_ = c;
    h_ = h;
    w_ = w;
    data = new Real[n * c * w * h];
    memset((char*) data, 0, n * c * w * h * sizeof (Real));
    }
// -------------------------------------------
// Constructor from opencv matrix
// -------------------------------------------

tensor::tensor(cv::Mat& m)
    {
    n_ = 1;
    c_ = m.channels();
    h_ = m.rows;
    w_ = m.cols;
    //  int ind = 0;
    data = new Real[n_ * c_ * w_ * h_];
    memset((char*) data, 0, n_ * c_ * w_ * h_ * sizeof (Real));
    if (c_ > 1)
        {
        for (int ch = 0; ch < c_; ++ch)
            {
            for (int i = 0; i < h_; ++i)
                {
                for (int j = 0; j < w_; ++j)
                    {
                    operator()(0, ch, i, j) = m.at<cv::Vec3f>(i, j)[ch];
                    //   ++ind;
                    }
                }
            }
        }
    else
        {
        for (int i = 0; i < h_; ++i)
            {
            for (int j = 0; j < w_; ++j)
                {
                operator()(0, 0, i, j) = m.at<float>(i, j);
                //  ++ind;
                }
            }
        }
    }

void tensor::setImage(cv::Mat & m)
    {
    // If image size is differs from current, reallocate memory.
    if (n_ != 1 || c_ != m.channels() || h_ != m.rows || w_ != m.cols)
        {
        if (data != NULL)
            {
            delete data;
            data = NULL;
            }
        n_ = 1;
        c_ = m.channels();
        h_ = m.rows;
        w_ = m.cols;
        data = new Real[n_ * c_ * w_ * h_];
        }
    // Now copy image data to tensor
    int ind = 0;
    memset((char*) data, 0, n_ * c_ * w_ * h_ * sizeof (Real));
    // Grayscale
    if (c_ > 1)
        {
        for (int ch = 0; ch < c_; ++ch)
            {
            for (int i = 0; i < h_; ++i)
                {
                for (int j = 0; j < w_; ++j)
                    {
                    operator()(0, ch, i, j) = m.at<cv::Vec3f>(i, j)[ch];
                    ++ind;
                    }
                }
            }
        }
        // Color
    else
        {
        for (int i = 0; i < h_; ++i)
            {
            for (int j = 0; j < w_; ++j)
                {
                operator()(0, 0, i, j) = m.at<float>(i, j);
                ++ind;
                }
            }
        }
    }
// -------------------------------------------
// Constructor from binary file
// -------------------------------------------

tensor::tensor(int n, int c, int h, int w, std::string filename)
    {
    n_ = n;
    c_ = c;
    h_ = h;
    w_ = w;
    int size = n * c * w * h;

    // File stored in float format, so we need to read it in float format
    float* data_tmp = new float[n * c * w * h];
    std::ifstream dataFile;
    dataFile.open(filename, std::ios::in | std::ios::binary);
    if (!dataFile)
        {
        std::cout << "Error opening file " << filename << std::endl;
        }
    int size_b = size * sizeof (float);
    if (!dataFile.read((char*) data_tmp, size_b))
        {
        std::cout << "Error reading file " << filename << std::endl;
        }
    dataFile.close();

    data = new Real[size];
    for (int i = 0; i < size; ++i)
        {
        data[i] = data_tmp[i];
        }
    delete data_tmp;
    }
// -------------------------------------------
//   Destructor
// -------------------------------------------

tensor::~tensor()
    {
    if (data != NULL)
        {
        delete data;
        data = NULL;
        }
    }
// -------------------------------------------
//   Extract 1 channel Real type matrix from tensor
// -------------------------------------------

void tensor::getImage(int n, int c, cv::Mat& dst)
    {
    int ind = n * c_ * w_ * h_ + c * w_*h_;
    dst = cv::Mat::zeros(h_, w_, CV_32FC1);
    for (int i = 0; i < h_; ++i)
        {
        for (int j = 0; j < w_; ++j)
            {
            dst.at<float>(i, j) = operator()(n, c, i, j);
            ++ind;
            }
        }
    }

// -------------------------------------------
// Convolve
// -------------------------------------------

void tensor::convlove(const tensor& filter, tensor& result)
    {
    int dim_x = h_ - filter.h_ + 1;
    int dim_y = w_ - filter.w_ + 1;
    result.create(n_, filter.n_, dim_y, dim_x);

    int result_n_stride = result.c_ * result.w_ * result.h_;
    int result_c_stride = result.w_ * result.h_;
    int result_h_stride = result.w_;

    int n_stride = c_ * w_ * h_;
    int c_stride = w_ * h_;
    int h_stride = w_;

    int filter_n_stride = filter.c_ * filter.w_ * filter.h_;
    int filter_c_stride = filter.w_ * filter.h_;
    int filter_h_stride = filter.w_;

    for (int n = 0; n < n_; ++n)
        {
        for (int m = 0; m < filter.n_; ++m)
            {
            int base_addr_0_f = m*filter_n_stride;
            for (int h = 0; h < dim_y; ++h)
                {
                int base_addr_0 = n * n_stride + h * h_stride;
        #pragma omp parallel for  firstprivate(base_addr_0_f,base_addr_0)      
                for (int w = 0; w < dim_x; ++w)
                    {
                    int base_addr_1 = base_addr_0 + w;
                    Real sum = 0;
                    for (int d = 0; d < c_; ++d)
                        {
                        int base_addr2 = d * c_stride + base_addr_1;
                        int base_addr_1_f = base_addr_0_f + d * filter_c_stride;
                        for (int y = 0; y < filter.h_; ++y)
                            {
                            int base_addr3 = base_addr2 + y * h_stride;
                            int base_addr_2_f = base_addr_1_f + y * filter_h_stride;
                            for (int x = 0; x < filter.w_; ++x)
                                {
                                sum += data[base_addr3 + x] * filter.data[base_addr_2_f + x];
                                }
                            }
                        }
                    result.data[result_n_stride * n + result_c_stride * m + result_h_stride * h + w] = sum;
                    }
                }
            }
        }
    }

// -------------------------------------------
// Pooling max
// ------------------------------------------- 

void tensor::maxPool(Size pool_size, tensor& result)
    {
    int stride_x = 2;
    int stride_y = 2;
    int dim_x = (w_ - pool_size.width) / stride_x + 1;
    int dim_y = (h_ - pool_size.height) / stride_y + 1;
    result.create(n_, c_, dim_y, dim_x);
    //Real val = 0;

    int n_stride = c_ * w_ * h_;
    int c_stride = w_ * h_;
    int h_stride = w_;

    // Create LUT to avoid index comutation
    int pool_area = pool_size.width * pool_size.height;
    char* LUT = new char[pool_area];
    int ind = 0;
    for (int y = 0; y < pool_size.height; ++y)
        {
        for (int x = 0; x < pool_size.width; ++x)
            {
            LUT[ind] = x + h_stride*y;
            ++ind;
            }
        }
    // Start processing
    int result_index = 0;
    for (int n = 0; n < n_; ++n)
        {
        for (int ch = 0; ch < c_; ++ch)
            {
            for (int i = 0; i < h_ - pool_size.height + 1; i += stride_y)
                {
                for (int j = 0; j < w_ - pool_size.width + 1; j += stride_x)
                    {
                    float m = operator()(n, ch, i, j);
                    int base_addr0 = n_stride * n + c_stride * ch + h_stride * i + j;
                    for (int p = 0; p < pool_area; ++p)
                        {
                        Real val = data[base_addr0 + LUT[p]];
                        if (m < val)
                            {
                            m = val;
                            }
                        }
                    result.data[result_index] = m;
                    ++result_index;
                    }
                }
            }
        }
    delete LUT;
    }
// -------------------------------------------
// ReLu
// ------------------------------------------- 

void tensor::ReLu(void)
    {
    int total_size = n_ * c_ * w_*h_;
#pragma omp parallel for
    for (int n = 0; n < total_size; ++n)
        {
        // data[n] = log(1+exp(data[n]));
        data[n] = std::max((Real) 0.0, data[n]);
        }
    }

// -------------------------------------------
// Local normalization layer
// ------------------------------------------- 

void tensor::LRN_across_channels(void)
    {
    int local_size = 5;
    Real beta = 0.6;
    Real alpha = 0.1;

    for (int n = 0; n < n_; ++n)
        {
        for (int h = 0; h < h_; ++h)
            {
            for (int w = 0; w < w_; ++w)
                {
                for (int c = 0; c < c_ - local_size + 1; c += local_size)
                    {
                    Real sq_sum = 0;
                    for (int i = 0; i < local_size; ++i)
                        {
                        Real v = operator()(n, i + c, h, w);
                        sq_sum += v*v;
                        }

                    Real divisor = pow(1 + (alpha / local_size) * sq_sum, beta);

                    for (int i = 0; i < local_size; ++i)
                        {
                        if (divisor > 0)
                            {
                            operator()(n, i + c, h, w) /= divisor;
                            }
                        }
                    }
                }
            }
        }
    }
// -------------------------------------------
// Add bias
// ------------------------------------------- 

void tensor::addBias(const tensor& bias)
    {
    int ind = 0;
    int ind_b = 0;
    for (int n = 0; n < n_; ++n)
        {
        for (int ch = 0; ch < c_; ++ch)
            {
            Real b = bias.data[ind_b]; // bias(n, ch, 0, 0);
            ++ind_b;
            for (int i = 0; i < h_; ++i)
                {
                for (int j = 0; j < w_; ++j)
                    {
                    data[ind] += b;
                    ++ind;
                    }
                }
            }
        }
    }
//-----------------------------------
//
//------------------------------------

void tensor::softMax(void)
    {
    Real sum = 0;
    int total_size = n_ * c_ * w_*h_;

    Real scale_data = 0;

    for (int n = 0; n < total_size; ++n)
        {
        scale_data = max(scale_data, data[n]);
        }

    for (int n = 0; n < total_size; ++n)
        {
        data[n] = exp(data[n] - scale_data);
        sum += data[n];
        }

    for (int n = 0; n < total_size; ++n)
        {
        data[n] = data[n] / sum;
        }
    }

void tensor::FullConnected(tensor& weights, tensor& bias, tensor& result)
    {
    // It works only for data tensor with n=1.
    result.create(1, weights.n_, 1, 1);
    int data_size = w_ * h_*c_;
    int weigths_stride = weights.c_ * weights.h_ * weights.w_;
    for (int n1 = 0; n1 < weights.n_; ++n1)
        {
        Real sum = 0;
        int offset = weigths_stride*n1;
#pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < data_size; ++i)
            {
            sum += data[i] * weights.data[offset + i];
            }
        result.data[n1] = sum;
        result.data[n1] += bias.data[n1];
        }
    }

void LeNet::setImage(cv::Mat& m)
    {
    data->setImage(m);
    }

LeNet::LeNet()
    {
    // ---------------------------------------------
    // Load our network weights
    // ---------------------------------------------
    conv1 = new tensor(20, 1, 5, 5, "conv1.bin");
    conv1bias = new tensor(1, 20, 1, 1, "conv1.bias.bin");
    conv2 = new tensor(50, 20, 5, 5, "conv2.bin");
    conv2bias = new tensor(1, 50, 1, 1, "conv2.bias.bin");
    fc1 = new tensor(500, 800, 1, 1, "ip1.bin");
    fc1bias = new tensor(1, 500, 1, 1, "ip1.bias.bin");
    fc2 = new tensor(10, 500, 1, 1, "ip2.bin");
    fc2bias = new tensor(1, 10, 1, 1, "ip2.bias.bin");

    data = new tensor();
    data1 = new tensor();
    pool1 = new tensor();
    data2 = new tensor();
    pool2 = new tensor();
    fc1_res = new tensor();
    fc2_res = new tensor();
    }

LeNet::~LeNet()
    {
    delete conv1;
    delete conv1bias;
    delete conv2;
    delete conv2bias;
    delete fc1;
    delete fc1bias;
    delete fc2;
    delete fc2bias;
    delete data;
    delete data1;
    delete pool1;
    delete data2;
    delete pool2;
    delete fc1_res;
    delete fc2_res;
    }

void LeNet::predict(cv::Mat& m, std::vector<Real>& result)
    {
    setImage(m);
    data->convlove(*conv1, *data1);
    data1->addBias(*conv1bias);
    data1->maxPool(cv::Size(2, 2), *pool1);

    pool1->convlove(*conv2, *data2);
    data2->addBias(*conv2bias);
    data2->maxPool(cv::Size(2, 2), *pool2);

    pool2->FullConnected(*fc1, *fc1bias, *fc1_res);
    fc1_res->ReLu();
    fc1_res->FullConnected(*fc2, *fc2bias, *fc2_res);
    fc2_res->softMax();
    for (int i = 0; i < fc2_res->c_; ++i)
        {
        result.push_back(fc2_res->operator()(0, i, 0, 0));
        }
    }




