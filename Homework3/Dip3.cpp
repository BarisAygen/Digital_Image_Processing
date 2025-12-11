//============================================================================
// Name    : Dip3.cpp
// Author      : Ronny Haensch, Andreas Ley, Simon Matern
// Version     : 3.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip3.h"

#include <stdexcept>
#include <numeric>

namespace dip3 {

const char* const filterModeNames[NUM_FILTER_MODES] = {
    "FM_SPATIAL_CONVOLUTION",
    "FM_FREQUENCY_CONVOLUTION",
    "FM_SEPERABLE_FILTER",
    "FM_INTEGRAL_IMAGE",
};

static unsigned int __SEP_FILTER_MODE = 0;
/**
 * @brief Generates 1D gaussian filter kernel of given size
 * @param kSize Kernel size (used to calculate standard deviation)
 * @returns The generated filter kernel
 */
cv::Mat_<float> createGaussianKernel1D(int kSize){

    // TO DO !!!
    cv::Mat_<float> kernel = cv::Mat_<float>::zeros(1, kSize);
    // We based our calculation on lecture 4 slide 76 and the given exercise

    // I'm not sure if that's correct (sigma), i take the value given as e.g. 5
    float sigma = static_cast<float>(kSize) / 5;
    int half = kSize / 2;
    float sum = 0.0;

    // computing the values and the sum for normalization
    for(int i = 0; i < kSize; i++){
        float value = (1/(2*M_PI*sigma)) * std::exp(-((i-half) * (i-half)) / (2 * sigma * sigma));
        kernel(0, i) = value;
        sum += value;
    }

    // normaliing total sum = 1
    kernel /= sum;

    return kernel;
}

/**
 * @brief Generates 2D gaussian filter kernel of given size
 * @param kSize Kernel size (used to calculate standard deviation)
 * @returns The generated filter kernel
 */
cv::Mat_<float> createGaussianKernel2D(int kSize) {

    // TO DO !!!
    float sigma = static_cast<float>(kSize) / 5;
    cv::Point2i anchor(kSize / 2, kSize / 2);
    std::vector<int> kernel_range(kSize);
    // Prepare matrices for position differences
    std::iota(kernel_range.begin(), kernel_range.end(), 0);
    cv::Mat row_indices;
    cv::Mat col_indices;
    cv::repeat(kernel_range, kSize, 1, col_indices);
    cv::transpose(col_indices, row_indices);

    row_indices -= anchor.y;
    col_indices -= anchor.x;
    row_indices = row_indices.mul(row_indices);
    col_indices = col_indices.mul(col_indices);

    // Apply gaussian function elementwise
    cv::Mat_<float> sum_squared;
    cv::add(row_indices, col_indices, sum_squared);
    sum_squared = -(sum_squared / (2 * sigma * sigma));
    cv::Mat_<float> exp_spatial;
    cv::exp(sum_squared, exp_spatial);
    exp_spatial *= 1.0f / (2.0f * CV_PI * sigma * sigma);
 
    return exp_spatial / cv::sum(exp_spatial)[0];
}

/**
 * @brief Performes a circular shift in (dx,dy) direction
 * @param in Input matrix
 * @param dx Shift in x-direction
 * @param dy Shift in y-direction
 * @returns Circular shifted matrix
 */
cv::Mat_<float> circShift(const cv::Mat_<float>& in, int dx, int dy) {

    cv::Mat_<float> result = cv::Mat_<float>::zeros(in.rows, in.cols);
    cv::Mat_<float> tmp = cv::Mat_<float>::zeros(in.rows, in.cols);
    // Left shift
    if (dx < 0) { 
        in(cv::Range(0, in.rows), cv::Range(-dx, in.cols)).copyTo(tmp(cv::Range(0, tmp.rows), cv::Range(0, tmp.cols + dx)));
        in(cv::Range(0, in.rows), cv::Range(0, -dx)).copyTo(tmp(cv::Range(0, tmp.rows), cv::Range(tmp.cols + dx, tmp.cols)));
    // Right shift
    } else {
        in(cv::Range(0, in.rows), cv::Range(0, in.cols - dx)).copyTo(tmp(cv::Range(0, tmp.rows), cv::Range(dx, tmp.cols)));
        in(cv::Range(0, in.rows), cv::Range(in.cols - dx, in.cols)).copyTo(tmp(cv::Range(0, tmp.rows), cv::Range(0, dx)));
    }
    // Up shift
    if (dy < 0) {
        tmp(cv::Range(-dy, tmp.rows), cv::Range(0, tmp.cols)).copyTo(result(cv::Range(0, result.rows + dy), cv::Range(0, result.cols)));
        tmp(cv::Range(0, -dy), cv::Range(0, tmp.cols)).copyTo(result(cv::Range(result.rows + dy, result.rows), cv::Range(0, result.cols)));
    // Down shift 
    } else {
        tmp(cv::Range(0, tmp.rows - dy), cv::Range(0, tmp.cols)).copyTo(result(cv::Range(dy, result.rows), cv::Range(0, result.cols)));
        tmp(cv::Range(tmp.rows - dy, tmp.rows), cv::Range(0, tmp.cols)).copyTo(result(cv::Range(0, dy), cv::Range(0, result.cols)));
    }
    return result;
}

/**
 * @brief Performes convolution by multiplication in frequency domain
 * @param in Input image
 * @param kernel Filter kernel
 * @returns Output image
 */
cv::Mat_<float> frequencyConvolution(const cv::Mat_<float>& in,
                                     const cv::Mat_<float>& kernel) {

    // TO DO !!!
    cv::Mat_<float> in_like_kernel = cv::Mat_<float>::zeros(in.rows, in.cols);
    kernel.copyTo(in_like_kernel(cv::Rect(0, 0, kernel.cols, kernel.rows)));
    // Center kernel
    in_like_kernel = circShift(in_like_kernel, -kernel.cols / 2, -kernel.rows / 2);
    cv::Mat_<float> kernelDFT, inputDFT;

    cv::dft(in_like_kernel, kernelDFT);
    cv::dft(in, inputDFT);

    cv::Mat_<float> result;
    cv::mulSpectrums(inputDFT, kernelDFT, result, 0);
    cv::idft(result, result, cv::DFT_SCALE);


    return result(cv::Rect(0, 0, in.cols, in.rows));

}

/**
 * @brief  Performs UnSharp Masking to enhance fine image structures
 * @param in The input image
 * @param filterMode How convolution for smoothing operation is done
 * @param size Size of used smoothing kernel
 * @param thresh Minimal intensity difference to perform operation
 * @param scale Scaling of edge enhancement
 * @returns Enhanced image
 */
cv::Mat_<float> usm(const cv::Mat_<float>& in, FilterMode filterMode, int size,
                    float thresh, float scale) {
    // TO DO !!!

    // use smoothImage(...) for smoothing
    cv::Mat_<float> result = smoothImage(in, size, filterMode);
    cv::Mat_<float> y_2 = in - result;
    cv::Mat_<float> y_2_minus_Threshold;
    cv::Mat_<float> y_2_plus_Threshold;
    // Act only where distance > Threshold
    cv::threshold(y_2, y_2_minus_Threshold, -thresh, 255, cv::THRESH_TOZERO_INV);
    cv::threshold(y_2, y_2_plus_Threshold, thresh, 255, cv::THRESH_TOZERO);
    y_2 = scale * (y_2_minus_Threshold + y_2_plus_Threshold);
    cv::Mat_<float> final_img = y_2 + in;
    // Bring to the 0,255 range
    cv::threshold(final_img, final_img, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(final_img, final_img, 0, 255, cv::THRESH_TOZERO);

    return final_img;
}

/**
 * @brief Convolution in spatial domain
 * @param src Input image
 * @param kernel Filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src,
                                   const cv::Mat_<float>& kernel) {

    // Hopefully already DONE, copy from last homework
    cv::Mat_<float> output = cv::Mat_<float>::zeros(src.rows, src.cols);
    cv::Mat_<float> flipped_kernel(kernel.rows, kernel.cols);
    // Flip the kernel
    cv::flip(kernel, flipped_kernel, -1);
    // Set the anchor for the kernel
    cv::Point_<int> anchor(kernel.cols / 2, kernel.rows / 2);
    auto kernel_rows = kernel.rows;
    auto kernel_cols = kernel.cols;
    // Create padded img -> iterate through pixels ->
    // create window for the neighbourhood -> do elementwise mat mul with kernel
    // -> take sum -> set
    cv::Mat_<float> padded_img;
    cv::copyMakeBorder(src, padded_img, anchor.y, anchor.y, anchor.x, anchor.x,
                       cv::BORDER_WRAP);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {

            float sum = 0.0f;
            // In case of separable filter just iterate over the kernel and multiply/add (slicing the neighbourhood for 1D was bottlenecking)
            if (__SEP_FILTER_MODE) {
                for (int col = -kernel.cols / 2; col <= kernel.cols / 2; col++) {
                    sum += kernel.at<float>(col + anchor.x) * padded_img.at<float>(anchor.y + i, anchor.x + col + j);
                }
            } else {
                // Slice the neighborhood from the img
                auto window = padded_img(cv::Rect(j, i, kernel_cols, kernel_rows));
                sum = cv::sum(window.mul(flipped_kernel))[0];

            }
            output.at<float>(i, j) = sum;
        }
    }

    return output;
}

/**
 * @brief Convolution in spatial domain by seperable filters
 * @param src Input image
 * @param size Size of filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> separableFilter(const cv::Mat_<float>& src,
                                const cv::Mat_<float>& kernel) {

    // TO DO !!!
    __SEP_FILTER_MODE = 1;
    cv::Mat_<float> result;
    cv::transpose(spatialConvolution(src, kernel), result);
    cv::transpose(spatialConvolution(result, kernel), result);
    __SEP_FILTER_MODE = 0;
    
    return result;
}

/**
 * @brief Convolution in spatial domain by integral images
 * @param src Input image
 * @param size Size of filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> satFilter(const cv::Mat_<float>& src, int size) {

    // optional

    return src;
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

/**
 * @brief Performs a smoothing operation but allows the algorithm to be chosen
 * @param in Input image
 * @param size Size of filter kernel
 * @param type How is smoothing performed?
 * @returns Smoothed image
 */
cv::Mat_<float> smoothImage(const cv::Mat_<float>& in, int size,
                            FilterMode filterMode) {
    switch (filterMode) {
    case FM_SPATIAL_CONVOLUTION:
        return spatialConvolution(
            in, createGaussianKernel2D(size)); // 2D spatial convolution
    case FM_FREQUENCY_CONVOLUTION:
        return frequencyConvolution(
            in,
            createGaussianKernel2D(
                size)); // 2D convolution via multiplication in frequency domain
    case FM_SEPERABLE_FILTER:
        return separableFilter(
            in,
            createGaussianKernel1D(size)); // seperable filter
    case FM_INTEGRAL_IMAGE:
        return satFilter(in, size); // integral image
    default:
        throw std::runtime_error("Unhandled filter type!");
    }
}

} // namespace dip3