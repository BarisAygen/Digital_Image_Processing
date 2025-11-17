//============================================================================
// Name        : Dip2.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip2.h"

namespace dip2 {


/**
 * @brief Convolution in spatial domain.
 * @details Performs spatial convolution of image and filter kernel.
 * @params src Input image
 * @params kernel Filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel)
{
    cv::Mat_<float> result = cv::Mat_<float>::zeros(src.size());

    cv::Mat_<float> flipped_kernel;
    cv::flip(kernel, flipped_kernel, -1);

    int k_cx = flipped_kernel.cols / 2;
    int k_cy = flipped_kernel.rows / 2;

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {

            float sum = 0.0f;

            for (int r = 0; r < flipped_kernel.rows; r++) {
                for (int s = 0; s < flipped_kernel.cols; s++) {

                    int src_y = y + (r - k_cy);
                    int src_x = x + (s - k_cx);

                    if (src_y >= 0 && src_y < src.rows && src_x >= 0 && src_x < src.cols) {
                        sum += flipped_kernel(r, s) * src(src_y, src_x);
                    }
                }
            }

            result(y, x) = sum;
        }
    }

    return result;
}

/**
 * @brief Moving average filter (aka box filter)
 * @note: you might want to use Dip2::spatialConvolution(...) within this function
 * @param src Input image
 * @param kSize Window size used by local average
 * @returns Filtered image
 */
cv::Mat_<float> averageFilter(const cv::Mat_<float>& src, int kSize)
{
    cv::Mat_<float> kernel(kSize, kSize);

    float kernel_value = 1.0f / (kSize * kSize);

    kernel.setTo(kernel_value);

    return spatialConvolution(src, kernel);
}

/**
 * @brief Median filter
 * @param src Input image
 * @param kSize Window size used by median operation
 * @returns Filtered image
 */
cv::Mat_<float> medianFilter(const cv::Mat_<float>& src, int kSize)
{
    cv::Mat_<float> result = cv::Mat_<float>::zeros(src.size());

    int k_half = kSize / 2;

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {

            std::vector<float> neighbors;

            for (int r = -k_half; r <= k_half; r++) {
                for (int s = -k_half; s <= k_half; s++) {

                    int neighbor_y = y + r;
                    int neighbor_x = x + s;

                    if (neighbor_y >= 0 && neighbor_y < src.rows && neighbor_x >= 0 && neighbor_x < src.cols) {
                        neighbors.push_back(src(neighbor_y, neighbor_x));
                    }
                }
            }

            std::sort(neighbors.begin(), neighbors.end());

            result(y, x) = neighbors[neighbors.size() / 2];
        }
    }

    return result;
}

/**
 * @brief Bilateral filer
 * @param src Input image
 * @param kSize Size of the kernel
 * @param sigma_spatial Standard-deviation of the spatial kernel
 * @param sigma_radiometric Standard-deviation of the radiometric kernel
 * @returns Filtered image
 */
cv::Mat_<float> bilateralFilter(const cv::Mat_<float>& src, int kSize, float sigma_spatial, float sigma_radiometric)
{
    cv::Mat_<float> result = cv::Mat_<float>::zeros(src.size());

    int k_half = kSize / 2;

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {

            float center_intensity = src(y, x);

            float weightedSum = 0.0f;  
            float totalWeight = 0.0f;  

            for (int r = -k_half; r <= k_half; r++) {
                for (int s = -k_half; s <= k_half; s++) {

                    int neighbor_y = y + r;
                    int neighbor_x = x + s;

                    if (neighbor_y >= 0 && neighbor_y < src.rows && neighbor_x >= 0 && neighbor_x < src.cols) {
                        
                        float neighbor_intensity = src(neighbor_y, neighbor_x);

                        
                        float spatial_dist_sq = r*r + s*s;
                        float weight_spatial = std::exp(-spatial_dist_sq / (2 * sigma_spatial * sigma_spatial));
                        
                        float radiometric_dist_sq = (neighbor_intensity - center_intensity) * (neighbor_intensity - center_intensity);
                        float weight_radiometric = std::exp(-radiometric_dist_sq / (2 * sigma_radiometric * sigma_radiometric));

                        float final_weight = weight_spatial * weight_radiometric;

                        weightedSum += final_weight * neighbor_intensity;
                        totalWeight += final_weight;
                    }
                }
            }

            result(y, x) = weightedSum / totalWeight;
        }
    }

    return result;
}

/**
 * @brief Non-local means filter
 * @note: This one is optional!
 * @param src Input image
 * @param searchSize Size of search region
 * @param sigma Optional parameter for weighting function
 * @returns Filtered image
 */
cv::Mat_<float> nlmFilter(const cv::Mat_<float>& src, int searchSize, double sigma)
{
    cv::Mat_<float> result = cv::Mat_<float>::zeros(src.size());

    int patchSize = 7;
    int patchHalf = patchSize / 2; 
    int searchHalf = searchSize / 2;

    double h_sq = sigma * sigma;

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {

            float weightedSum = 0.0f;
            float totalWeight = 0.0f;

            for (int r = -searchHalf; r <= searchHalf; r++) {
                for (int s = -searchHalf; s <= searchHalf; s++) {
                    
                    int q_y = y + r;
                    int q_x = x + s;

                    if (q_y >= 0 && q_y < src.rows && q_x >= 0 && q_x < src.cols) {
                        
                        double ssd = 0.0; 

                        for (int pr = -patchHalf; pr <= patchHalf; pr++) {
                            for (int ps = -patchHalf; ps <= patchHalf; ps++) {

                                int p_y_offset = y + pr;
                                int p_x_offset = x + ps;
                                int q_y_offset = q_y + pr;
                                int q_x_offset = q_x + ps;

                                if (p_y_offset >= 0 && p_y_offset < src.rows && p_x_offset >= 0 && p_x_offset < src.cols &&
                                    q_y_offset >= 0 && q_y_offset < src.rows && q_x_offset >= 0 && q_x_offset < src.cols)
                                {
                                    float diff = src(p_y_offset, p_x_offset) - src(q_y_offset, q_x_offset);
                                    ssd += diff * diff;
                                }
                            }
                        }
                        
                        ssd /= (patchSize * patchSize);
                        float weight = std::exp(-ssd / h_sq);

                        weightedSum += weight * src(q_y, q_x);
                        totalWeight += weight;
                    }
                }
            }
        
            result(y, x) = weightedSum / totalWeight;
        }
    }
    
    return result;
}



/**
 * @brief Chooses the right algorithm for the given noise type
 * @note: Figure out what kind of noise NOISE_TYPE_1 and NOISE_TYPE_2 are and select the respective "right" algorithms.
 */
NoiseReductionAlgorithm chooseBestAlgorithm(NoiseType noiseType)
{
    switch (noiseType) {
        case NOISE_TYPE_1:
            return NR_MEDIAN_FILTER;
        case NOISE_TYPE_2:
            return NR_MOVING_AVERAGE_FILTER; 
        default:
            return (NoiseReductionAlgorithm) -1;
    }
}



cv::Mat_<float> denoiseImage(const cv::Mat_<float> &src, NoiseType noiseType, dip2::NoiseReductionAlgorithm noiseReductionAlgorithm)
{
    switch (noiseReductionAlgorithm) {
        case dip2::NR_MOVING_AVERAGE_FILTER:
            switch (noiseType) {
                case NOISE_TYPE_1:
                    return dip2::averageFilter(src, 5); 
                case NOISE_TYPE_2:
                    return dip2::averageFilter(src, 5); 
                default:
                    throw std::runtime_error("Unhandled noise type!");
            }
        case dip2::NR_MEDIAN_FILTER:
            switch (noiseType) {
                case NOISE_TYPE_1:
                    return dip2::medianFilter(src, 5); 
                case NOISE_TYPE_2:
                    return dip2::medianFilter(src, 5); 
                default:
                    throw std::runtime_error("Unhandled noise type!");
            }
        case dip2::NR_BILATERAL_FILTER:
            switch (noiseType) {
                case NOISE_TYPE_1:
                    return dip2::bilateralFilter(src, 5, 2.0f, 50.0f); 
                case NOISE_TYPE_2:
                    return dip2::bilateralFilter(src, 5, 2.0f, 50.0f); 
                default:
                    throw std::runtime_error("Unhandled noise type!");
            }
        default:
            throw std::runtime_error("Unhandled filter type!");
    }
}





// Helpers, don't mind these

const char *noiseTypeNames[NUM_NOISE_TYPES] = {
    "NOISE_TYPE_1",
    "NOISE_TYPE_2",
};

const char *noiseReductionAlgorithmNames[NUM_FILTERS] = {
    "NR_MOVING_AVERAGE_FILTER",
    "NR_MEDIAN_FILTER",
    "NR_BILATERAL_FILTER",
};


}
