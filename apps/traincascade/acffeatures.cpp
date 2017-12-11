#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include "acffeatures.h"
#include "cascadeclassifier.h"

using namespace cv;

CvACFFeatureParams::CvACFFeatureParams() : shrink (4),
{
        name = ACFF_NAME;

}

CvACFFeatureParams::CvACFFeatureParams( int _shrink ) : shrink( _shrink )
{
    name = ACFF_NAME;
    featSize = ();
}

// Not sure if I keep this
void CvACFFeatureParams::init( const CvFeatureParams& fp )
{
    CvFeatureParams::init( fp );
}

// Nor this
void CvACFFeatureParams::write( FileStorage &fs ) const
{
    CvFeatureParams::write( fs );
}

void CvACFEvaluator::init(const CvFeatureParams *_featureParams,
                           int _maxSampleCount, Size _winSize )
{
    CV_Assert(_maxSampleCount > 0);
    int cols = (_winSize.width / shrink) * (_winSize.height / shrink);
    //_featureParams->featSize = (_winSize.width / shrink) * (_winSize.height / shrink);
    single_sample_features.create((int)_maxSampleCount, cols, CV_32FC1);
    CvFeatureEvaluator::init( _featureParams, _maxSampleCount, _winSize );
}

void CvACFEvaluator::setImage(const Mat &img, uchar clsLabel, int idx)
{
    CV_DbgAssert( !single_sample_features.empty() );
    CvFeatureEvaluator::setImage( img, clsLabel, idx );
    computeColorFeatureChannels(img, feature_channel_vec);
}

void CvACFEvaluator::computeColorFeatureChannels(const cv::Mat& img, std::vector<cv::Mat>& feature_channel_vec) const
{
    cv::Mat frameLUV;
    cv::cvtColor(frame, frameLUV, CV_BGR2Luv);

    std::vector<cv::Mat> channels;
    cv::split(frameLUV, channels);

    feature_channel_vec.insert(feature_channel_vec.end(), channels.begin(), channels.end());
}

void CvACFEvaluator::computeHOG(const cv::Mat& img, std::vector<cv::Mat>& feature_channel_vec) const
{
    cv::Mat_<float> mag, angles;
    cv::Mat floatImg, grayImg;
    floatImg = cv::Mat(frame.rows, frame.cols, CV_32FC3);

    // Convert frame into a float mat and then to grayscale
    frame.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);
    cv::cvtColor(floatImg, grayImg, CV_BGR2GRAY);

    cv::Mat_<float> gx, gy;
    int xOrder, yOrder, kernelSize;

    // Contour extraction
    cv::Sobel(grayImg, gx, CV_32F, xOrder = 1, yOrder = 0, kernelSize = 3);
    cv::Sobel(grayImg, gy, CV_32F, xOrder = 0, yOrder = 1, kernelSize = 3);

    // Getting magnitude and angle values from gx and gy (angles are in degree)
    cv::cartToPolar(gx, gy, mag, angles, true);

    // Pushing the mag channel in the vector
    feature_chn_vec.push_back(mag);

    // creating a Mat of depth 6
    cv::Mat_<cv::Vec6f> histograms = cv::Mat_<cv::Vec6f>::zeros(mag.rows, mag.cols);

    // Assigning each angle value to the right depth
    for (int row = 0; row < mag.rows; row++) {
        for (int col = 0; col < mag.cols; col++) {
            int angle = (int)angles(row, col);

            angle %= 180;
            int layer = std::min(angle / 30, 5);
            histograms(row, col)[layer] = mag(row, col) * 255;
        }
    }

    std::vector<cv::Mat> histSplitter;
    cv::split(histograms, histSplitter);

    feature_channel_vec.insert(feature_channel_vec.end(), histSplitter.begin(), histSplitter.end());

}

cv::Mat
CvACFEvaluator::apply_shrink(cv::Mat& integral_channel_feature,
                  const std::vector<cv::Point>& all_coordinates) {
    cv::Mat shrunk_channel_feature = cv::Mat::zeros(integral_channel_feature.rows / m_shrink,
                                                    integral_channel_feature.cols / m_shrink,
                                                    CV_32FC1);

    // Number of possible sliding horizontal/ vertical windows
    size_t nb_windows_x = ((integral_channel_feature.cols - _window.width) / stride) + 1;
    size_t nb_windows_y = ((integral_channel_feature.rows - _window.height) / stride) + 1;

    for (size_t y = 0; y < nb_windows_y; y++) {
        for (size_t x = 0; x < nb_windows_x; x++) {
            std::vector<cv::Point> cell_coordinates = fetch_coordinates(y, x, all_coordinates);
            float avg = (apply_summation(cell_coordinates, integral_channel_feature)) /
                        (m_shrink * m_shrink);
            shrunk_channel_feature.ptr<float>(y)[x] = avg;
        }
    }

    return shrunk_channel_feature;
}

void CvACFEvaluator::generateFeatures()
{
        compute_coordinates(0, 0, call_coordinates);
        for (const auto feature_channel : feature_channel_vec) {
        cv::Mat integral_img;
        cv::integral(feature_channel, integral_img, CV_32FC1);
        cv::Mat shrunk_channel = apply_shrink(integral_img, all_coordinates);
        cv::Mat smoothed_channel = apply_smoothing(shrunk_channel);

        for (size_t row_index = 0; row_index < (size_t)smoothed_channel.rows; row_index++) {
            const float* row = smoothed_channel.ptr<float>(row_index);
            for (size_t col_index = 0; col_index < (size_t)smoothed_channel.cols; col_index++) {
                vectorized_features.push_back(row[col_index]);
            }
        }
    }
    return vectorized_features;
}
