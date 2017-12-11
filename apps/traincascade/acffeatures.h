#ifndef _OPENCV_ACFFEATURES_H_
#define _OPENCV_ACFFEATURES_H_

#include "traincascade_features.h"

#define ACFF_NAME "acfFeatureParams"

struct CvACFFeatureParams : public CvFeatureParams
{
    CvACFFeatureParams();

};

class CvACFEvaluator : public CvFeatureEvaluator
{
public:
    virtual ~CvACFEvaluator() {}

    virtual void init(const CvFeatureParams *_featureParams,
        int _maxSampleCount, cv::Size _winSize );

    virtual void setImage(const cv::Mat& img, uchar clsLabel, int idx);
    virtual float operator()(int featureIdx /* could be varIdx*/, int sampleIdx) const;
    virtual void writeFeatures( cv::FileStorage &fs, const cv::Mat& featureMap ) const;
protected:
    virtual void generateFeatures();

    // Subjects to change
    virtual void computeColorFeatureChannels(const cv::Mat& img, std::vector<cv::Mat>& feature_channel_vec) const;
    virtual void compteHOG(const cv::Mat& img, std::vector<cv::Mat>& feature_channel_vec) const;
    virtual void compute_coordinates(size_t slide_on_y, size_t slide_on_x, std::vector<cv::Point>& all_coordinates);
    std::vector<cv::Point>
    fetch_coordinates(int cell_row, int cell_col, const std::vector<cv::Point>& coordinates_vec);

uint shrink;
    class Feature
    {
    public:
        Feature();
        Feature( int offset, int x, int y, int _block_w, int _block_h  );

        // could replace apply_summation
        uchar calc( const cv::Mat& _sum, size_t y ) const;
        void write( cv::FileStorage &fs ) const;

        cv::Rect rect;
        int p[16];
    };
    std::vector<Feature> features;

    std::vector<cv::Mat> feature_channel_vec;
    std::vector<cv::Point> all_coordinates;


    uint stride;

    cv::Mat single_sample_features;
};

#endif
