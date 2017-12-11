#ifndef _OPENCV_ACFFEATURES_H_
#define _OPENCV_ACFFEATURES_H_

#include "traincascade_features.h"

#define CBF_NAME "cbFeatureParams"
struct CvCBFeatureParams : CvFeatureParams
{
    CvCBFeatureParams();

};

class CvCBEvaluator : public CvFeatureEvaluator
{
public:
    virtual ~CvCBEvaluator() {}
    virtual void init(const CvFeatureParams *_featureParams,
        int _maxSampleCount, cv::Size _winSize );
    virtual void setImage(const cv::Mat& img, uchar clsLabel, int idx);
    virtual float operator()(int featureIdx, int sampleIdx) const;
    virtual void writeFeatures( cv::FileStorage &fs, const cv::Mat& featureMap ) const;
protected:
    virtual void generateFeatures();
    // ANYTHING RELATED TO CALCULATING SHOULD BE HERE

    class Feature
    {
    public:
        Feature();
        Feature( int offset, int x, int y, int _block_w, int _block_h  );
        uchar calc( const cv::Mat& _sum, size_t y ) const;
        void write( cv::FileStorage &fs ) const;

        cv::Rect rect;
        int p[16];
    };
    std::vector<Feature> features;

    cv::Mat sum;
};

#endif
