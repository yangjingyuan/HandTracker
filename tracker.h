/***************************************************************************
 * 
 * Copyright (c) 2016 yangjingyuan. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file tracker.h
 * @author yangjingyuan
 * @email hlbryang@163.com
 * @date 2016/11/02 14:16:35
 * @brief 
 *  
 **/

#ifndef TRACKER_H
#define TRACKER_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <algorithm>
#include <map>
#include <vector>
#include <set>
#include <ostream>
#include <math.h>
#include <stdlib.h>
#include <sstream>
#include <fstream>

namespace hand_tracker{

class hypothesis {
public:
    double getCenterX();
    double getCenterY();
    double getAlpha();
    double getBeta();
    double getThea();
    cv::Scalar getColor();
    double getRed();
    double getBlue();
    double getGreen();
    void setParameters(double centerX,double centerY,double a,double b,double angle,double red,double green,double blue);
    void setCenterX(double x);
    void setCenterY(double y);
    void setColor(double red, double green, double blue);
    bool firstTime = false;
private:
    double centerX,centerY;
    double aplha,beta,thea;
    int red, green, blue;
};


class tracker {
public:
    cv::Mat track(std::map<int, std::vector<cv::Point> > hand_blobs);
    //cv::Mat getTrackedFrame(cv::Mat videoFrame, cv::Mat detFrame);
    //cv::Mat getBlobImg();
    std::map<int, hypothesis> getHandHypotheses(cv::Mat videoFrame, cv::Mat detFrame);
    std::map<int, hypothesis> obtainHandHypotheses(cv::Mat videoFrame, cv::Mat detFrame);
    void display(cv::Mat& draw_img);
    int track_time = 1;
private:
    std::map<int, std::vector<cv::Point> > hypoGeneration(std::map<int, std::vector<cv::Point> > blobRecorder);
    void hypoTracking(std::map<int, std::vector<cv::Point> > blobRecorder);
    std::map<int,hypothesis> removeHypothesis(std::map<int, std::vector<cv::Point> > blobs);
    void hypoPredition();
    double disToEllpse(cv::Point p, hypothesis h);
    bool blob_has_hypothesis(std::vector<cv::Point> blob);
    hypothesis getHypoByBlob(std::vector<cv::Point> blob);
    std::map<int,hypothesis> currHypoList;
    std::map<int,hypothesis> t_1HypoList;
    std::map<int,hypothesis> t_2HypoList;
    void drawBlobs(std::map<int, std::vector<cv::Point> > blobRecorder);
    cv::Mat blobImg;
};

} //hand_tracker
#endif
