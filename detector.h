/***************************************************************************
 * 
 * Copyright (c) 2016 yangjingyuan. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file detector.h
 * @author yangjingyuan
 * @email hlbryang@163.com
 * @date 2016/11/02 14:16:35
 * @brief 
 *  
 **/

#ifndef DETECTOR_H
#define DETECTOR_H

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

//U and V elements in YUV color space
typedef std::pair<int, int> UV;

class detector {	

public:

/**
 *@brief load models trained under weak, middle and strong light conditions
 *@param model_path the dir where molde stores
 *@retrun 0 if success, -1 otherwise  
 **/
int init_models(std::string model_dir);

void calculate_classifier_threshold(int Y);

double hand_classifier(int Y, int U, int V);

int check_blobs(cv::Mat frame);

void remove_faces();

void two_pass_labeling();

void size_filtering();

//cv::Mat detect(cv::Mat frame);
std::map<int, std::vector<cv::Point> > detect(cv::Mat frame);

void display();

private:
int load_model_file(std::string model_path,\
    std::map<UV, double>& classifier);

std::map<UV, double> _weak_classifier;

std::map<UV, double> _middle_classifier;

std::map<UV, double> _strong_classifier;

const double _weak_cluster = 61.5669;

const double _middle_cluster = 108.1497;

const double _strong_cluster = 145.1402;

std::vector<cv::Rect> _face_rects;

cv::CascadeClassifier _face_detector;

double _classifier_threshold;

cv::Mat _gray_frame;

cv::Mat _classify_frame;

cv::Mat _label_frame;

std::map<int, std::vector<cv::Point> > _hand_blobs;

};

} //hand_tracker

#endif
