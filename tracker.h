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

//U and V elements in YUV color space
typedef std::pair<int, int> UV;

class tracker {	

public:
cv::Mat track(cv::Mat frame);

private:

};

} //hand_tracker

#endif
