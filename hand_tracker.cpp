/***************************************************************************
 * 
 * Copyright (c) 2016 yangjingyuan. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file hand_tracker.cpp
 * @author yangjingyuan
 * @email hlbryang@163.com
 * @date 2016/11/02 14:16:35
 * @brief 
 *  
 **/

#include "detector.h"

int test_hand_tracker(){
    cv::VideoCapture capture;

    capture.open(0);

    cv::Mat frame;

    hand_tracker::detector detector;
    detector.init_models("./models");
    char stop;

    while (true) {
        capture >> frame;
        if (!frame.empty()){
            cv::resize(frame, frame, cv::Size(320,240), 0, 0);
            cv::Mat det_ret = detector.detect(frame);
            cv::imshow("REAL-TIME VIDEO", det_ret);
        }

        stop = (char)cvWaitKey(50); // it wait n milionseconds until a keypress, if not return -1

        if (stop == 27){
            break;
        }
    }

    return 0;
}


int main(int argc, char** argv){

    test_hand_tracker();

    return 0;
}

