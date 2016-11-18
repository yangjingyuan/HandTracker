/***************************************************************************
 * 
 * Copyright (c) 2016 yangjingyuan. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file hand_tracker.cpp
 * @author yangjingyuan
 * @email hlbryang@gmail.com
 * @date 2016/11/02 14:16:35
 * @brief 
 *  
 **/

#include "detector.h"
#include "tracker.h"

int test_hand_tracker(){
    cv::VideoCapture capture;

    capture.open(0);

    cv::Mat frame;

    hand_tracker::detector detector;
    detector.init_models("./models");
    char stop;

    hand_tracker::tracker tracker;

    while (true) {
        capture >> frame;
        if (!frame.empty()){
            cv::resize(frame, frame, cv::Size(320,240), 0, 0);
            std::map<int, std::vector<cv::Point> > hand_blobs = detector.detect(frame);
            //detector.display(hand_blobs);
            tracker.track(hand_blobs);
            tracker.display(frame);
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

