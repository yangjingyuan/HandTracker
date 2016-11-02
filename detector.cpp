/***************************************************************************
 * 
 * Copyright (c) 2016 yangjingyuan. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file detector.cpp
 * @author yangjingyuan
 * @email hlbryang@163.com
 * @date 2016/11/02 14:16:35
 * @brief 
 *  
 **/

#include "detector.h"

namespace hand_tracker{

int detector::init_models(std::string model_dir){
    int ret = 0;

    std::string weak_model = model_dir + "/bayes_weak_light";
    std::string middle_model = model_dir + "/bayes_middle_light";
    std::string strong_model = model_dir + "/bayes_strong_light";
    std::string face_model = model_dir + "/face_model.xml";

    this->load_model_file(weak_model, _weak_classifier);
    this->load_model_file(middle_model, _middle_classifier);
    this->load_model_file(strong_model, _strong_classifier);
    this->_face_detector.load(face_model);

    return ret;
}

void detector::calculate_classifier_threshold(int Y){

    double dis_to_weak = std::abs(Y - _weak_cluster);
    double dis_to_middle = std::abs(Y - _middle_cluster);
    double dis_to_strong = std::abs(Y - _strong_cluster);
    
    if (dis_to_weak <= dis_to_middle) {
        _classifier_threshold = 0.4;
        if (dis_to_weak >= dis_to_strong) {
            _classifier_threshold = 0.25;
        }
    }else{
        _classifier_threshold = 0.25;
        if (dis_to_middle >= dis_to_strong) {
            _classifier_threshold = 0.25;
        }
    }
}


double detector::hand_classifier(int Y, int U ,int V){
    double dis_to_weak = std::abs(Y - _weak_cluster);
    double dis_to_middle = std::abs(Y - _middle_cluster);
    double dis_to_strong = std::abs(Y - _strong_cluster);
    double total_dis = dis_to_weak + dis_to_middle + dis_to_strong;

       
    double weak_prob = _weak_classifier[std::make_pair(U,V)];
    double middle_prob = _middle_classifier[std::make_pair(U,V)];
    double strong_prob = _strong_classifier[std::make_pair(U,V)];
    
    double weighted_prob = ((total_dis - dis_to_weak) / total_dis) * weak_prob\
    + ((total_dis - dis_to_middle) / total_dis) * middle_prob\
    + ((total_dis - dis_to_strong) / total_dis) * strong_prob;

    return weighted_prob;
}

int detector::check_blobs(cv::Mat frame){
    int ret = 0;

    cv::Mat I = frame.clone();
    cv::cvtColor(I, I, cv::COLOR_BGR2YCrCb);
    int channels = I.channels();
    int nRows = I.rows;
    int nCols = I.cols * channels;
    
    //define binary image
    cv::Mat binary_image(nRows,I.cols,CV_8UC1, cv::Scalar::all(0));

    int i,j;
    uchar* p;
    for (i = 0; i < nRows; ++i){
        p = I.ptr<uchar>(i);
        for ( j = 0; j < nCols; j+=3){
            int V = p[j+2];
            int U = p[j+1];
            int Y = p[j];

            calculate_classifier_threshold(Y);
            if (hand_classifier(Y, U, V) > _classifier_threshold){
                binary_image.at<uchar>(i,(j/3)) = 255;
            }
        }
    }

    this->_classify_frame = binary_image;    

    return ret;
}

void detector::remove_faces(){
    int row_num = _classify_frame.rows;
    int col_num = _classify_frame.cols;

    int i,j;
    uchar* p;
    for (i = 0; i < row_num; i++){
        p = _classify_frame.ptr<uchar>(i);
        for (j = 0; j < col_num; j++){
            if (p[j] == 255) {
                //reject face regions
                for (size_t k = 0; k < _face_rects.size(); k++){
                    cv::Point center(_face_rects[k].x + _face_rects[k].width*0.5, _face_rects[k].y + _face_rects[k].height*0.5 );
                    int hei = _face_rects[k].height * 0.6;
                    int wid = _face_rects[k].width * 0.6;
                    //out of the rect boundary
                    if (i > (center.y - hei) &&\
                        i < (center.y + hei) &&\
                        j > (center.x - wid) &&\
                        j < (center.x + wid)) {
                        p[j]=0;
                    }
                }
            }
        }
    }
}

void detector::two_pass_labeling(){
    /*
// 1. first pass
    
    _lableImg.release() ;
    _binImg.convertTo(_lableImg, CV_32SC1) ;
    
    int label = 1 ;  // start by 2
    std::vector<int> labelSet ;
    labelSet.push_back(0) ;   // background: 0
    labelSet.push_back(1) ;   // foreground: 1
    
    int rows = _binImg.rows - 1 ;
    int cols = _binImg.cols - 1 ;
    for (int i = 1; i < rows; i++)
    {
        int* data_preRow = _lableImg.ptr<int>(i-1) ;
        int* data_curRow = _lableImg.ptr<int>(i) ;
        for (int j = 1; j < cols; j++)
        {
            if (data_curRow[j] == 1)
            {
                vector<int> neighborLabels ;
                neighborLabels.reserve(2) ;
                int leftPixel = data_curRow[j-1] ;
                int upPixel = data_preRow[j] ;
                
                int leftUp = _lableImg.at<int>(i-1,j-1);
                int rightUp = _lableImg.at<int>(i-1,j+1);
                //8 connectively
                if ( leftPixel > 1)
                {
                    neighborLabels.push_back(leftPixel) ;
                }
                if (upPixel > 1)
                {
                    neighborLabels.push_back(upPixel) ;
                }
                if (leftUp > 1)
                {
                    neighborLabels.push_back(leftUp) ;
                }
                if (rightUp > 1)
                {
                    neighborLabels.push_back(rightUp) ;
                }
                
                if (neighborLabels.empty())
                {
                    labelSet.push_back(++label) ;  // assign to a new label
                    data_curRow[j] = label ;
                    labelSet[label] = label ;
                }
                else
                {
                    std::sort(neighborLabels.begin(), neighborLabels.end()) ;
                    int smallestLabel = neighborLabels[0] ;
                    data_curRow[j] = smallestLabel ;
                    
                    // save equivalence
                    for (size_t k = 1; k < neighborLabels.size(); k++)
                    {
                        int tempLabel = neighborLabels[k] ;
                        int& oldSmallestLabel = labelSet[tempLabel] ;
                        if (oldSmallestLabel > smallestLabel)
                        {
                            labelSet[oldSmallestLabel] = smallestLabel ;
                            oldSmallestLabel = smallestLabel ;
                        }
                        else if (oldSmallestLabel < smallestLabel)
                        {
                            labelSet[smallestLabel] = oldSmallestLabel ;
                        }
                    }
                }
            }
        }
    }
    
    // update equivalent labels
    // assigned with the smallest label in each equivalent label set
    for (size_t i = 2; i < labelSet.size(); i++)
    {
        int curLabel = labelSet[i] ;
        int preLabel = labelSet[curLabel] ;
        while (preLabel != curLabel)
        {
            curLabel = preLabel ;
            preLabel = labelSet[preLabel] ;
        }
        labelSet[i] = curLabel ;
    }
    
    
    // 2. second pass
    for (int i = 0; i < rows; i++)
    {
        int* data = _lableImg.ptr<int>(i) ;
        for (int j = 0; j < cols; j++)
        {
            int& pixelLabel = data[j] ;
            
            pixelLabel = labelSet[pixelLabel] ;
            //cout<<pixelLabel<<"\t";
            //record blobs
            if (pixelLabel!=0) {
                //blobRecorder[pixelLabel].push_back(make_pair(j+1,i+1));
                blobRecorder[pixelLabel].push_back(Point(j+1,i+1));
                //pointsMap[pixelLabel].push_back(Point(j+1,i+1));
            }
            
        }
        //cout<<endl;
    }
    return blobRecorder;
    */
}

cv::Mat detector::detect(cv::Mat frame){
    //cv::Mat gray_frame(frame.size(), CV_8UC1);

    cv::cvtColor(frame, _gray_frame, CV_BGR2GRAY);

    cv::equalizeHist(_gray_frame, _gray_frame);
    _face_detector.detectMultiScale(_gray_frame, _face_rects);
    check_blobs(frame);
    remove_faces();

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    morphologyEx(_classify_frame, _classify_frame, CV_MOP_CLOSE, kernel);

    return _classify_frame;
}



int detector::load_model_file(std::string model_path,\
    std::map<UV, double>& classifier){

    int ret = 0;
    classifier.clear();

    std::ifstream intput_model(model_path);
    if(!intput_model.is_open()){
        std::cout << "NO YUV MODEL FILE AVAILABLE"<< std::endl;
        return -1;
    }
    std::string* result_list = new std::string[9999999];
    std::string info;
    int wordCount=0;

    while (intput_model >> info) {
        result_list[wordCount++] = info;
    }

    for (int index=0; index < wordCount; index += 3) {
        double U,V,pro;
        U = atof(result_list[index].c_str());
        V = atof(result_list[index+1].c_str());
        pro = atof(result_list[index+2].c_str());
        classifier[std::make_pair(U,V)] = pro;
    }
    intput_model.close();
    return ret;
}

} //color_tracker

