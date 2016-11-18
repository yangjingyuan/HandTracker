/***************************************************************************
 * 
 * Copyright (c) 2016 yangjingyuan. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file tracker.cpp
 * @author yangjingyuan
 * @email hlbryang@163.com
 * @date 2016/11/02 14:16:35
 * @brief 
 *  
 **/

#include "tracker.h"

namespace hand_tracker{

double hypothesis::getCenterX(){
    return this->centerX;
}
double hypothesis::getCenterY(){
    return this->centerY;
}
double hypothesis::getAlpha(){
    return this->aplha;
}
double hypothesis::getBeta(){
    return this->beta;
}
double hypothesis::getThea(){
    return this->thea;
}
cv::Scalar hypothesis::getColor(){
    return cv::Scalar(this->red,this->green,this->blue);
}
void hypothesis::setParameters(double centerX,double centerY,double a,double b,double angle,double red,double green,double blue){
    this->centerX = centerX;
    this->centerY = centerY;
    this->aplha = a;
    this->beta = b;
    this->thea = angle;
    this->red = red;
    this->green = green;
    this->blue = blue;
}


void hypothesis::setColor(double red, double green, double blue){
    this->red = red;
    this->green = green;
    this->blue = blue;
}

double hypothesis::getRed(){
    return this->red;
}
double hypothesis::getBlue(){
    return this->blue;
}
double hypothesis::getGreen(){
    return this->green;
}

void hypothesis::setCenterX(double x){
    this->centerX = x;
}
void hypothesis::setCenterY(double y){
    this->centerY = y;
}


cv::Mat tracker::track(std::map<int, std::vector<cv::Point> > hand_blobs){
    if (track_time == 1) {
        std::map<int, std::vector<cv::Point> > trackBlobs = hypoGeneration(hand_blobs);
        hypoTracking(trackBlobs);
        t_2HypoList = removeHypothesis(hand_blobs);
        track_time++;
    }else if (track_time==2) {
        std::map<int, std::vector<cv::Point> > trackBlobs = hypoGeneration(hand_blobs);
        hypoTracking(trackBlobs);
        t_1HypoList = removeHypothesis(hand_blobs);
        track_time++;
    }else if (track_time==3) {
        hypoPredition();
        track_time=1;
    }else{
        std::map<int, std::vector<cv::Point> > trackBlobs = hypoGeneration(hand_blobs);
        hypoTracking(trackBlobs);
        removeHypothesis(hand_blobs);
        track_time++;
    }
}


void tracker::hypoPredition(){
    for(std::map<int,hypothesis>::iterator it=currHypoList.begin(); it!=currHypoList.end();it++){
        double c_x_1 = t_1HypoList[it->first].getCenterX();
        double c_x_2 = t_2HypoList[it->first].getCenterX();
        double c_y_1 = t_1HypoList[it->first].getCenterY();
        double c_y_2 = t_2HypoList[it->first].getCenterY();
        it->second.setCenterX(c_x_1+(c_x_1-c_x_2));
        it->second.setCenterY(c_y_1+(c_y_1-c_y_2));
    }
}




//detetmin hypothesis by every point in blob
bool tracker::blob_has_hypothesis(std::vector<cv::Point> blob){
    bool inside = false;
    for (std::vector<cv::Point>::iterator it=blob.begin(); it!=blob.end(); it++) {
        for (std::map<int,hypothesis>::iterator it2=currHypoList.begin(); it2!=currHypoList.end(); it2++) {
            double distance = disToEllpse(*it, it2->second);
            if (distance<1) {
                inside = true;
                return inside;
            }
        }
    }
    return inside;
}



//generate hypothesis
std::map<int, std::vector<cv::Point> > tracker::hypoGeneration(std::map<int, std::vector<cv::Point> > blobRecorder){
    int k=1;
    std::map<int, std::vector<cv::Point> > trackBlobs;
    double h_size = currHypoList.size();
    //initalize
    if(h_size==0) {
        int index=1;
        for (std::map<int, std::vector<cv::Point> >::iterator it=blobRecorder.begin(); it!=blobRecorder.end(); it++) {
            hypothesis hpo = getHypoByBlob(it->second);
            currHypoList[index++] = hpo;
        }
    }else{
        //blobs
        for (std::map<int, std::vector<cv::Point> >::iterator it=blobRecorder.begin(); it!=blobRecorder.end(); it++) {
            if(!blob_has_hypothesis(it->second)){
                hypothesis hypo = getHypoByBlob(it->second);
                currHypoList[++h_size] = hypo;
            }else{
                trackBlobs[k++] = it->second;
            }
            
        }
    }
    return trackBlobs;
}


void tracker::hypoTracking(std::map<int, std::vector<cv::Point> > blobs){
    std::map<int, std::vector<cv::Point> > updateBlobs;
    
    for (std::map<int, std::vector<cv::Point> >::iterator it=blobs.begin(); it!=blobs.end(); it++) {
        std::vector<cv::Point> points = it->second;
        for (std::vector<cv::Point>::iterator pt = points.begin(); pt!=points.end(); pt++) {
            cv::Point point(pt->x,pt->y);
            //loop for hypothesis
            double smallest_dis = 999;
            double smallest_hypothesis=-1;
            bool inside_hypothesis = false;
            for (std::map<int,hypothesis>::iterator hp = currHypoList.begin() ; hp!=currHypoList.end(); hp++) {
                    double distance = disToEllpse(point, hp->second);
                    if (distance<=1) {
                        updateBlobs[hp->first].push_back(point);
                        inside_hypothesis = true;
                    }else{
                        if (inside_hypothesis==false && distance<smallest_dis && distance>1) {
                            smallest_dis = distance;
                            smallest_hypothesis = hp->first;
                        }
                    }
            }
                if(!inside_hypothesis){
                    updateBlobs[smallest_hypothesis].push_back(point);
                }
    
        }
    }
    
    
    //vector update list
    for (std::map<int, std::vector<cv::Point> >::iterator up=updateBlobs.begin(); up!=updateBlobs.end(); up++) {
            hypothesis hypo = getHypoByBlob(up->second);
            hypothesis lastHypo = currHypoList[up->first];
            hypo.setColor(lastHypo.getRed(), lastHypo.getGreen(), lastHypo.getBlue());
            currHypoList[up->first] = hypo;
    }
    
    
    //deal with overlap issus
    for (std::map<int,hypothesis>::iterator it = currHypoList.begin(); it!=currHypoList.end(); it++) {
        std::map<int,int> supportList;
        for (std::map<int, std::vector<cv::Point> >::iterator pt=blobs.begin(); pt!=blobs.end(); pt++) {
            std::vector<cv::Point> points = pt->second;
            for (std::vector<cv::Point>::iterator p = points.begin(); p!=points.end(); p++) {
                double distance = disToEllpse(*p, it->second);
                if(distance<1){
                    supportList[pt->first]++;
                }
            }
        }
        //multilple support
        if (supportList.size()>=2) {
            int mainSupportID = 0;
            int supportNum=0;
            for (std::map<int,int>::iterator cm = supportList.begin(); cm!=supportList.end(); cm++) {
                if (cm->second>supportNum) {
                    supportNum = cm->second;
                    mainSupportID = cm->first;
                }
            }
            hypothesis mainHypo= getHypoByBlob(blobs[mainSupportID]);
            int red = currHypoList[it->first].getRed();
            int green = currHypoList[it->first].getGreen();
            int blue = currHypoList[it->first].getBlue();
            mainHypo.setColor(red, green, blue);
            currHypoList[it->first] = mainHypo;
        }
       
    }
}


//filter small blobs
std::map<int,hypothesis> tracker::removeHypothesis(std::map<int, std::vector<cv::Point> > blobs){
    std::set<int> remainingHypo;
    for (std::map<int,hypothesis>::iterator it=currHypoList.begin(); it!=currHypoList.end(); it++) {
        for (std::map<int, std::vector<cv::Point> >::iterator bl=blobs.begin(); bl!=blobs.end(); bl++) {
            std::vector<cv::Point> points = bl->second;
            for (std::vector<cv::Point>::iterator pt=points.begin(); pt!=points.end(); pt++) {
                double distance = disToEllpse(*pt,it->second);
                if(distance<=1){
                    remainingHypo.insert(it->first);
                }
            }
        }
    }
    //update hypothesis
    std::map<int,hypothesis> recentHypoList;
    for (std::set<int>::iterator it=remainingHypo.begin();it!=remainingHypo.end();it++){
        recentHypoList[*it] = currHypoList[*it];
    }
    currHypoList = recentHypoList;
    
    return currHypoList;
}

double tracker::disToEllpse(cv::Point p,hypothesis h){
    double center_x = h.getCenterX();
    double center_y = h.getCenterY();
    double a = h.getAlpha();
    double b = h.getBeta();
    double angle = h.getThea()* 3.1415 / 180;
    double x = p.x;
    double y = p.y;
    double tranX = (x - center_x)/a;
    double tranY = (y - center_y)/b;
    
    double distance = pow((cos(angle)*tranX - sin(angle)*tranY),2)+ pow((sin(angle)*tranX+cos(angle)*tranY), 2);
    
    //double distance = pow((cos(angle)*tranX + sin(angle)*tranY),2)+ pow(-(sin(angle)*tranX+cos(angle)*tranY), 2);
    
    distance = sqrt(distance);
  
    return distance;
}


//get parameters by using open cv function
hypothesis tracker::getHypoByBlob(std::vector<cv::Point> blob){

    int type=2;
    hypothesis hypo;
    if (type==1) {
        cv::RotatedRect box;
        if (blob.size()>100) {
            box = fitEllipse(blob);
            double r = 255 * (rand()/(1.0 + RAND_MAX));
            double g = 255 * (rand()/(1.0 + RAND_MAX));
            double b = 255 * (rand()/(1.0 + RAND_MAX));
            hypo.setParameters(box.center.x, box.center.y, 1.5*(box.size.width/2), 1.5*(box.size.height/2), box.angle, r, g, b);
            }
    }else{
        double pointsNum = blob.size();
        if (pointsNum>100) {
            
            cv::RotatedRect box;
            box = cv::fitEllipse(blob);
            
            
            double avg_x=0;
            double avg_y=0;
            for (std::vector<cv::Point>::iterator it=blob.begin(); it!=blob.end(); it++) {
                avg_x+=it->x;
                avg_y+=it->y;
            }
            //get center position
            avg_x = avg_x / pointsNum;
            avg_y = avg_y / pointsNum;
            
            double centerX = avg_x;
            double centerY = avg_y;
            
            //stand variance
            double var_x_x=0;
            double var_y_y=0;
            double var_x_y=0;
            //corvance matrix
            for (std::vector<cv::Point>::iterator it=blob.begin(); it!=blob.end(); it++) {
                double x = it->x;
                double y = it->y;
                var_x_x += pow((x-centerX),2);
                var_y_y += pow((y-centerY),2);
                var_x_y += (x-centerX)*(y-centerY);
            }
            var_x_x = var_x_x/pointsNum;
            var_y_y = var_y_y/pointsNum;
            var_x_y = var_x_y/pointsNum;
            
            //temp parameter
            /***********************QUESTION***************************/
            //double lamada = sqrt(pow((var_x_x+var_y_y),2) - 4*(var_x_x*var_x_y-pow(var_x_y,2)));
            double lamada = sqrt(pow((var_x_x-var_y_y),2) + 4*pow(var_x_y,2));
            /***********************QUESTION***************************/
            
            double l_1 = (var_x_x + var_y_y +lamada)/2;
            double l_2 = (var_x_x + var_y_y -lamada)/2;
            double a = sqrt(l_1);
            double b = sqrt(l_2);
            double thea = atan( (-var_x_y) / (l_1-var_y_y) ) * 180 / 3.1415;
            
            double red = 255 * (rand()/(1.0 + RAND_MAX));
            double green = 255 * (rand()/(1.0 + RAND_MAX));
            double blue = 255 * (rand()/(1.0 + RAND_MAX));
            //hypothesis
            //hypothesis hypo;
            hypo.setParameters(centerX, centerY, 2*a, 2*b, -thea, red, green, blue);
        }
    }
    
    return hypo;
}


//draw current hypothesis
void tracker::display(cv::Mat& draw_img){
    for (std::map<int,hypothesis>::iterator it=currHypoList.begin(); it!=currHypoList.end(); it++) {
        int thickness = 2;
        int lineType = 8;
        hypothesis hypo = it->second;
        double center_x = hypo.getCenterX();
        double center_y = hypo.getCenterY();
        double a = hypo.getAlpha();
        double b = hypo.getBeta();
        double thea = hypo.getThea();
        //random color
        cv::Scalar color = hypo.getColor();
        
        if(!isnan(center_x) && !isnan(center_y) && !isnan(a) && !isnan(b) && !isnan(thea)){
            cv::ellipse(draw_img, cv::Point( center_x, center_y ), cv::Size(a,b),thea,0,360,color,thickness,lineType);
        }else{
        }
    }
}


} //color_tracker

