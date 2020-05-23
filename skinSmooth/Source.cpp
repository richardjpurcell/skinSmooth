/*
 * File:    skinSmooth.cpp
 * Author:  Richard Purcell 
 * Date:    2020/05/14
 * Version: 1.0
 *
 * Purpose: This program smoothes skin tones in an image.
 *          It is part of Project 2 submission for Computer Vision 1.
 *
 * Notes:   https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
 *          
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;

const size_t inWidth = 200;
const size_t inHeight = 200;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.4;
const cv::Scalar meanVal(104.0, 177.0, 123.0);
const std::string caffeConfigFile = "./data/models/deploy.prototxt";
const std::string caffeWeightFile = "./data/models/res10_300x300_ssd_iter_140000_fp16.caffemodel";

const std::string tensorflowConfigFile = "./data/models/opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = "./data/models/opencv_face_detector_uint8.pb";

void detectFaceOpenCVDNN(Net net, Mat& frameOpenCVDNN);

int main(int argc, char** argv)
{
    string filename;

    if (argc != 2)
    {
        cout << "Usage:chromaKeyer.exe video_path background_path" << endl;
        cout << "ie:chromaKeyer ./greenscreen-asteroid.mp4 sampleBG1.png" << endl;
        cout << "Loading default video..." << endl;

        filename = "./hillary_clinton.jpg";
    }
    else
    {
        filename = argv[1];
    }

    Mat img = imread(filename);

#ifdef CAFFE
    Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
#else
    Net net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
#endif

    detectFaceOpenCVDNN(net, img);

    imshow("Image", img);
    waitKey(0);

    return 0;

}

void detectFaceOpenCVDNN(Net net, Mat& frameOpenCVDNN)
{
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;
    cv:Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor,
                        cv::Size(inWidth, inHeight), meanVal, false, false);
    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > confidenceThreshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),
                frameHeight / 150, 8);
        }
    }
}