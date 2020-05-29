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

Point topLeft, bottomRight;

Vec3b skinColorLOW(180, 255, 255), skinColorHIGH(0, 0, 0);
size_t roiSize = 8; //must be an even number

void detectFaceOpenCVDNN(Net net, Mat& frameOpenCVDNN);
void getSampleRegions(Mat& img, vector<Point>& samplePoints);
void defineSkin(Mat& imgHSV, vector<Point> samplePoints);
void processImg(Mat& img, Mat& imgHSV);

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

    vector<Point> sampleRegions;
    getSampleRegions(img, sampleRegions);

    Mat imgHSV;
    cvtColor(img, imgHSV, COLOR_BGR2HSV);

    defineSkin(imgHSV, sampleRegions);

    processImg(img, imgHSV);

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

           // cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),
           //     frameHeight / 150, 8);

            topLeft.x = x1;
            topLeft.y = y1;
            bottomRight.x = x2;
            bottomRight.y = y2;
        }
    }
}

void getSampleRegions(Mat& img, vector<Point>& samplePoints)
{
    Point leftCheek, rightCheek, forehead, chin;
    int diffX = bottomRight.x - topLeft.x;
    int diffY = bottomRight.y - topLeft.y;

    leftCheek.x = topLeft.x + (int)round(diffX /5.0);
    leftCheek.y = topLeft.y + (int)round(diffY * (4.0 / 7.0));
    rightCheek.x = topLeft.x + (int)round(diffX * (4.0 / 5.0));
    rightCheek.y = topLeft.y + (int)round(diffY * (4.0 / 7.0));
    forehead.x = topLeft.x + (int)round(diffX / 2.0);
    forehead.y = topLeft.y + (int)round(diffY * (1.0 / 5.0));
    chin.x = topLeft.x + (int)round(diffX / 2.0);
    chin.y = topLeft.y + (int)round(diffY * (15.0 / 16.0));
    //circle(img, leftCheek, 10, Scalar(0, 0, 0), 4);
    //circle(img, rightCheek, 10, Scalar(0, 0, 0), 4);
    //circle(img, forehead, 10, Scalar(0, 0, 0), 4);
    //circle(img, chin, 10, Scalar(0, 0, 0), 4);

    samplePoints.push_back(leftCheek);
    samplePoints.push_back(rightCheek);
    samplePoints.push_back(forehead);
    samplePoints.push_back(chin);

}

void defineSkin(Mat& imgHSV, vector<Point> samplePoints)
{
    vector<Mat> splitHSV;
    split(imgHSV, splitHSV);
    int q, r;
    for (int i = 0; i<samplePoints.size(); i++)
    {
        for (int m = 0; m < roiSize; m++)
        {
            for (int n = 0; n < roiSize; n++)
            {
                q = (samplePoints.at(i).x - roiSize / 2.0) + m;
                r = (samplePoints.at(i).y - roiSize / 2.0) + n;
                if (splitHSV[0].at<uchar>(q, r) < skinColorLOW[0])
                    skinColorLOW[0] = splitHSV[0].at<uchar>(q, r);
                else if (splitHSV[0].at<uchar>(q, r) > skinColorHIGH[0])
                    skinColorHIGH[0] = splitHSV[0].at<uchar>(q, r);
                if (splitHSV[1].at<uchar>(q, r) < skinColorLOW[1])
                    skinColorLOW[1] = splitHSV[1].at<uchar>(q, r);
                else if (splitHSV[1].at<uchar>(q, r) > skinColorHIGH[1])
                    skinColorHIGH[1] = splitHSV[1].at<uchar>(q, r);
                if (splitHSV[2].at<uchar>(q, r) < skinColorLOW[2])
                    skinColorLOW[2] = splitHSV[2].at<uchar>(q, r);
                else if (splitHSV[2].at<uchar>(q, r) > skinColorHIGH[2])
                    skinColorHIGH[2] = splitHSV[2].at<uchar>(q, r);
            }
        }
    }

    cout << "High values are : " << skinColorHIGH << endl;
    cout << "Low values are : " <<skinColorLOW << endl;
}

void processImg(Mat& img, Mat& imgHSV)
{
    //create mask
    Mat mask, maskInvert, maskBlurred, out;
    out = img.clone();
    skinColorHIGH[1] = skinColorHIGH[1] - 0;
    skinColorHIGH[2] = skinColorHIGH[2] - 20;
    skinColorLOW[1] = skinColorLOW[1] + 0;
    skinColorLOW[2] = skinColorLOW[2] + 10;
    inRange(imgHSV, skinColorLOW, skinColorHIGH, mask);
    //blur mask
    int blurVal01 = 39; //must be an odd number
    maskInvert = ~mask;
    GaussianBlur(maskInvert, maskBlurred, Size(blurVal01, blurVal01), 0, 0);
    //blur skin
    int blurVal02 = 9; //must be an odd number
    GaussianBlur(img, out, Size(blurVal02, blurVal02), 0, 0);
    //blur original image based on mask
    for (int y = 0; y < out.rows; ++y) {
        for (int x = 0; x < out.cols; ++x) {
            Vec3b pixelOrig = out.at<Vec3b>(y, x);
            Vec3b pixelBG = img.at<Vec3b>(y, x);
            float blurVal = maskBlurred.at<uchar>(y, x) / 255.0f;
            Vec3b pixelOut = blurVal * pixelBG + (1.0f - blurVal) * pixelOrig;

            out.at<Vec3b>(y, x) = pixelOut;
        }
    }

    imshow("mask", out);
}