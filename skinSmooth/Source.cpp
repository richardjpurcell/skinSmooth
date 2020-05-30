/*
 * File:    skinSmooth.cpp
 * Author:  Richard Purcell 
 * Date:    2020/05/26
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

//defines are standins for sliders that could be implemented in the future
#define ROISIZE 10   //the patch height and width for sampling skin
#define ERODESIZE 25 //the kernel size used to erode mask
#define ERODEITER 4  //number of times to iterate through erosion/dilation
#define HUEADJUST 11 //clamp hue in with positive values or out with negative
#define SATADJUST 5  //clamp sat in with positive values or out with negative
#define VALADJUST 5  //clamp val in with positive values or out with negative
#define MASKBLUR 39  //how much to blur the face mask (odd number only)
#define FACEBLUR 9   //how much to blur the face (odd number only)

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

void detectFaceOpenCVDNN(Net net, Mat& frameOpenCVDNN);
void getSampleRegions(Mat& img, vector<Point>& samplePoints);
void defineSkin(Mat& imgHSV, vector<Point> samplePoints);
void processImg(Mat& img, Mat& imgHSV);

int main(int argc, char** argv)
{
    string filename;

    if (argc != 2)
    {
        cout << "Usage:skinSmooth.exe image_path" << endl;
        cout << "ie:skinSmooth.exe ./hillary_clinton.jpg" << endl;
        cout << "Loading default image..." << endl;

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

    imshow("Original Image", img);

    waitKey(0);

    return 0;
}

/*
 * Name:        detectFaceOpenCVDNN
 * Purpose:     detect faces, creating rectangular bounding box
 * Arguments:   DNN, image to look for faces in
 * Output:      Upper left point and bottom right point of face bounding box
 * Modifies:    Point topLeft, bottomRight
 * Returns:     Void
 * Assumptions: None.
 * Bugs:        None.
 * Notes:       This code is originally from Computer Vision 1, Week 10,
 *              Deep Learning based Face Detection
 */
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

            //for testing
            // cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),
            //     frameHeight / 150, 8);

            topLeft.x = x1;
            topLeft.y = y1;
            bottomRight.x = x2;
            bottomRight.y = y2;
        }
    }
}

/*
 * Name:        getSampleRegions
 * Purpose:     select points on cheeks, forehead, and chin
 * Arguments:   img, vector of points
 * Output:      Adds location points to vector for cheeks, forehead, and chin
 * Modifies:    vector samplePoints
 * Returns:     Void
 * Assumptions: None.
 * Bugs:        None.
 * Notes:       None.
 */
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

/*
 * Name:        defineSkin
 * Purpose:     get high and low HSV values for skin tone
 * Arguments:   imgHSV, vector of points to sample
 * Output:      high and low HSV values
 * Modifies:    Vec3b skinColorLOW, skinColorHIGH
 * Returns:     Void
 * Assumptions: None.
 * Bugs:        None.
 * Notes:       None.
 */
void defineSkin(Mat& imgHSV, vector<Point> samplePoints)
{
    vector<Mat> splitHSV;
    split(imgHSV, splitHSV);
    int q, r;
    for (int i = 0; i<samplePoints.size(); i++)
    {
        for (int m = 0; m < ROISIZE; m++)
        {
            for (int n = 0; n < ROISIZE; n++)
            {
                q = (samplePoints.at(i).x - ROISIZE / 2.0) + m;
                r = (samplePoints.at(i).y - ROISIZE / 2.0) + n;
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
    //for testing
    //cout << "High values are : " << skinColorHIGH << endl;
    //cout << "Low values are : " <<skinColorLOW << endl;
}

/*
 * Name:        processImg
 * Purpose:     create mask of skin areas, merge blurred skin with original image
 * Arguments:   img, imgHSV
 * Output:      displays final image
 * Modifies:    None
 * Returns:     Void
 * Assumptions: None.
 * Bugs:        None.
 * Notes:       None.
 */
void processImg(Mat& img, Mat& imgHSV)
{
    //create mask
    Mat mask, maskInvert, maskBlurred, out;
    out = img.clone();
    skinColorHIGH[0] = skinColorHIGH[0] - HUEADJUST;
    skinColorHIGH[1] = skinColorHIGH[1] - SATADJUST;
    skinColorHIGH[2] = skinColorHIGH[2] - VALADJUST;
    skinColorLOW[0] = skinColorLOW[0] + HUEADJUST;
    skinColorLOW[1] = skinColorLOW[1] + SATADJUST;
    skinColorLOW[2] = skinColorLOW[2] + VALADJUST;
    inRange(imgHSV, skinColorLOW, skinColorHIGH, mask);

    //erode and dilate mask
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(ERODESIZE, ERODESIZE));
    erode(mask, kernel, ERODEITER);
    dilate(mask, kernel, ERODEITER);

    //blur mask
    maskInvert = ~mask;
    GaussianBlur(maskInvert, maskBlurred, Size(MASKBLUR, MASKBLUR), 0, 0);

    //blur skin
    GaussianBlur(img, out, Size(FACEBLUR, FACEBLUR), 0, 0);
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

    imshow("final image", out);
}