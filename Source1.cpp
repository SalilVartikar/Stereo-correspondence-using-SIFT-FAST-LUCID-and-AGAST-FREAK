#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
void sift(Mat, Mat);
void fastLucid(Mat, Mat);
void agastFreak(Mat, Mat);
void matching(std::vector<KeyPoint>, std::vector<KeyPoint>, Mat, Mat, int);
void display(Mat, Mat);
Mat img1;
Mat img2;
double t1, t2, t;

int main(int argc, char** argv)
{
	img1 = imread("left.png");
	img2 = imread("right.png");
	
	sift(img1, img2);
	fastLucid(img1, img2);
	agastFreak(img1, img2);
	cout << "DONE" << "\n";
	
	waitKey(0);
}

void sift(Mat img1, Mat img2)
{
	cout << "\nSIFT";
	Ptr<SIFT> detector1 = SIFT::create();
	Ptr<SIFT> detector2 = SIFT::create();

	std::vector<KeyPoint> keypoints1, keypoints2;
	
	/*converting to grayscale*/
	cvtColor(img1, img1, CV_BGR2GRAY);
	cvtColor(img2, img2, CV_BGR2GRAY);

	/*detecting keypoints*/
	cout << "\nDetecting keypoints....";
	t1 = (double)getTickCount();
	detector1->detect(img1, keypoints1);
	detector2->detect(img2, keypoints2);
	t2 = (double)getTickCount();
	t = (t2 - t1) / getTickFrequency();
	cout << t << "s\n";
	std::cout << "No. of keypoints:\t" << keypoints1.size() << "\t" << keypoints2.size() << "\n";
	/*drawing keypoints*/
	Mat img_kp1; Mat img_kp2;
	drawKeypoints(img1, keypoints1, img_kp1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img2, keypoints2, img_kp2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imwrite("SIFT Keypoints 1.jpg", img_kp1);
	imwrite("SIFT Keypoints 2.jpg", img_kp2);

	Mat desc1, desc2;
	Ptr<SIFT> descriptor1 = SIFT::create();
	Ptr<SIFT> descriptor2 = SIFT::create();
	
	/*computeing descriptors*/
	cout << "Computing descriptors....";
	t1 = (double)getTickCount();
	detector1, descriptor1->compute(img1, keypoints1, desc1);
	detector2, descriptor2->compute(img2, keypoints2, desc2);
	t2 = (double)getTickCount();
	t = (t2 - t1) / getTickFrequency();
	cout << t << "s\n";

	matching(keypoints1, keypoints2, desc1, desc2, 1);
}

void fastLucid(Mat img1, Mat img2)
{
	cout << "\nFAST + LUCID";
	Ptr<FastFeatureDetector> detector1 = FastFeatureDetector::create();
	Ptr<FastFeatureDetector> detector2 = FastFeatureDetector::create();

	std::vector<KeyPoint> keypoints1, keypoints2;

	/*detecting keypoints*/
	cout << "\nDetecting keypoints....";
	t1 = (double)getTickCount();
	detector1->detect(img1, keypoints1);
	detector2->detect(img2, keypoints2);
	t2 = (double)getTickCount();
	t = (t2 - t1) / getTickFrequency();
	cout << t << "s\n";
	std::cout << "No. of keypoints:\t" << keypoints1.size() << "\t" << keypoints2.size() << "\n";

	/*drawing keypoints*/
	Mat img_kp1, img_kp2;
	drawKeypoints(img1, keypoints1, img_kp1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img2, keypoints2, img_kp2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imwrite("FAST keypoints 1.jpg", img_kp1);
	imwrite("FAST keypoints 2.jpg", img_kp2);

	Mat desc1, desc2;
	Ptr<LUCID> descriptor1 = LUCID::create();
	Ptr<LUCID> descriptor2 = LUCID::create();
	
	/*computeing descriptors*/
	cout << "Computing descriptors....";
	t1 = (double)getTickCount();
	detector1, descriptor1->compute(img1, keypoints1, desc1);
	detector2, descriptor2->compute(img2, keypoints2, desc2);
	t2 = (double)getTickCount();
	t = (t2 - t1) / getTickFrequency();
	cout << t << "s\n";

	matching(keypoints1, keypoints2, desc1, desc2, 2);
}

void agastFreak(Mat img1, Mat img2)
{
	cout << "\nAGAST + FREAK";
	Ptr<AgastFeatureDetector> detector1 = AgastFeatureDetector::create();
	Ptr<AgastFeatureDetector> detector2 = AgastFeatureDetector::create();

	/*converting to grayscale*/
	cvtColor(img1, img1, CV_BGR2GRAY);
	cvtColor(img2, img2, CV_BGR2GRAY);

	std::vector<KeyPoint> keypoints1, keypoints2;
	
	/*detecting keypoints*/
	cout << "\nDetecting keypoints....";
	t1 = (double)getTickCount();
	detector1->detect(img1, keypoints1);
	detector2->detect(img2, keypoints2);
	t2 = (double)getTickCount();
	t = (t2 - t1) / getTickFrequency();
	cout << t << "s\n";
	std::cout << "No. of keypoints:\t" << keypoints1.size() << "\t" << keypoints2.size() << "\n";

	/*drawing keypoints*/
	Mat img_kp1, img_kp2;
	drawKeypoints(img1, keypoints1, img_kp1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img2, keypoints2, img_kp2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imwrite("AGAST keypoints 1.jpg", img_kp1);
	imwrite("AGAST keypoints 2.jpg", img_kp2);

	Ptr<FREAK> descriptor1 = FREAK::create();
	Ptr<FREAK> descriptor2 = FREAK::create();
	Mat desc1, desc2;

	/*computeing descriptors*/
	cout << "Computing descriptors....";
	t1 = (double)getTickCount();
	descriptor1->compute(img1, keypoints1, desc1);
	descriptor2->compute(img2, keypoints2, desc2);
	t2 = (double)getTickCount();
	t = (t2 - t1) / getTickFrequency();
	cout << t << "s\n";

	matching(keypoints1, keypoints2, desc1, desc2, 3);
}

void matching(std::vector<KeyPoint>keypoints1, std::vector<KeyPoint>keypoints2, Mat desc1, Mat desc2, int n)
{
	/*using brute-force matcher to find matches between the 2 descriptors*/
	cout << "Matching....";
	t1 = (double)getTickCount();
	cv::BFMatcher bf_matcher = cv::BFMatcher(cv::NORM_L2, true);
	std::vector< DMatch > matches;
	bf_matcher.match(desc1, desc2, matches);

	/*finding the minimum and maximum distance between the matches*/
	double min_dist = 20;
	for (int i = 0; i < matches.size(); i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist)
			min_dist = dist;
	}
	t2 = (double)getTickCount();
	t = (t2 - t1) / getTickFrequency();
	cout << t << "s\n";

	std::vector< DMatch > good;
	
	/*filtering matches based on thresholds*/
	for (int i = 0; i < matches.size(); i++)
	{
		if (n == 1)
		{
			if (matches[i].distance <= max(2 * min_dist, 1.0))
			{
				good.push_back(matches[i]);
			}
		}
		else if (n == 2)
		{
			if (matches[i].distance <= max(2 * min_dist, 0.5))
			{
				good.push_back(matches[i]);
			}
		}
		else
		{
			if (matches[i].distance <= max(2 * min_dist, 50.0))
			{
				good.push_back(matches[i]);
			}
		}
	}

	/*drawing the good matches*/
	cout << "Drawing image with good matches....";
	t1 = (double)getTickCount();
	Mat img;
	cv::drawMatches(img1, keypoints1, img2, keypoints2, good, img, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	if (n == 1)
	{
		imwrite("SIFT_GoodMatches.png", img);
	}
	else if (n == 2)
	{
		imwrite("FAST_LUCID_GoodMatches.png", img);
	}
	else
	{
		imwrite("AGAST_FREAK_GoodMatches.png", img);
	}
	t2 = (double)getTickCount();
	t = (t2 - t1) / getTickFrequency();
	cout << t << "s\n";

	/*calculating disparity*/
	float disparity = 0;
	for (int i = 0; i < (int)good.size(); i++)
	{
		disparity = disparity + good[i].distance;
	}
	cout << "Disparity: " << disparity/(int) good.size() << "\n";

	Mat dmap = Mat::zeros(img1.size(), CV_8U);
	int intensity = 0;
	int temp = 0;
	int left = 0, right = 0;

	/*adjusting intensity*/
	for (int i = 0; i < dmap.rows; i++)
	{
		for (int j = 0; j < dmap.cols; j++)
		{
			temp = max(j - (int)disparity, 0);
			left = (int)img1.at<uchar>(i, j);
			right = (int)img2.at<uchar>(i, temp);
			dmap.at<uchar>(i, j) = (uchar)(int)abs(left - right);
			if ((int)abs(left - right) > intensity)
				intensity = (int)abs(left - right);
		}
	}

	float adjust = (float) 255.0 / intensity;
	for (int i = 0; i < dmap.rows; i++)
	{
		for (int j = 0; j < dmap.cols; j++)
		{
			temp = (int)dmap.at<uchar>(i, j);
			dmap.at<uchar>(i, j) = (uchar)((int)(temp*adjust));
		}
	}

	double error = 0;

	/*comparing with groundtruth*/
	Mat groundtruth = imread("truth.png");
	for (int i = 0; i < (int)good.size(); i++)
	{
		temp = (int)keypoints1[good[i].queryIdx].pt.y;
		left = (int)dmap.at<uchar>(temp, (int)keypoints1[good[i].queryIdx].pt.x);
		right = (int)groundtruth.at<uchar>(temp, (int)keypoints1[good[i].queryIdx].pt.x);
		error = error + pow(abs(left - right), 2);
	}

	float mse = (float)(error / (int)keypoints1.size());
	double rmse = (double)sqrtf(mse);
	cout << "RMSE = " << rmse << "\n";

	float ep = 100*(matches.size() - good.size()) / matches.size();
	cout << "Error percent = " << ep << "\n";
}