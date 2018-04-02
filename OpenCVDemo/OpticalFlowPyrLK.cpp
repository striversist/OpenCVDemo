#if 1
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/features2d.hpp>
#include "opencv2/calib3d.hpp"
#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;

template <typename T>
void reduceVector(vector<T> &v, vector<uchar> status)
{
	int j = 0;
	for (int i = 0; i < int(v.size()); i++)
		if (status[i])
			v[j++] = v[i];
	v.resize(j);
}

int main()
{
	Mat img1;
	Mat img2;
	vector<Point2f> features;
	vector<Point2f> features_after;
	vector<Point2f> moved;
	vector<DMatch> good_matches;
	vector<uchar> status;
	vector<float> err;
	int maxCout = 300;//����������
	double minDis = 20;//������С����
	double qLevel = 0.01;//��������ˮƽ
	
	//��ȡ����ͼ��---����֡
	img1 = imread("I5.jpg", 0);//��ȡΪ�Ҷ�ͼ��
	img2 = imread("I6.jpg", 0);
	//����һ֡��������
	goodFeaturesToTrack(img1, features, maxCout, qLevel, minDis);
	//������ڶ�֡��������
	calcOpticalFlowPyrLK(img1, img2, features, features_after, status, err);
	//�б���Щ�����˶���������
	for (int i = 0; i < features_after.size(); i++) {
		//״̬Ҫ��1����������Ҫ�ƶ��µ���Щ��
		if (status[i] && ((abs(features[i].x - features_after[i].x) +
			abs(features[i].y - features_after[i].y)) > 4)) {
			moved.push_back(features_after[i]);
			good_matches.push_back(DMatch(i, i, 0));
		}
	}
	
	// ��ͼ��1��������
	Mat img2_copy = img2.clone();
	for (int i = 0; i < moved.size(); i++) {
		//�������㻭һ��СԲ����--��ϸΪ2
		circle(img2_copy, moved[i], 3, Scalar(255), 2);
	}
	imshow("features", img2_copy);

	// ������ͼ�Ĳ�������
	Mat diff;
	addWeighted(img1, 1, img2, -1, 0, diff);
	imshow("diff", diff);
	
	vector<KeyPoint> kp1, kp2;
	Mat res;
	// ��Raw matches
	KeyPoint::convert(features, kp1);
	KeyPoint::convert(features_after, kp2);
	drawMatches(img1, kp1, img2, kp2, good_matches, res);
	imshow("raw matches", res);

	// ��refine matches(inliners)
	vector<Point2f> pts1(features), pts2(features_after);
	reduceVector(pts1, status);
	reduceVector(pts2, status);
	vector<uchar> status2;
	findFundamentalMat(pts1, pts2, FM_RANSAC, 1.0, 0.99, status2);
	reduceVector(pts1, status2);
	reduceVector(pts2, status2);
	KeyPoint::convert(pts1, kp1);
	KeyPoint::convert(pts2, kp2);
	good_matches.clear();
	for (int i = 0; i < kp1.size(); ++i) {
		good_matches.push_back(DMatch(i, i, 0));
	}
	drawMatches(img1, kp1, img2, kp2, good_matches, res);
	imshow("refine matches", res);

	waitKey(0);
	return 0;
}
#endif