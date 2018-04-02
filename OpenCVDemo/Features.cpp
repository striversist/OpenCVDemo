#if 0
#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;

int main() {

	Mat image;
	vector<Point2f> features;
	int maxCout = 500;//定义最大个数
	double minDis = 20;//定义最小距离
	double qLevel = 0.01;//定义质量水平

	image = imread("lena.jpg", 0);//读取为灰度图像
	goodFeaturesToTrack(image, features, maxCout, qLevel, minDis);
	for (int i = 0; i < features.size(); i++) {
		//将特征点画一个小圆出来--粗细为2
		circle(image, features[i], 3, Scalar(255), 2);
	}
	imshow("features", image);
	waitKey(0);

	return 0;
}
#endif