#if 0
#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;

int main() {

	Mat image;
	vector<Point2f> features;
	int maxCout = 500;//����������
	double minDis = 20;//������С����
	double qLevel = 0.01;//��������ˮƽ

	image = imread("lena.jpg", 0);//��ȡΪ�Ҷ�ͼ��
	goodFeaturesToTrack(image, features, maxCout, qLevel, minDis);
	for (int i = 0; i < features.size(); i++) {
		//�������㻭һ��СԲ����--��ϸΪ2
		circle(image, features[i], 3, Scalar(255), 2);
	}
	imshow("features", image);
	waitKey(0);

	return 0;
}
#endif