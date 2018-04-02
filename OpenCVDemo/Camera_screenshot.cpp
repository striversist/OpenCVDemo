#if 0
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
	VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.

	if (!stream1.isOpened()) { //check if video device has been initialised
		cout << "cannot open camera";
		return -1;
	}

	//unconditional loop
	int fileIndex = 0;
	while (true) {
		Mat cameraFrame;
		stream1.read(cameraFrame);
		imshow("cam", cameraFrame);
		int key = waitKey(30);
		switch (key) {
			case 27:	// ESC
				return 0;
			case 32:	// Space
				imwrite(std::to_string(fileIndex++) + ".jpg", cameraFrame);
				break;
			default:
				break;
		}
	}
	return 0;
}
#endif