#if 1
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>      //for imshow
#include <vector>
#include <iostream>
#include <iomanip>

#include "stats.h" // Stats structure definition
#include "utils.h" // Drawing and printing functions

using namespace std;
using namespace cv;

const double akaze_thresh = 3e-4; // AKAZE detection threshold set to locate about 1000 keypoints
const double ransac_thresh = 2.5f; // RANSAC inlier threshold
const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
const int bb_min_inliers = 100; // Minimal number of inliers to draw bounding box
const int stats_update_period = 10; // On-screen statistics are updated every 10 frames

namespace example {
class Tracker
{
	typedef pair<int, int> Match;

public:
    Tracker(Ptr<Feature2D> _detector, Ptr<DescriptorMatcher> _matcher) :
        detector(_detector),
        matcher(_matcher)
    {}

    void setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats);
    Mat process(const Mat frame, Stats& stats);
    Ptr<Feature2D> getDetector() {
        return detector;
    }
	bool decomposeH(const Mat& homography, const vector<KeyPoint>& vKeys1, const vector<KeyPoint>& vKeys2,
		const vector<Match>& vMatches12, vector<bool> &vbMatchesInliers, Mat& R21, Mat& t21);
	int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
		const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
		const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);
	void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
protected:
    Ptr<Feature2D> detector;
    Ptr<DescriptorMatcher> matcher;
    Mat first_frame, first_desc;
    vector<KeyPoint> first_kp;
    vector<Point2f> object_bb;
};

void Tracker::setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats)
{
    cv::Point *ptMask = new cv::Point[bb.size()];
    const Point* ptContain = { &ptMask[0] };
    int iSize = static_cast<int>(bb.size());
    for (size_t i=0; i<bb.size(); i++) {
        ptMask[i].x = static_cast<int>(bb[i].x);
        ptMask[i].y = static_cast<int>(bb[i].y);
    }
    first_frame = frame.clone();
    cv::Mat matMask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::fillPoly(matMask, &ptContain, &iSize, 1, cv::Scalar::all(255));
    detector->detectAndCompute(first_frame, matMask, first_kp, first_desc);
    stats.keypoints = (int)first_kp.size();
    drawBoundingBox(first_frame, bb);
    putText(first_frame, title, Point(0, 60), FONT_HERSHEY_PLAIN, 5, Scalar::all(0), 4);
    object_bb = bb;
    delete[] ptMask;
}

Mat Tracker::process(const Mat frame, Stats& stats)
{
    vector<KeyPoint> kp;
    Mat desc;
    detector->detectAndCompute(frame, noArray(), kp, desc);
    stats.keypoints = (int)kp.size();
	// aarontang fix bug: begin <<
	if (kp.size() == 0) {
		Mat res;
		hconcat(first_frame, frame, res);
		stats.inliers = 0;
		stats.ratio = 0;
		return res;
	}
	// end >>

    vector< vector<DMatch> > matches;
    vector<KeyPoint> matched1, matched2;
	vector<Match> vMatches12;
	vector<bool> vbMatched1(first_kp.size(), false);
    matcher->knnMatch(first_desc, desc, matches, 2);
    for(unsigned i = 0; i < matches.size(); i++) {
        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
            matched1.push_back(first_kp[matches[i][0].queryIdx]);
            matched2.push_back(      kp[matches[i][0].trainIdx]);
			vMatches12.push_back(make_pair(matches[i][0].queryIdx, matches[i][0].trainIdx));
			vbMatched1.at(matches[i][0].queryIdx) = true;
        }
    }
    stats.matches = (int)matched1.size();

    Mat inlier_mask, homography;
    vector<KeyPoint> inliers1, inliers2;
    vector<DMatch> inlier_matches;
    if(matched1.size() >= 4) {
        homography = findHomography(Points(matched1), Points(matched2),
                                    RANSAC, ransac_thresh, inlier_mask);
    }

    if(matched1.size() < 4 || homography.empty()) {
        Mat res;
        hconcat(first_frame, frame, res);
        stats.inliers = 0;
        stats.ratio = 0;
        return res;
    }
    for(unsigned i = 0; i < matched1.size(); i++) {
        if(inlier_mask.at<uchar>(i)) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            inlier_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }
    stats.inliers = (int)inliers1.size();
    stats.ratio = stats.inliers * 1.0 / stats.matches;

	Mat R21, t21;
	bool result = decomposeH(homography, first_kp, kp, vMatches12, vbMatched1, R21, t21);
	if (result) {
		cout << "R21:" << endl << R21 << endl;
		cout << "t21:" << endl << t21 << endl;
	}

    vector<Point2f> new_bb;
    perspectiveTransform(object_bb, new_bb, homography);
    Mat frame_with_bb = frame.clone();
    if(stats.inliers >= bb_min_inliers) {
        drawBoundingBox(frame_with_bb, new_bb);
    }
    Mat res;
    drawMatches(first_frame, inliers1, frame_with_bb, inliers2,
                inlier_matches, res,
                Scalar(255, 0, 0), Scalar(255, 0, 0));
    return res;
}

bool Tracker::decomposeH(const Mat& homography, const vector<KeyPoint>& vKeys1, const vector<KeyPoint>& vKeys2,
	const vector<Match>& vMatches12, vector<bool> &vbMatchesInliers, Mat& R21, Mat& t21)
{
	int N = 0;
	for (size_t i = 0, iend = vbMatchesInliers.size(); i<iend; i++)
		if (vbMatchesInliers[i])
			N++;

	// FIXME: hard code for test
	float fx = 601.155f;
	float fy = 601.155f;
	float cx = 320;
	float cy = 240;

	cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
	K.at<float>(0, 0) = fx;
	K.at<float>(1, 1) = fy;
	K.at<float>(0, 2) = cx;
	K.at<float>(1, 2) = cy;

	vector<Mat> Rs_decomp, ts_decomp, normals_decomp;
	int solutions = decomposeHomographyMat(homography, K, Rs_decomp, ts_decomp, normals_decomp);
	int bestGood = 0;
	int secondBestGood = 0;
	int bestSolutionIdx = -1;
	float bestParallax = -1;
	float Sigma = 1.0f;
	float Sigma2 = Sigma * Sigma;
	vector<cv::Point3f> bestP3D;
	vector<bool> bestTriangulated;

	//cout << "Decompose homography matrix computed from the camera displacement:" << endl << endl;
	for (int i = 0; i < solutions; i++) {
		float parallaxi;
		vector<cv::Point3f> vP3Di;
		vector<bool> vbTriangulatedi;
		Rs_decomp.at(i).convertTo(Rs_decomp.at(i), CV_32F);
		ts_decomp.at(i).convertTo(ts_decomp.at(i), CV_32F);
		int nGood = CheckRT(Rs_decomp.at(i), ts_decomp.at(i), vKeys1, vKeys2, vMatches12, vbMatchesInliers,
			K, vP3Di, 4.0*Sigma2, vbTriangulatedi, parallaxi);

		// 保留最优的和次优的
		if (nGood>bestGood) {
			secondBestGood = bestGood;
			bestGood = nGood;
			bestSolutionIdx = i;
			bestParallax = parallaxi;
			bestP3D = vP3Di;
			bestTriangulated = vbTriangulatedi;
		} else if (nGood>secondBestGood) {
			secondBestGood = nGood;
		}
	}

	/*if (secondBestGood<0.75*bestGood && bestParallax >= 1.0 && bestGood>50 && bestGood>0.9*N) {
		Rs_decomp.at(bestSolutionIdx).copyTo(R21);
		ts_decomp.at(bestSolutionIdx).copyTo(t21);
	}*/
	if (bestSolutionIdx != -1) {
		Rs_decomp.at(bestSolutionIdx).copyTo(R21);
		ts_decomp.at(bestSolutionIdx).copyTo(t21);
		return true;
	}

	return false;
}

int Tracker::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
	const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
	const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
	// Calibration parameters
	const float fx = K.at<float>(0, 0);
	const float fy = K.at<float>(1, 1);
	const float cx = K.at<float>(0, 2);
	const float cy = K.at<float>(1, 2);

	vbGood = vector<bool>(vKeys1.size(), false);
	vP3D.resize(vKeys1.size());

	vector<float> vCosParallax;
	vCosParallax.reserve(vKeys1.size());

	// Camera 1 Projection Matrix K[I|0]
	// 步骤1：得到一个相机的投影矩阵
	// 以第一个相机的光心作为世界坐标系
	cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
	K.copyTo(P1.rowRange(0, 3).colRange(0, 3));
	// 第一个相机的光心在世界坐标系下的坐标
	cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

	// Camera 2 Projection Matrix K[R|t]
	// 步骤2：得到第二个相机的投影矩阵
	cv::Mat P2(3, 4, CV_32F);
	R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
	t.copyTo(P2.rowRange(0, 3).col(3));
	P2 = K*P2;
	// 第二个相机的光心在世界坐标系下的坐标
	cv::Mat O2 = -R.t()*t;

	int nGood = 0;

	for (size_t i = 0, iend = vMatches12.size(); i<iend; i++)
	{
		if (!vbMatchesInliers[i])
			continue;

		// kp1和kp2是匹配特征点
		const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
		const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
		cv::Mat p3dC1;

		// 步骤3：利用三角法恢复三维点p3dC1
		Triangulate(kp1, kp2, P1, P2, p3dC1);

		if (!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
		{
			vbGood[vMatches12[i].first] = false;
			continue;
		}

		// Check parallax
		// 步骤4：计算视差角余弦值
		cv::Mat normal1 = p3dC1 - O1;
		float dist1 = cv::norm(normal1);

		cv::Mat normal2 = p3dC1 - O2;
		float dist2 = cv::norm(normal2);

		float cosParallax = normal1.dot(normal2) / (dist1*dist2);

		// 步骤5：判断3D点是否在两个摄像头前方

		// Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
		// 步骤5.1：3D点深度为负，在第一个摄像头后方，淘汰
		if (p3dC1.at<float>(2) <= 0 && cosParallax<0.99998)
			continue;

		// Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
		// 步骤5.2：3D点深度为负，在第二个摄像头后方，淘汰
		cv::Mat p3dC2 = R*p3dC1 + t;

		if (p3dC2.at<float>(2) <= 0 && cosParallax<0.99998)
			continue;

		// 步骤6：计算重投影误差

		// Check reprojection error in first image
		// 计算3D点在第一个图像上的投影误差
		float im1x, im1y;
		float invZ1 = 1.0 / p3dC1.at<float>(2);
		im1x = fx*p3dC1.at<float>(0)*invZ1 + cx;
		im1y = fy*p3dC1.at<float>(1)*invZ1 + cy;

		float squareError1 = (im1x - kp1.pt.x)*(im1x - kp1.pt.x) + (im1y - kp1.pt.y)*(im1y - kp1.pt.y);

		// 步骤6.1：重投影误差太大，跳过淘汰
		// 一般视差角比较小时重投影误差比较大
		if (squareError1>th2)
			continue;

		// Check reprojection error in second image
		// 计算3D点在第二个图像上的投影误差
		float im2x, im2y;
		float invZ2 = 1.0 / p3dC2.at<float>(2);
		im2x = fx*p3dC2.at<float>(0)*invZ2 + cx;
		im2y = fy*p3dC2.at<float>(1)*invZ2 + cy;

		float squareError2 = (im2x - kp2.pt.x)*(im2x - kp2.pt.x) + (im2y - kp2.pt.y)*(im2y - kp2.pt.y);

		// 步骤6.2：重投影误差太大，跳过淘汰
		// 一般视差角比较小时重投影误差比较大
		if (squareError2>th2)
			continue;

		// 步骤7：统计经过检验的3D点个数，记录3D点视差角
		vCosParallax.push_back(cosParallax);
		vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
		nGood++;

		if (cosParallax<0.99998)
			vbGood[vMatches12[i].first] = true;
	}

	// 步骤8：得到3D点中较大的视差角
	if (nGood>0)
	{
		// 从小到大排序
		sort(vCosParallax.begin(), vCosParallax.end());

		// trick! 排序后并没有取最大的视差角
		// 取一个较大的视差角
		size_t idx = min(50, int(vCosParallax.size() - 1));
		parallax = acos(vCosParallax[idx]) * 180 / CV_PI;
	}
	else
		parallax = 0;

	return nGood;
}

void Tracker::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
	// 在DecomposeE函数和ReconstructH函数中对t有归一化
	// 这里三角化过程中恢复的3D点深度取决于 t 的尺度，
	// 但是这里恢复的3D点并没有决定单目整个SLAM过程的尺度
	// 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变

	cv::Mat A(4, 4, CV_32F);

	A.row(0) = kp1.pt.x*P1.row(2) - P1.row(0);
	A.row(1) = kp1.pt.y*P1.row(2) - P1.row(1);
	A.row(2) = kp2.pt.x*P2.row(2) - P2.row(0);
	A.row(3) = kp2.pt.y*P2.row(2) - P2.row(1);

	cv::Mat u, w, vt;
	cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
	x3D = vt.row(3).t();
	x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
}
}

int main(int argc, char **argv)
{
    cerr << "Usage: " << endl
         << "akaze_track input_path" << endl
         << "  (input_path can be a camera id, like 0,1,2 or a video filename)" << endl;

    CommandLineParser parser(argc, argv, "{@input_path |0|input path can be a camera id, like 0,1,2 or a video filename}");
    string input_path = parser.get<string>(0);
    string video_name = input_path;

    VideoCapture video_in;

    if ( ( isdigit(input_path[0]) && input_path.size() == 1 ) )
    {
    int camera_no = input_path[0] - '0';
        video_in.open( camera_no );
    }
    else {
        video_in.open(video_name);
    }

    if(!video_in.isOpened()) {
        cerr << "Couldn't open " << video_name << endl;
        return 1;
    }

    Stats stats, akaze_stats, orb_stats;
    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->setThreshold(akaze_thresh);
    Ptr<ORB> orb = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    example::Tracker akaze_tracker(akaze, matcher);
    example::Tracker orb_tracker(orb, matcher);

    Mat frame;
    video_in >> frame;
    namedWindow(video_name, WINDOW_NORMAL);
    cv::resizeWindow(video_name, frame.cols, frame.rows);

    cout << "Please select a bounding box, and press any key to continue." << endl;
    vector<Point2f> bb;
    cv::Rect uBox = cv::selectROI(video_name, frame);
    bb.push_back(cv::Point2f(static_cast<float>(uBox.x), static_cast<float>(uBox.y)));
    bb.push_back(cv::Point2f(static_cast<float>(uBox.x+uBox.width), static_cast<float>(uBox.y)));
    bb.push_back(cv::Point2f(static_cast<float>(uBox.x+uBox.width), static_cast<float>(uBox.y+uBox.height)));
    bb.push_back(cv::Point2f(static_cast<float>(uBox.x), static_cast<float>(uBox.y+uBox.height)));

    akaze_tracker.setFirstFrame(frame, bb, "AKAZE", stats);
    orb_tracker.setFirstFrame(frame, bb, "ORB", stats);

    Stats akaze_draw_stats, orb_draw_stats;
    Mat akaze_res, orb_res, res_frame;
    int i = 0;
    for(;;) {
        i++;
        bool update_stats = (i % stats_update_period == 0);
        video_in >> frame;
        // stop the program if no more images
        if(frame.empty()) break;

        akaze_res = akaze_tracker.process(frame, stats);
        akaze_stats += stats;
        if(update_stats) {
            akaze_draw_stats = stats;
        }

        orb->setMaxFeatures(stats.keypoints);
        orb_res = orb_tracker.process(frame, stats);
        orb_stats += stats;
        if(update_stats) {
            orb_draw_stats = stats;
        }

        drawStatistics(akaze_res, akaze_draw_stats);
        drawStatistics(orb_res, orb_draw_stats);
        vconcat(akaze_res, orb_res, res_frame);
        cv::imshow(video_name, res_frame);
        if(waitKey(1)==27) break; //quit on ESC button
    }
    akaze_stats /= i - 1;
    orb_stats /= i - 1;
    printStatistics("AKAZE", akaze_stats);
    printStatistics("ORB", orb_stats);
    return 0;
}
#endif