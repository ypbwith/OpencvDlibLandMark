#include "stdafx.h"
#include <dlib/opencv.h>  
#include <opencv2/opencv.hpp>  
#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>  
#include <dlib/gui_widgets.h>  
#include "opencv2/opencv.hpp"
#include <iostream>
#include "Python.h"
#include <math.h>

using namespace dlib;
using namespace std;
using namespace cv;

#define RATIO 1
#define SKIP_FRAMES 10
#define AlarmLevel 0.05
#define AlarmCount 15
int alarmCount = 0;
double eyesClosedLevel;

struct leftEyePoint 
{
	double x[6];
	double y[6];
};

struct rightEyePoint
{
	double x[6];
	double y[6];
};

struct eyeKeyPoint 
{
	leftEyePoint leftEye;

	rightEyePoint rightEye;

};

eyeKeyPoint eyePoint;

void draw_polyline(cv::Mat &img, const dlib::full_object_detection& d, const int start, const int end, bool isClosed = false)
{
	std::vector <cv::Point> points;
	for (int i = start; i <= end; ++i)
	{
		points.push_back(cv::Point(d.part(i).x(), d.part(i).y()));
	}

	cv::polylines(img, points, isClosed, cv::Scalar(255, 0, 0), 1, 8);

}

void render_face(cv::Mat &img, const dlib::full_object_detection& d)
{
	DLIB_CASSERT
	(
		d.num_parts() == 68,
		"\n\t Invalid inputs were given to this function. "
		<< "\n\t d.num_parts():  " << d.num_parts()
	);

	draw_polyline(img, d, 0, 16);           // Jaw line
	draw_polyline(img, d, 17, 21);          // Left eyebrow
	draw_polyline(img, d, 22, 26);          // Right eyebrow
	draw_polyline(img, d, 27, 30);          // Nose bridge
	draw_polyline(img, d, 30, 35, true);    // Lower nose
	draw_polyline(img, d, 36, 41, true);    // Left eye
	draw_polyline(img, d, 42, 47, true);    // Right Eye
	draw_polyline(img, d, 48, 59, true);    // Outer lip
	draw_polyline(img, d, 60, 67, true);    // Inner lip

	

}

void ZeroRect(dlib::rectangle &rect)
{
	rect.set_top(0);
	rect.set_bottom(0);
	rect.set_left(0);
	rect.set_right(0);
}

int main(int argc, char* argv[])
{
	Py_Initialize();//初始化python  
	
	// 检查初始化是否成功  
	if (!Py_IsInitialized()) {
		return -1;
	}

	PyObject *pModule = NULL, *pFunc = NULL, *pArg = NULL;

	pModule = PyImport_ImportModule("pysound");//引入模块  
	
	if (!pModule) {
		printf("can't find pysound.py");
		getchar();
		return -1;
	}

	pFunc = PyObject_GetAttrString(pModule, "sound_alarm");//直接获取模块中的函数  

	//pArg = Py_BuildValue("(s)", "alarm.wav"); //参数类型转换，传递一个字符串。将c/c++类型的字符串转换为python类型，元组中的python类型查看python文档  

	//PyEval_CallObject(pFunc, pArg); //调用直接获得的函数，并传递参数  

	//Py_Finalize(); //释放python  

	// Initialize the points of last frame
	std::vector<cv::Point2f> last_object;
	for (int i = 0; i < 68; ++i) {
		last_object.push_back(cv::Point2f(0.0, 0.0));
	}

	double scaling = 0.5;
	int flag = -1;
	int count = 0;

	// Initialize measurement points
	std::vector<cv::Point2f> kalman_points;
	for (int i = 0; i < 68; i++) {
		kalman_points.push_back(cv::Point2f(0.0, 0.0));
	}

	// Initialize prediction points
	std::vector<cv::Point2f> predict_points;
	for (int i = 0; i < 68; i++) {
		predict_points.push_back(cv::Point2f(0.0, 0.0));
	}

	// Kalman Filter Setup (68 Points Test)
	const int stateNum = 272;
	const int measureNum = 136;

	KalmanFilter KF(stateNum, measureNum, 0);
	Mat state(stateNum, 1, CV_32FC1);
	Mat processNoise(stateNum, 1, CV_32F);
	Mat measurement = Mat::zeros(measureNum, 1, CV_32F);

	// Generate a matrix randomly
	randn(state, Scalar::all(0), Scalar::all(0.0));

	// Generate the Measurement Matrix
	KF.transitionMatrix = Mat::zeros(272, 272, CV_32F);
	for (int i = 0; i < 272; i++) {
		for (int j = 0; j < 272; j++) {
			if (i == j || (j - 136) == i) {
				KF.transitionMatrix.at<float>(i, j) = 1.0;
			}
			else {
				KF.transitionMatrix.at<float>(i, j) = 0.0;
			}
		}
	}

	//!< measurement matrix (H) 观测模型  
	setIdentity(KF.measurementMatrix);

	//!< process noise covariance matrix (Q)  
	setIdentity(KF.processNoiseCov, Scalar::all(1e-2));

	//!< measurement noise covariance matrix (R)  
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-2));

	//!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/  A代表F: transitionMatrix  
	setIdentity(KF.errorCovPost, Scalar::all(1));

	randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));

	cv::Mat prevgray, gray;

	std::vector<cv::Point2f> prevTrackPts;
	std::vector<cv::Point2f> nextTrackPts;
	for (int i = 0; i < 68; i++) {
		prevTrackPts.push_back(cv::Point2f(0, 0));
		// nextTrackPts.push_back(cv::Point2f(0, 0));
	}

	try
	{
		cv::VideoCapture cap(0);
		//image_window win;
		//cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);  
		//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);  
		// Load face detection and pose estimation models.  
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

		int countframe = 0;
		std::vector<dlib::rectangle> faces;
		// Grab and process frames until the main window is closed by the user. 

		correlation_tracker tracker;

		bool trackflag = 0;
		//while (!win.is_closed())
		// Grab a frame  
		cv::Mat img, img_small;

		// Define initial boundibg box
		Rect2d facebox(0, 0, 0, 0);
		Rect2d facebox_RATIO(0, 0, 0, 0);
		dlib::rectangle facePostion(0,0,0,0);

		while (1)
		{
			cap >> img;
			cv::resize(img, img_small, cv::Size(), 1.0 / RATIO, 1.0 / RATIO);

			cv_image<bgr_pixel> cimg(img);
			cv_image<bgr_pixel> cimg_small(img_small);
			// Detect faces  
			countframe++;
			/*if (countframe == SKIP_FRAMES)
				countframe = 0;*/

			//cout << countframe << endl;
			if ((trackflag == 0) || countframe == SKIP_FRAMES)
			{
				countframe = 0;
				faces = detector(cimg_small);
				if (faces.size() >0 )
				{
					facePostion = faces.at(0);
					facebox_RATIO.x = faces.at(0).left();
					facebox_RATIO.y = faces.at(0).top();
					facebox_RATIO.width = faces.at(0).right() - facebox_RATIO.x;
					facebox_RATIO.height = faces.at(0).bottom() - facebox_RATIO.y;

					tracker.start_track(cimg_small, centered_rect(point(facebox_RATIO.x + 0.5*facebox_RATIO.width, facebox_RATIO.y + 0.5*facebox_RATIO.height), facebox_RATIO.width, facebox_RATIO.height));//centered_rect(point(93, 110), 38, 86));

					trackflag = 1;
				}
			}

			else if (trackflag == 1 && countframe != SKIP_FRAMES)
			{
	           
				if(!facePostion.is_empty())
				{
					//printf("----------------------what  0----------------------------\n");
					//
					tracker.update(cimg_small);
					facePostion = tracker.get_position();
				}
				else
				{
					facePostion = dlib::rectangle(0,0,0,0);
					trackflag = 0;
				}
			}

			if (!facePostion.is_empty() && faces.size()>0)
			{
				// Find the pose of each face.  
				std::vector<full_object_detection> shapes;

			

				dlib::rectangle r(
					(long)(facePostion.left() * RATIO),
					(long)(facePostion.top() * RATIO),
					(long)(facePostion.right() * RATIO),
					(long)(facePostion.bottom() * RATIO)
				);

				////cout<< (long)(faces[1].left())<<'\n';
				//facebox.x = r.left();
				//facebox.y = r.top();
				//facebox.width = r.right() - facebox.x;
				//facebox.height = r.bottom() - facebox.y;
				//cv::rectangle(img, facebox, Scalar(255, 0, 0), 2, 1);


				// Landmark detection on full sized image
				full_object_detection shape = pose_model(cimg, r);
				shapes.push_back(shape);

				//------------------------------Filter-------------------------------------------
				// We cannot modify temp so we clone a new one
				cv::Mat face = img.clone();
				// We strict to detecting one face
				cv::Mat face_2 = img.clone();
				cv::Mat face_3 = img.clone();
				cv::Mat frame = img.clone();
				
			
				if (shapes.size() == 1)
				{
					const full_object_detection& d = shapes[0];
					for (int i = 0; i < d.num_parts(); i++)
					{
						prevTrackPts[i].x = d.part(i).x();
						prevTrackPts[i].y = d.part(i).y();
					}

				}

				// Simple Filter 低通滤波y=0.5*now+0.5last
			  /*if (shapes.size() == 1)
				{
					const full_object_detection& d = shapes[0];
					if (flag == -1) {
						for (int i = 0; i < d.num_parts(); i++) {
							cv::circle(face, cv::Point(d.part(i).x(), d.part(i).y()), 2, cv::Scalar(0, 0, 255), -1);
							std::cout << i << ": " << d.part(i) << std::endl;
						}
						flag = 1;
					}
					else {
						for (int i = 0; i < d.num_parts(); i++) {
							cv::circle(face, cv::Point2f(d.part(i).x() * 0.5 + last_object[i].x * 0.5, d.part(i).y() * 0.5 + last_object[i].y * 0.5), 2, cv::Scalar(0, 0, 255), -1);
							std::cout << i << ": " << d.part(i) << std::endl;
						}
					}
					for (int i = 0; i < d.num_parts(); i++) 
					{
						last_object[i].x = d.part(i).x();
						last_object[i].y = d.part(i).y();
					}
				}
				imshow("Frame3", face);*/

				// No Filter
		       	if (shapes.size() == 1) {
					const full_object_detection& d = shapes[0];
					/*for (int i = 0; i < d.num_parts(); i++) {
						cv::circle(face_2, cv::Point2f(int(d.part(i).x()), int(d.part(i).y())), 2, cv::Scalar(0, 255, 255), -1);
						std::cout << i << ": " << d.part(i) << std::endl;
					}*/
					for (int i = 0; i < d.num_parts(); i++) {
						kalman_points[i].x = d.part(i).x();
						kalman_points[i].y = d.part(i).y();
					}
				}

				// Kalman Prediction
				// cv::Point2f statePt = cv::Point2f(KF.statePost.at<float>(0), KF.statePost.at<float>(1));
				Mat prediction = KF.predict();
				// std::vector<cv::Point2f> predict_points;
				for (int i = 0; i < 68; i++) {
					predict_points[i].x = prediction.at<float>(i * 2);
					predict_points[i].y = prediction.at<float>(i * 2 + 1);
				}

				// Update Measurement
				for (int i = 0; i < 136; i++) {
					if (i % 2 == 0) {
						measurement.at<float>(i) = (float)kalman_points[i / 2].x;
					}
					else {
						measurement.at<float>(i) = (float)kalman_points[(i - 1) / 2].y;
					}
				}

				measurement += KF.measurementMatrix * state;

				// Correct Measurement
				KF.correct(measurement);

				// Show 68-points utilizing kalman filter
				for (int i = 0; i < 68; i++) {
					cv::circle(face_3, predict_points[i], 2, cv::Scalar(255, 0, 0), -1);
				}
				imshow("Frame2", face_3);
				//-----------------------------------------------------------------------------------------



				// Custom Face Render
				render_face(img, shape);

				for (int i = 36; i <= 41; i++)
				{

					eyePoint.leftEye.x[i - 36] = shape.part(i).x();
					eyePoint.leftEye.y[i - 36] = shape.part(i).y();
	            }

				for (int i = 42; i <= 47; i++)
				{

					eyePoint.leftEye.x[i - 42] = shape.part(i).x();
					eyePoint.leftEye.y[i - 42] = shape.part(i).y();
				}

				eyesClosedLevel =
					   (
					   ((pow(eyePoint.leftEye.x[1] - eyePoint.leftEye.x[5], 2) + pow(eyePoint.leftEye.y[1] - eyePoint.leftEye.y[5], 2)) +
						(pow(eyePoint.leftEye.x[2] - eyePoint.leftEye.x[4], 2) + pow(eyePoint.leftEye.y[2] - eyePoint.leftEye.y[4], 2))
						)
						/ (2 * (pow(eyePoint.leftEye.x[0] - eyePoint.leftEye.x[3], 2) + pow(eyePoint.leftEye.y[0] - eyePoint.leftEye.y[3], 2)))

						+

						((pow(eyePoint.leftEye.x[1] - eyePoint.leftEye.x[5], 2) + pow(eyePoint.leftEye.y[1] - eyePoint.leftEye.y[5], 2)) +
						(pow(eyePoint.leftEye.x[2] - eyePoint.leftEye.x[4], 2) + pow(eyePoint.leftEye.y[2] - eyePoint.leftEye.y[4], 2))
							)
						/ (2 * (pow(eyePoint.leftEye.x[0] - eyePoint.leftEye.x[3], 2) + pow(eyePoint.leftEye.y[0] - eyePoint.leftEye.y[3], 2)))

						) / 2;

				
				if (eyesClosedLevel < AlarmLevel)
				{
                    alarmCount++;
					if (alarmCount > AlarmCount)
					{
							alarmCount = 0;

							pArg = Py_BuildValue("(s)", "alarm.wav"); //参数类型转换，传递一个字符串。将c/c++类型的字符串转换为python类型，元组中的python类型查看python文档  

							PyEval_CallObject(pFunc, pArg); //调用直接获得的函数，并传递参数  
					}
				
				}
				else
				{
					alarmCount = 0;
				}

				char PutString[20] ;
				sprintf(PutString, "eyeCloseLevel:%f\n", eyesClosedLevel);
				printf(PutString);
				cv::putText(img, PutString, cv::Point(15, 15),
					FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

			}
		    

			//std::cout << "count:" << count << std::endl;
			//Display it all on the screen  
			//win.clear_overlay();
			//win.set_image(img);
			//win.add_overlay(render_face_detections(shapes));

			//If the frame is empty, break immediately
			if (img.empty())
				break;

			// Display the resulting frame
			imshow("Frame", img);

			// Press  ESC on keyboard to exit
			char c = (char)waitKey(25);
			if (c == 27)
				break;
		}
	}
	catch (serialization_error& e)
	{
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << "You can get it from the following URL: " << endl;
		cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}
	system("pause");

	Py_Finalize(); //释放python  

	return 0;

}
