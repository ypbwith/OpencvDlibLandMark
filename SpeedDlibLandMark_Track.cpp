#include "stdafx.h"
#include <dlib/opencv.h>  
#include <opencv2/opencv.hpp>  
#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>  
#include <dlib/gui_widgets.h>  
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace dlib;
using namespace std;
using namespace cv;

#define RATIO 1
#define SKIP_FRAMES 300


void draw_polyline(cv::Mat &img, const dlib::full_object_detection& d, const int start, const int end, bool isClosed = false)
{
	std::vector <cv::Point> points;
	for (int i = start; i <= end; ++i)
	{
		points.push_back(cv::Point(d.part(i).x(), d.part(i).y()));
	}
	cv::polylines(img, points, isClosed, cv::Scalar(255, 0, 0), 2, 16);

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
int main()
{
	try
	{
		//cv::VideoCapture cap("The Great Dictator.mp4");
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

					//printf("----------------------what  1----------------------------\n");

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

				//cout<< (long)(faces[1].left())<<'\n';
				facebox.x = r.left();
				facebox.y = r.top();
				facebox.width = r.right() - facebox.x;
				facebox.height = r.bottom() - facebox.y;
				cv::rectangle(img, facebox, Scalar(255, 0, 0), 2, 1);


				// Landmark detection on full sized image
				full_object_detection shape = pose_model(cimg, r);
				shapes.push_back(shape);

				// Custom Face Render
				render_face(img, shape);

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
}