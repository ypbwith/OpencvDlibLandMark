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
#include <QtCharts/QChartView>
#include <QChartView>
#include <QLineSeries>
#include <QtCharts>
#include <playsound.h>

using namespace dlib;
using namespace std;
using namespace cv;

#define RATIO 1
#define SKIP_FRAMES 3
#define AlarmLevel 0.18
#define AlarmCount 5
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

    cv::polylines(img, points, isClosed, cv::Scalar(255, 0, 0), 1, 16);

}

void render_face(cv::Mat &img, const dlib::full_object_detection& d)
{
    DLIB_CASSERT
    (
        d.num_parts() == 68,
        "\n\t Invalid inputs were given to this function. "
        << "\n\t d.num_parts():  " << d.num_parts()
    );

    //draw_polyline(img, d, 0, 16);           // Jaw line
    //draw_polyline(img, d, 17, 21);          // Left eyebrow
    //draw_polyline(img, d, 22, 26);          // Right eyebrow
    draw_polyline(img, d, 27, 30);          // Nose bridge
    draw_polyline(img, d, 30, 35, true);    // Lower nose
    draw_polyline(img, d, 36, 41, true);    // Left eye
    draw_polyline(img, d, 42, 47, true);    // Right Eye
    //draw_polyline(img, d, 48, 59, true);    // Outer lip
    //draw_polyline(img, d, 60, 67, true);    // Inner lip

}

void ZeroRect(dlib::rectangle &rect)
{
    rect.set_top(0);
    rect.set_bottom(0);
    rect.set_left(0);
    rect.set_right(0);
}

int main(int argc, char **argv)
{
  QApplication a(argc, argv);


  playsound sound;

     sound.start();
     sound.soundName = new char[20];
     strcpy(sound.soundName,"alarm.wav");
     //sound.exit(0);

    double lineshow[100];
   //构建 series，作为图表的数据源，为其添加 6 个坐标点
   QLineSeries *series = new QLineSeries();


   // 构建图表
   QChart *chart = new QChart();
   chart->legend()->hide();  // 隐藏图例
   chart->addSeries(series);  // 将 series 添加至图表中
   chart->createDefaultAxes();  // 基于已添加到图表的 series 来创轴
   chart->setTitle("Simple line chart");  // 设置图表的标题

   QValueAxis *axisX = new QValueAxis; //定义X轴
    axisX->setRange(0, 10); //设置范围
    axisX->setLabelFormat("%g"); //设置刻度的格式
    axisX->setTitleText("X Axis"); //设置X轴的标题
    axisX->setGridLineVisible(true); //设置是否显示网格线
    axisX->setMinorTickCount(4); //设置小刻度线的数目
   // axisX->setLabelsVisible(false); //设置刻度是否显示

    QValueAxis *axisY = new QValueAxis;
    axisY->setRange(0, 0.5);
    axisY->setTitleText("Y Axis");
    axisY->setLabelFormat("%.2f");
    axisY->setGridLineVisible(true);

    chart->setAxisX(axisX, series);
    chart->setAxisY(axisY, series);

   // 构建 QChartView，并设置抗锯齿、标题、大小
   QChartView *chartView = new QChartView(chart);
   chartView->setRenderHint(QPainter::Antialiasing);
   chartView->setWindowTitle("Simple line chart");
   chartView->resize(500, 300);
   chartView->show();

    //Py_Initialize();//初始化python

    //				// 检查初始化是否成功
    //if (!Py_IsInitialized()) {
    //	return -1;
    //}

    //PyObject *pModule = NULL, *pFunc = NULL, *pArg = NULL;

    //pModule = PyImport_ImportModule("pysound");//引入模块

    //if (!pModule) {
    //	printf("can't find pysound.py");
    //	getchar();
    //	return -1;
    //}

    //pFunc = PyObject_GetAttrString(pModule, "sound_alarm");//直接获取模块中的函数

    //pArg = Py_BuildValue("(s)", "alarm.wav"); //参数类型转换，传递一个字符串。将c/c++类型的字符串转换为python类型，元组中的python类型查看python文档

    //PyEval_CallObject(pFunc, pArg); //调用直接获得的函数，并传递参数

    //Py_Finalize(); //释放python

    // Initialize the points of last frame
    //---------------------------------------------------------------------------------------------
    std::vector<cv::Point2f> last_object;
    for (int i = 0; i < 68; ++i) {
        last_object.push_back(cv::Point2f(0.0, 0.0));
    }

    //double scaling = 0.5;
    //int flag = -1;
    //int count = 0;

    //// Initialize measurement points
    //std::vector<cv::Point2f> kalman_points;
    //for (int i = 0; i < 68; i++) {
    //	kalman_points.push_back(cv::Point2f(0.0, 0.0));
    //}

    //// Initialize prediction points
    //std::vector<cv::Point2f> predict_points;
    //for (int i = 0; i < 68; i++) {
    //	predict_points.push_back(cv::Point2f(0.0, 0.0));
    //}

    //// Kalman Filter Setup (68 Points Test)
    //const int stateNum = 272;
    //const int measureNum = 136;

    //KalmanFilter KF(stateNum, measureNum, 0);
    //Mat state(stateNum, 1, CV_32FC1);
    //Mat processNoise(stateNum, 1, CV_32F);
    //Mat measurement = Mat::zeros(measureNum, 1, CV_32F);

    //// Generate a matrix randomly
    //randn(state, Scalar::all(0), Scalar::all(0.0));

    //// Generate the Measurement Matrix
    //KF.transitionMatrix = Mat::zeros(272, 272, CV_32F);
    //for (int i = 0; i < 272; i++) {
    //	for (int j = 0; j < 272; j++) {
    //		if (i == j || (j - 136) == i) {
    //			KF.transitionMatrix.at<float>(i, j) = 1.0;
    //		}
    //		else {
    //			KF.transitionMatrix.at<float>(i, j) = 0.0;
    //		}
    //	}
    //}

    ////!< measurement matrix (H) 观测模型
    //setIdentity(KF.measurementMatrix);

    ////!< process noise covariance matrix (Q)
    //setIdentity(KF.processNoiseCov, Scalar::all(1e-3));

    ////!< measurement noise covariance matrix (R)
    //setIdentity(KF.measurementNoiseCov, Scalar::all(1e-2));

    ////!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/  A代表F: transitionMatrix
    //setIdentity(KF.errorCovPost, Scalar::all(1));

    //randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));

    //cv::Mat prevgray, gray;

    //std::vector<cv::Point2f> prevTrackPts;
    //std::vector<cv::Point2f> nextTrackPts;
    //for (int i = 0; i < 68; i++) {
    //	prevTrackPts.push_back(cv::Point2f(0, 0));
    //	// nextTrackPts.push_back(cv::Point2f(0, 0));
    //}

    //-----------------------------------------------------------------------------------------------
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
        dlib::rectangle facePostion(0, 0, 0, 0);

        unsigned long im_mum=0;
        Mat grayscale_image, img_adaptive,img_EqualizeHist;
//        int block_size = 25;
//        double offset = 10;
//        int threadshould_type = CV_THRESH_BINARY;
//        int adaptive_method = CV_ADAPTIVE_THRESH_GAUSSIAN_C;
        double eyesClosedLevel_filter[100];
        while (1)
        {
            cap >> img;

            cvtColor(img, grayscale_image, CV_BGR2GRAY);

            //equalizeHist(grayscale_image,img_EqualizeHist);

            //adaptiveThreshold(grayscale_image, img_adaptive, 255, adaptive_method, threadshould_type, block_size, offset);


//            Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
//            filter2D(img_EqualizeHist, img_adaptive, CV_8UC3, kernel);
            //imshow("facefilter",img_adaptive);

            //cv::resize(img, img_small, cv::Size(), 1.0 / RATIO, 1.0 / RATIO);

            cv_image<dlib::bgr_pixel> cimg(img);
            cv_image<dlib::bgr_pixel> cimg_small(img);
            // Detect faces
            countframe++;
            /*if (countframe == SKIP_FRAMES)
            countframe = 0;*/

            //cout << countframe << endl;
            if ((trackflag == 0) || countframe == SKIP_FRAMES)
            {
                countframe = 0;
                faces = detector(cimg_small);
                if (faces.size() >0)
                {
                    facePostion = faces.at(0);
                    facebox_RATIO.x = faces.at(0).left();
                    facebox_RATIO.y = faces.at(0).top();
                    facebox_RATIO.width = faces.at(0).right() - facebox_RATIO.x;
                    facebox_RATIO.height = faces.at(0).bottom() - facebox_RATIO.y;

                    tracker.start_track(cimg_small, centered_rect(point(facebox_RATIO.x + 0.5*facebox_RATIO.width, facebox_RATIO.y + 0.5*facebox_RATIO.height), facebox_RATIO.width, facebox_RATIO.height));//centered_rect(point(93, 110), 38, 86));
                    //printf("--------111--\n");
                    trackflag = 1;
                }
            }

            else if (trackflag == 1 && countframe != SKIP_FRAMES)
            {

//                if (!facePostion.is_empty())
//                {
//                    printf("----------------------what  0----------------------------\n");
//                    tracker.update(cimg_small);
//                    facePostion = tracker.get_position();
//                }
//                else
//                {
//                    facePostion = dlib::rectangle(0, 0, 0, 0);
//                    trackflag = 0;
//                }
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

                //------------------------------KF--------------------------------------------------------------------
                //if (shapes.size() == 1)
                //{
                //	const full_object_detection& d = shapes[0];
                //	for (int i = 0; i < d.num_parts(); i++)
                //	{
                //		prevTrackPts[i].x = d.part(i).x();
                //		prevTrackPts[i].y = d.part(i).y();
                //	}

                //}

                //if (shapes.size() == 1) {
                //	const full_object_detection& d = shapes[0];
                //	/*for (int i = 0; i < d.num_parts(); i++) {
                //	cv::circle(face_2, cv::Point2f(int(d.part(i).x()), int(d.part(i).y())), 2, cv::Scalar(0, 255, 255), -1);
                //	std::cout << i << ": " << d.part(i) << std::endl;
                //	}*/
                //	for (int i = 0; i < d.num_parts(); i++) {
                //		kalman_points[i].x = d.part(i).x();
                //		kalman_points[i].y = d.part(i).y();
                //	}
                //}

                //// Kalman Prediction
                //// cv::Point2f statePt = cv::Point2f(KF.statePost.at<float>(0), KF.statePost.at<float>(1));
                //Mat prediction = KF.predict();
                //// std::vector<cv::Point2f> predict_points;
                //for (int i = 0; i < 68; i++) {
                //	predict_points[i].x = prediction.at<float>(i * 2);
                //	predict_points[i].y = prediction.at<float>(i * 2 + 1);
                //}

                //// Update Measurement
                //for (int i = 0; i < 136; i++) {
                //	if (i % 2 == 0) {
                //		measurement.at<float>(i) = (float)kalman_points[i / 2].x;
                //	}
                //	else {
                //		measurement.at<float>(i) = (float)kalman_points[(i - 1) / 2].y;
                //	}
                //}

                //measurement += KF.measurementMatrix * state;

                //// Correct Measurement
                //KF.correct(measurement);

                // Show 68-points utilizing kalman filter
                /*for (int i = 0; i < 68; i++) {
                cv::circle(face_kf, predict_points[i], 2, cv::Scalar(255, 0, 0), -1);
                }
                imshow("Frame2", face_3);*/
                //-------------------------------------------------------------------------------------------------


                for (int i = 36; i <= 41; i++)
                {

                    eyePoint.leftEye.x[i - 36] =shape.part(i).x() ; //0.9*shape.part(i).x() + 0.1*last_object[i].x;//predict_points[i].x;
                    eyePoint.leftEye.y[i - 36] =shape.part(i).y() ;//0.9*shape.part(i).y() + 0.1*last_object[i].y;//predict_points[i].y;

                    //cv::circle(facefilter, cv::Point(eyePoint.leftEye.x[i - 36], eyePoint.leftEye.y[i - 36]), 2, cv::Scalar(255, 0, 0), -1);
                }

                for (int i = 42; i <= 47; i++)
                {

                    eyePoint.rightEye.x[i - 42] = shape.part(i).x();//0.9*shape.part(i).x() + 0.1*last_object[i].x;//predict_points[i].x;
                    eyePoint.rightEye.y[i - 42] = shape.part(i).y();//0.9*shape.part(i).y() + 0.1*last_object[i].y;//predict_points[i].y;

                    //cv::circle(facefilter, cv::Point(eyePoint.rightEye.x[i - 42], eyePoint.rightEye.y[i - 42]), 2, cv::Scalar(255, 0, 0), -1);
                }


                // -----------------------------low pass-y=0.4now+0.6last---------------------------------------------


                if (shapes.size() == 1)
                {
                    im_mum++;

                    const full_object_detection& d = shapes[0];

                    for (int i = 0; i < d.num_parts(); i++)
                    {
                        last_object[i].x = d.part(i).x();
                        last_object[i].y = d.part(i).y();
                    }

                    Rect2d eyebox ;
                    eyebox.width = abs (d.part(36).x() - d.part(39).x() );
                    eyebox.height = eyebox.width;
                    eyebox.x = d.part(36).x();
                    eyebox.y = abs(d.part(36).y() - 0.5 * eyebox.width);

                    //cout << eyebox<<endl;

                    Mat img_clone =grayscale_image.clone();
                    Rect2d  dstbox = eyebox;

                    bool dstbox_flag = 0;

                    if(dstbox.x + dstbox.width<= 0)
                    {
                        dstbox_flag =1;
                    }
                    else if (dstbox.x<=0)
                    {

                        dstbox.width = dstbox.width + dstbox.x;

                        dstbox.x=0;
                    }

                    if(dstbox.y + dstbox.height<= 0)
                    {
                        dstbox_flag =1;

                    }
                    else if (dstbox.y<=0)
                    {
                        dstbox.height = dstbox.height + dstbox.y;
                        dstbox.y=0;
                    }

                    if(dstbox.x>img_clone.size().width)
                    {
                       dstbox_flag =1;
                    }
                    else if( dstbox.x + dstbox.width>img_clone.size().width)
                    {
                        dstbox.width = img_clone.size().width - dstbox.x;
                    }

                    if(dstbox.y + dstbox.height >img_clone.size().height)
                    {
                        dstbox_flag =1;
                    }
                    else if( dstbox.y + dstbox.height >img_clone.size().height)
                    {
                         dstbox.height = img_clone.size().width - dstbox.y;
                    }
                    if(dstbox_flag == 0)
                    {
                        cv::Mat eye_left (img_clone,dstbox );
                        cv::Mat eye_left_24x24;

                        dstbox_flag == 0;

                        resize(eye_left,eye_left_24x24,Size(64,64),0,0,CV_INTER_LINEAR);
                        char im_str[sizeof("eye_close/im%06d.jpg")];
                        sprintf(im_str,"eye_close/im%06d.jpg", im_mum);
                        //imwrite(im_str,eye_left_24x24); //c版本中的保存图片为cvSaveImage()函数，c++版本中直接与matlab的相似，imwrite()函数。
                        imshow( "face", eye_left_24x24 );
                    }

                }


                /*//cv::Mat facefilter = img.clone();
                Show 68-points utilizing kalman filter
                for (int i = 0; i < 68; i++) {
                cv::circle(facefilter, cv::Point(0.1*shape.part(i).x() + 0.9*last_object[i].x, 0.1*shape.part(i).y() + 0.9*last_object[i].y), 2, cv::Scalar(255, 0, 0), -1);
                }
                imshow("facefilter", facefilter);*/
                //----------------------------------------------------------------------------------------------------

                eyesClosedLevel =
                    (
                    (sqrt(pow(eyePoint.leftEye.x[1] - eyePoint.leftEye.x[5], 2) + pow(eyePoint.leftEye.y[1] - eyePoint.leftEye.y[5], 2)) +
                        sqrt(pow(eyePoint.leftEye.x[2] - eyePoint.leftEye.x[4], 2) + pow(eyePoint.leftEye.y[2] - eyePoint.leftEye.y[4], 2))
                        )
                        / (2 * sqrt(pow(eyePoint.leftEye.x[0] - eyePoint.leftEye.x[3], 2) + pow(eyePoint.leftEye.y[0] - eyePoint.leftEye.y[3], 2)))

                        +

                        (sqrt(pow(eyePoint.rightEye.x[1] - eyePoint.rightEye.x[5], 2) + pow(eyePoint.rightEye.y[1] - eyePoint.rightEye.y[5], 2)) +
                            sqrt(pow(eyePoint.rightEye.x[2] - eyePoint.rightEye.x[4], 2) + pow(eyePoint.rightEye.y[2] - eyePoint.rightEye.y[4], 2))
                            )
                        / (2 * sqrt(pow(eyePoint.rightEye.x[0] - eyePoint.rightEye.x[3], 2) + pow(eyePoint.rightEye.y[0] - eyePoint.rightEye.y[3], 2)))

                        ) / 2;

                 eyesClosedLevel_filter[0] = eyesClosedLevel ;
                 for (int i=14;i>=0;i--)
                 {
                     eyesClosedLevel += eyesClosedLevel_filter[i];

                     eyesClosedLevel_filter[i+1] = eyesClosedLevel_filter[i];

                 }

                 eyesClosedLevel =  eyesClosedLevel/15.0;

                series->clear();
                lineshow[0] = eyesClosedLevel;
                for (int i=99;i>=0;i--)
                {
                     lineshow[i+1] = lineshow[i];
                }

                for(double x=0;x<10;x+=0.1)
                {
                    //printf("%f",lineshow[int(x*10)]);
                    series->append(x,lineshow[int(x*10)]);
                }

              chartView->show();

                if (eyesClosedLevel < AlarmLevel)
                {
//                    alarmCount++;
//                    if (alarmCount > AlarmCount)
                    {
                        alarmCount = 0;
                        sound.start();
                        sound.soundName = new char[20];
                        strcpy(sound.soundName,"2.wav");
                        //sound.exit(0);
                        //pArg = Py_BuildValue("(s)", "alarm.wav"); //参数类型转换，传递一个字符串。将c/c++类型的字符串转换为python类型，元组中的python类型查看python文档

                        //PyEval_CallObject(pFunc, pArg); //调用直接获得的函数，并传递参数
                    }
                }
                else
                {
                    alarmCount = 0;
                }


                if (eyesClosedLevel > AlarmLevel +0.1)
                {

                    alarmCount = 0;
                    sound.start();
                    sound.soundName = new char[20];
                    strcpy(sound.soundName,"alarm.wav");
                }

                char PutString[20];
                sprintf(PutString, "eyeCloseLevel:%f\n", eyesClosedLevel);
                //printf(PutString);
                cv::putText(img, PutString, cv::Point(15, 15),
                    FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));


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

    //Py_Finalize(); //释放python

   return a.exec();

}
