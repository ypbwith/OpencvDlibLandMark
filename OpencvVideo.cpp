#include <QCoreApplication>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstring>
#include <cstdlib>
using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
      cv::VideoCapture cap(0);
      Mat img ;

      while(1)
      {
          cap>>img;
          if (img.empty())
              break;

          // Display the resulting frame
          imshow("Frame", img);

          // Press  ESC on keyboard to exit
          char c = (char)waitKey(25);
          if (c == 27)
              break;


      }

   return a.exec();
}
