QT += core
QT += charts
QT -= core gui


CONFIG += c++11

TARGET = FatigueDetector
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES = main.cpp

CONFIG += no_keywords

INCLUDEPATH+=D:\ProgramFiles\python352\include \
             D:\ProgramFiles\python352

INCLUDEPATH+=D:\ProgramFiles\OpencvMyBuild\MyBuildLibVC14\install\include/opencv
INCLUDEPATH+=D:\ProgramFiles\OpencvMyBuild\MyBuildLibVC14\install\include/opencv2
INCLUDEPATH+=D:\ProgramFiles\OpencvMyBuild\MyBuildLibVC14\install\include
INCLUDEPATH+=D:\ProgramFiles\dlib-19.4\build\install\include


LIBS+=D:\ProgramFiles\OpencvMyBuild\MyBuildLibVC14\install\x64\vc14\lib\opencv_world320.lib
LIBS+=D:\ProgramFiles\dlib-19.4\build\install\lib\dlib.lib
LIBS+=D:\ProgramFiles\python352\libs\python35.lib
LIBS+=D:\ProgramFiles\OpencvMyBuild\MyBuildLibVC14\install\x64\vc14\lib\opencv_tracking320.lib

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
