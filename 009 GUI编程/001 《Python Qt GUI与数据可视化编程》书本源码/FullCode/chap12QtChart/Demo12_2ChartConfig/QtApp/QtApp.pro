#-------------------------------------------------
#
# Project created by QtCreator 2016-12-23T00:04:20
#
#-------------------------------------------------

QT       += core gui

QT       += charts

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = QtApp
TEMPLATE = app


SOURCES += main.cpp\
    MainWindow.cpp \
    QWDialogPen.cpp

HEADERS  += \
    MainWindow.h \
    QWDialogPen.h

FORMS    += \
    MainWindow.ui \
    QWDialogPen.ui

RESOURCES += \
    res.qrc
