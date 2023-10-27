#include "MainWindow.h"
#include "ui_MainWindow.h"

#include    <QFileDialog>


MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::do_cameraStateChanged(QCamera::State state)
{
}


void MainWindow::do_imageReady(bool ready)
{
}

void MainWindow::do_imageCaptured(int id, const QImage &preview)
{
}

void MainWindow::do_imageSaved(int id, const QString &fileName)
{
}

void MainWindow::on_actStartCamera_triggered()
{//开启摄像头
}

void MainWindow::on_actStopCamera_triggered()
{//关闭摄像头
}

void MainWindow::on_actCapture_triggered()
{//抓图 按钮
}
