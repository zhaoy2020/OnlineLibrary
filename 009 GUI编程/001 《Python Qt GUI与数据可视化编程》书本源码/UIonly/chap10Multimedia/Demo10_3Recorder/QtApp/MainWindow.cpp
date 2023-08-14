#include "MainWindow.h"
#include "ui_MainWindow.h"

#include    <QFileDialog>
#include    <QMessageBox>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::do_stateChanged(QMediaRecorder::State state)
{ //录音状态变化
}

void MainWindow::do_durationChanged(qint64 duration)
{//录音持续时间变化
}

void MainWindow::do_processBuffer(const QAudioBuffer &buffer)
{ //处理探测到的缓冲区
}

void MainWindow::on_actRecord_triggered()
{ //开始录音
}

void MainWindow::on_btnGetFile_clicked()
{ //设置保存文件
}

void MainWindow::on_actPause_triggered()
{ //暂停
}

void MainWindow::on_actStop_triggered()
{//停止
}
