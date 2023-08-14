#include "MainWindow.h"
#include "ui_MainWindow.h"

#include    <QFileDialog>

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

void MainWindow::do_stateChanged(QMediaPlayer::State state)
{//播放器状态变化，更新按钮状态
}

void MainWindow::do_playlistChanged(int position)
{//播放列表变化,更新当前播放文件名显示
}

void MainWindow::do_durationChanged(qint64 duration)
{//文件时长变化，更新进度显示
}

void MainWindow::do_currentChanged(qint64 position)
{//当前文件播放位置变化，更新进度显示
}

void MainWindow::on_btnAdd_clicked()
{//添加文件
}

void MainWindow::on_btnPlay_clicked()
{//播放
}

void MainWindow::on_btnPause_clicked()
{//暂停播放
}

void MainWindow::on_btnStop_clicked()
{//停止播放
}

