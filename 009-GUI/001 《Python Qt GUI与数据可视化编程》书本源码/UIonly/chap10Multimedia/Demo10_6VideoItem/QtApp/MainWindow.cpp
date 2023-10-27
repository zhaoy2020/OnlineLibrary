#include "MainWindow.h"
#include "ui_MainWindow.h"

#include    <QFileDialog>
#include    <QGraphicsScene>


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
{
}


void MainWindow::do_durationChanged(qint64 duration)
{
}

void MainWindow::do_positionChanged(qint64 position)
{
}

void MainWindow::on_btnPlay_clicked()
{
}

void MainWindow::on_btnPause_clicked()
{
}

void MainWindow::on_btnStop_clicked()
{
}
