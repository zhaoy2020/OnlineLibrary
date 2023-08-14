#include "MainWindow.h"
#include "ui_MainWindow.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::do_updateBlockInfo(qint64 blockSize,
                  int maxValue, int minValue )
{//显示缓冲区大小和数据块大小
}

void MainWindow::do_IO_readyRead()
{
}

void MainWindow::do_stateChanged(QAudio::State state)
{
}


void MainWindow::on_actStart_triggered()
{//开始音频输入

}

