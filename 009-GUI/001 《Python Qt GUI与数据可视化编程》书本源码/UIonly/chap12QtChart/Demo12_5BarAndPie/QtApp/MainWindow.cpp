#include "MainWindow.h"
#include "ui_MainWindow.h"


void MainWindow::do_calcuAverage(QStandardItem *item)
{ //自动计算平均分
}

void MainWindow::do_PieSliceHighlight(bool show)
{ //鼠标移入、移出时触发hovered()信号，动态设置setExploded()效果
}

void MainWindow::do_PieSeriesHovered(QPieSlice *slice, bool state)
{
}

void MainWindow::do_legendMarkerClicked()
{
}

void MainWindow::do_barHovered(bool status, int index, QBarSet *barset)
{
}

void MainWindow::do_barClicked(int index, QBarSet *barset)
{
}

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
