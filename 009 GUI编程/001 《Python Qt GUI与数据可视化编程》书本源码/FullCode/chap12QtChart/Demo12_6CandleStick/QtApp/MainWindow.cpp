#include "MainWindow.h"
#include "ui_MainWindow.h"

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

void MainWindow::do_LegendMarkerClicked()
{//点击图例的marker的响应

}

void MainWindow::do_chartView_mouseMove(QPoint point)
{ //鼠标移动响应
}

void MainWindow::do_candleClicked(QCandlestickSet *dataSet)
{
}

void MainWindow::do_candleHovered(bool status, QCandlestickSet *dataSet)
{
}
