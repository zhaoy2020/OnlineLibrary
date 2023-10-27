#include "MainWindow.h"
#include "ui_MainWindow.h"


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


void MainWindow::do_seriesMag_hovered(const QPointF &point, bool state)
{
}

void MainWindow::do_seriesPhase_hovered(const QPointF &point, bool state)
{
}

void MainWindow::on_actZoomReset_triggered()
{ //恢复原始大小
}

void MainWindow::on_actZoomIn_triggered()
{//放大
}

void MainWindow::on_actZoomOut_triggered()
{//缩小
}

void MainWindow::on_actOpen_triggered()
{
}

