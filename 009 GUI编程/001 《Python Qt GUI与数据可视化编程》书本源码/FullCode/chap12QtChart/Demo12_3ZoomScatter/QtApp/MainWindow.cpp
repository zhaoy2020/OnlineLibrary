#include "MainWindow.h"
#include "ui_MainWindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setCentralWidget(ui->chartView);

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

void MainWindow::do_series_hovered(const QPointF &point, bool state)
{
}

void MainWindow::do_series_clicked(const QPointF &point)
{
}
