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

void MainWindow::on_actOpen_triggered()
{
}

void MainWindow::do_chartView_mouseMove(QPoint point)
{ //鼠标移动响应
}
