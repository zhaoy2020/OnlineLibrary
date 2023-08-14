#include "MainWindow.h"
#include "ui_MainWindow.h"

void MainWindow::iniGraph3D()
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


void MainWindow::on_actBar_ChangeValue_triggered()
{

}

void MainWindow::on_actData_Add_triggered()
{

}
