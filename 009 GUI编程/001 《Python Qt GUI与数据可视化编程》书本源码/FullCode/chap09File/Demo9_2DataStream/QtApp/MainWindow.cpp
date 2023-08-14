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

void MainWindow::on_btnFile_clicked()
{
}

void MainWindow::on_btnInt8_Write_clicked()
{
}

void MainWindow::on_btnInt8_Read_clicked()
{
}

void MainWindow::on_actSaveALL_triggered()
{
}

void MainWindow::on_actReadALL_triggered()
{

}

