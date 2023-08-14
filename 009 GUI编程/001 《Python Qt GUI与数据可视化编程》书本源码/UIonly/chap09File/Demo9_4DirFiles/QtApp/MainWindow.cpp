#include "MainWindow.h"
#include "ui_MainWindow.h"

void MainWindow::do_directoryChanged(const QString path)
{ //directoryChanged()信号的槽函数
}

void MainWindow::do_fileChanged(const QString path)
{//fileChanged()信号的槽函数
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

void MainWindow::on_btnOpenFile_clicked()
{
}

void MainWindow::on_btnOpenDir_clicked()
{
}

void MainWindow::on_btnFile_exists_clicked()
{
}

void MainWindow::on_btnFile_copy_clicked()
{
}
