#include "MainWindow.h"
#include "ui_MainWindow.h"
#include    <QMenu>


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


void MainWindow::on_actList_Ini_triggered()
{

}

void MainWindow::on_actList_Insert_triggered()
{

}

void MainWindow::on_actList_Delete_triggered()
{

}
