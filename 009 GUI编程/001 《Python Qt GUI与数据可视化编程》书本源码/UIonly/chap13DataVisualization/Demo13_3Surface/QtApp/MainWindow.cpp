#include "MainWindow.h"
#include "ui_MainWindow.h"

#include    <QSplitter>


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


void MainWindow::on_actSurf_Color_triggered()
{

}

void MainWindow::on_comboCamera_currentIndexChanged(int index)
{

}
