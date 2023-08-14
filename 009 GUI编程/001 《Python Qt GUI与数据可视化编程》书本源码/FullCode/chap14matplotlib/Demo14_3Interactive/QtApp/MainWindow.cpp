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


void MainWindow::on_actTightLayout_triggered()
{

}

void MainWindow::on_actSetCursor_triggered()
{

}

void MainWindow::on_actScatterAgain_triggered()
{

}
