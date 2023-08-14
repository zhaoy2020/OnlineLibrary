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

void MainWindow::on_chkBoxX_minorTick_clicked(bool checked)
{

}


void MainWindow::on_chkBoxX_minorTickBottom_clicked(bool checked)
{

}

void MainWindow::on_chkBoxX_minorTickTop_clicked(bool checked)
{

}

void MainWindow::on_chkBoxY_majorTickLeft_clicked(bool checked)
{

}

void MainWindow::on_chkBoxY_majorTickRight_clicked(bool checked)
{

}

void MainWindow::on_chkBoxAxisY_TickLabLeft_clicked(bool checked)
{

}

void MainWindow::on_chkBoxAxisY_TickLabRight_clicked(bool checked)
{

}

void MainWindow::on_chkBoxY_minorTickLeft_clicked(bool checked)
{

}

void MainWindow::on_chkBoxY_minorTickRight_clicked(bool checked)
{

}
