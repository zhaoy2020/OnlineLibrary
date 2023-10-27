#include "MainWindow.h"
#include "ui_MainWindow.h"

#include <QTextCharFormat>

MainWindow::MainWindow(QWidget *parent) :
   QMainWindow(parent),
   ui(new Ui::MainWindow)
{
   ui->setupUi(this);

   this->setCentralWidget(ui->textEdit);
}

MainWindow::~MainWindow()
{
   delete ui;
}


void MainWindow::on_actFile_New_triggered()
{

}

void MainWindow::on_actFile_Open_triggered()
{

}
