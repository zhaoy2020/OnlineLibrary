#include "MainWindow.h"
#include "ui_MainWindow.h"


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


void MainWindow::on_actLang_CN_triggered()
{

}

void MainWindow::on_actLang_EN_triggered()
{

}
