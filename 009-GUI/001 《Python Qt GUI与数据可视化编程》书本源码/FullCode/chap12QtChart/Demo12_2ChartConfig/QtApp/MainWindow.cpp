#include "MainWindow.h"
#include "ui_MainWindow.h"

#include    "QWDialogPen.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
   ui->setupUi(this);

   this->setCentralWidget(ui->splitter);
}

MainWindow::~MainWindow()
{
   delete ui;
}

void MainWindow::on_actDraw_triggered()
{

}

void MainWindow::on_btnTitleFont_clicked()
{

}
