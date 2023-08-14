#include "MainWindow.h"
#include "ui_MainWindow.h"

#include    <QFileDialog>
#include    <QMessageBox>

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

void MainWindow::do_currentRowChanged(const QModelIndex &current, const QModelIndex &previous)
{

}

void MainWindow::on_actOpenDB_triggered()
{//打开数据库

}

void MainWindow::on_actRecFirst_triggered()
{ //首记录

}

void MainWindow::on_actRecPrevious_triggered()
{ //前一条记录

}

void MainWindow::on_actRecNext_triggered()
{//后一条记录

}

void MainWindow::on_actRecLast_triggered()
{//最后一条记录

}
