#include "MainWindow.h"
#include "ui_MainWindow.h"

#include    <QFileDialog>
#include    <QMessageBox>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setCentralWidget(ui->tableView);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::do_currentChanged(const QModelIndex &current, const QModelIndex &previous)
{//更新actPost和actCancel 的状态
}

void MainWindow::on_actOpenDB_triggered()
{
}

void MainWindow::on_actRecAppend_triggered()
{//添加记录
}

