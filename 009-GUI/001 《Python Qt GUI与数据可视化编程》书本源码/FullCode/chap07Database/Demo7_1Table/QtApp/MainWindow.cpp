#include "MainWindow.h"
#include "ui_MainWindow.h"

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

void MainWindow::do_currentChanged(const QModelIndex &current, const QModelIndex &previous)
{//更新actPost和actCancel 的状态
}

void MainWindow::do_currentRowChanged(const QModelIndex &current, const QModelIndex &previous)
{
// 行切换时的状态控制
}

void MainWindow::on_actOpenDB_triggered()
{//打开数据表
}

void MainWindow::on_actRecAppend_triggered()
{//添加记录
}

