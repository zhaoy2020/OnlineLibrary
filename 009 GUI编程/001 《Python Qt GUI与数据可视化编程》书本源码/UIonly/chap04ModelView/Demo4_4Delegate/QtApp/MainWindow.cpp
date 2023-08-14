#include "MainWindow.h"
#include "ui_MainWindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    setCentralWidget(ui->splitter); //
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::do_currentChanged(const QModelIndex &current, const QModelIndex &previous)
{ //选择单元格变化时的响应

}

void MainWindow::on_actOpen_triggered()
{ //打开文件
}

void MainWindow::on_actAppend_triggered()
{ //在表格最后添加行
}

void MainWindow::on_actDelete_triggered()
{

}
