#include "MainWindow.h"
#include "ui_MainWindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    this->setCentralWidget(ui->mdiArea);
    this->setWindowState(Qt::WindowMaximized); //窗口最大化显示
    ui->mainToolBar->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_actDoc_Open_triggered()
{//打开文件
}


void MainWindow::on_actDoc_New_triggered()
{ //新建文件
}

void MainWindow::on_mdiArea_subWindowActivated(QMdiSubWindow *arg1)
{//当前活动子窗口切换时
}

