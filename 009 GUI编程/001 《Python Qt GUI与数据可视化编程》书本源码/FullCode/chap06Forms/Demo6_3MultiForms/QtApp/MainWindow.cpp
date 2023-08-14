#include "MainWindow.h"
#include "ui_MainWindow.h"

#include    <QPainter>
#include    <QFileDialog>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->tabWidget->setVisible(false);
    ui->tabWidget->clear();//清除所有页面
    ui->tabWidget->tabsClosable(); //Page有关闭按钮，可被关闭

    this->setCentralWidget(ui->tabWidget);

    this->setWindowState(Qt::WindowMaximized); //窗口最大化显示
    this->setAutoFillBackground(true);
}

MainWindow::~MainWindow()
{
   delete ui;
}

void MainWindow::do_docFileChanged(QString shortName)
{

}

void MainWindow::on_tabWidget_tabCloseRequested(int index)
{//关闭Tab

}


void MainWindow::on_actWidgetInsite_triggered()
{ //创建QFormDoc窗体，并在tabWidget中显示

}

void MainWindow::on_actWindowInsite_triggered()
{

}

void MainWindow::on_actWindow_triggered()
{

}


void MainWindow::on_actWidget_triggered()
{

}

void MainWindow::on_tabWidget_currentChanged(int index)
{

}
