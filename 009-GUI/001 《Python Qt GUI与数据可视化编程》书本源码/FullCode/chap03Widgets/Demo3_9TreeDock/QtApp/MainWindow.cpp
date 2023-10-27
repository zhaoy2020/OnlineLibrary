#include "MainWindow.h"
#include "ui_MainWindow.h"

#include    <QTreeWidgetItem>
#include    <QFileDialog>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    this->setCentralWidget(ui->scrollArea); //设置中心布局组件
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_actTree_AddFolder_triggered()
{

}

void MainWindow::on_treeFiles_currentItemChanged(QTreeWidgetItem *current, QTreeWidgetItem *previous)
{

}

void MainWindow::on_dockWidget_visibilityChanged(bool visible)
{

}

void MainWindow::on_dockWidget_topLevelChanged(bool topLevel)
{

}
