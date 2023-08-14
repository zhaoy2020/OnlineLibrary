#include "MainWindow.h"
#include "ui_MainWindow.h"

#include    <QFileDialog>
#include    <QMessageBox>
#include    "QWDialogData.h"

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

void MainWindow::do_currentRowChanged(const QModelIndex &current, const QModelIndex &previous)
{
}

void MainWindow::on_actOpenDB_triggered()
{
}

void MainWindow::on_actRecInsert_triggered()
{//插入记录
}

void MainWindow::on_tableView_doubleClicked(const QModelIndex &index)
{ //tableView上双击,编辑当前记录
}


