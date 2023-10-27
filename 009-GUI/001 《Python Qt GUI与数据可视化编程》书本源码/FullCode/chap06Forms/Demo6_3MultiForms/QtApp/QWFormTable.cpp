#include "QWFormTable.h"
#include "ui_QWFormTable.h"

QWFormTable::QWFormTable(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::QWFormTable)
{
    ui->setupUi(this);
    this->setAutoFillBackground(true);
    this->setCentralWidget(ui->tableView);
}

QWFormTable::~QWFormTable()
{
    delete ui;
}

void QWFormTable::on_actSetSize_triggered()
{

}

void QWFormTable::on_actSetHeader_triggered()
{

}

