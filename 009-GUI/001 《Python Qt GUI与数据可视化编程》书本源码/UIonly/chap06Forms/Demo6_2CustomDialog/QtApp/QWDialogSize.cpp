#include "QWDialogSize.h"
#include "ui_QWDialogSize.h"


QWDialogSize::QWDialogSize(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::QWDialogSize)
{
    ui->setupUi(this);
}

QWDialogSize::~QWDialogSize()
{
    delete ui;
}

