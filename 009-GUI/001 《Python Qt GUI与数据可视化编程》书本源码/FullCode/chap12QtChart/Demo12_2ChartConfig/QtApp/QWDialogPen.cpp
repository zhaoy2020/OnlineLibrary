#include "QWDialogPen.h"
#include "ui_QWDialogPen.h"

#include    <QColorDialog>


QWDialogPen::QWDialogPen(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::QWDialogPen)
{
    ui->setupUi(this);
}

QWDialogPen::~QWDialogPen()
{
    delete ui;
}

