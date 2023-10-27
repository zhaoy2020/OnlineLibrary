#include    "QWFormDoc.h"
#include    "ui_QWFormDoc.h"

QWFormDoc::QWFormDoc(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::QWFormDoc)
{
    ui->setupUi(this);
    this->setAutoFillBackground(true);
}

QWFormDoc::~QWFormDoc()
{
    delete ui;
}

void QWFormDoc::loadFromFile(QString &aFileName)
{
}

void QWFormDoc::on_actOpen_triggered()
{
}

void QWFormDoc::on_actFont_triggered()
{
}
