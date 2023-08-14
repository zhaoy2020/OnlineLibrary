#include "QWFormDoc.h"
#include "ui_QWFormDoc.h"


QWFormDoc::QWFormDoc(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::QWFormDoc)
{
    ui->setupUi(this);

    this->setWindowTitle("New Doc"); //窗口标题
    this->setAttribute(Qt::WA_DeleteOnClose); //关闭时自动删除
}

QWFormDoc::~QWFormDoc()
{
    delete ui;
}

