#include "Widget.h"
#include "ui_Widget.h"


Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
{
    ui->setupUi(this);
}

Widget::~Widget()
{
    delete ui;
}


void Widget::on_btnIniItems_clicked()
{

}

void Widget::on_comboBox_currentIndexChanged(const QString &arg1)
{

}
