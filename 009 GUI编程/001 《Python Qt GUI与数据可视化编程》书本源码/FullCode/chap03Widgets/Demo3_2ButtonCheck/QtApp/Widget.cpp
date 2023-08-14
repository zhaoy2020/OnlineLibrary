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



void Widget::on_btnAlign_Left_clicked()
{

}

void Widget::on_btnFont_Bold_clicked(bool checked)
{

}

void Widget::on_chkBox_ClearButton_clicked(bool checked)
{

}
