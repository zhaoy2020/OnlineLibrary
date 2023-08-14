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


void Widget::on_btnGetTime_clicked()
{

}

void Widget::on_calendarWidget_selectionChanged()
{

}
