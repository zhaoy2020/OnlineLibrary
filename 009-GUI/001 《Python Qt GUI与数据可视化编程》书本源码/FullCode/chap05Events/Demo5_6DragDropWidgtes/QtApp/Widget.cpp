#include "Widget.h"
#include "ui_Widget.h"


bool Widget::eventFilter(QObject *watched, QEvent *event)
{//处理删除操作
}

Widget::Widget(QWidget *parent):
   QWidget(parent),
   ui(new Ui::Widget)
{
   ui->setupUi(this);
}

Widget::~Widget()
{
   delete ui;
}

void Widget::on_radio_Source_clicked()
{
}

void Widget::on_chkBox_AcceptDrops_clicked(bool checked)
{
}

