#include "Widget.h"
#include "ui_Widget.h"

void Widget::dragEnterEvent(QDragEnterEvent *event)
{
}

void Widget::dropEvent(QDropEvent *event)
{
}

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

