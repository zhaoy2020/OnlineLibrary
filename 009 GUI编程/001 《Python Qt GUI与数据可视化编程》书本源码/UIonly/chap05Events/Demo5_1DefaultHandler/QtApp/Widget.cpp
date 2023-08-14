#include "Widget.h"
#include "ui_Widget.h"

#include <QMessageBox>

#include <QPainter>

void Widget::paintEvent(QPaintEvent *event)
{
}

void Widget::resizeEvent(QResizeEvent *event)
{
}

void Widget::closeEvent(QCloseEvent *event)
{
}

void Widget::keyPressEvent(QKeyEvent *event)
{
}

void Widget::showEvent(QShowEvent *event)
{
}

void Widget::hideEvent(QHideEvent *event)
{
}


void Widget::mousePressEvent(QMouseEvent *event)
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

