#include "Widget.h"
#include "ui_Widget.h"


bool Widget::eventFilter(QObject *watched, QEvent *event)
{
   if (watched==ui->LabHover) //上面的Label
   {
      if (event->type()==QEvent::Enter)
         ui->LabHover->setStyleSheet("background-color: rgb(170, 255, 255);");
      else if (event->type()==QEvent::Leave)
      {
         ui->LabHover->setStyleSheet("");
         ui->LabHover->setText("靠近我，点击我");
      }
      else if (event->type()==QEvent::MouseButtonPress)
         ui->LabHover->setText("button pressed");
      else if (event->type()==QEvent::MouseButtonRelease)
         ui->LabHover->setText("button released");
   }

   if (watched==ui->LabDBClick)//下面的Label
   {
      if (event->type()==QEvent::Enter)
         ui->LabDBClick->setStyleSheet("background-color: rgb(85, 255, 127);");
      else if (event->type()==QEvent::Leave)
      {
         ui->LabDBClick->setStyleSheet("");
         ui->LabDBClick->setText("可双击的标签");
      }
      else if (event->type()==QEvent::MouseButtonDblClick)
         ui->LabDBClick->setText("double clicked");
   }

   return QWidget::eventFilter(watched,event);
}

Widget::Widget(QWidget *parent) :
   QWidget(parent),
   ui(new Ui::Widget)
{
   ui->setupUi(this);

   ui->LabHover->installEventFilter(this);
   ui->LabDBClick->installEventFilter(this);
}

Widget::~Widget()
{
   delete ui;
}

