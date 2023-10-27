#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>

namespace Ui {
class Widget;
}

class Widget : public QWidget
{
   Q_OBJECT

public:
   explicit Widget(QWidget *parent = 0);
   ~Widget();

private slots:

   void on_btnCalculate_clicked();

   void on_spinCount_valueChanged(int arg1);

   void on_spinPrice_valueChanged(double arg1);

private:
   Ui::Widget *ui;
};

#endif // WIDGET_H
