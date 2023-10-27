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

   void on_btnAlign_Left_clicked();

   void on_btnFont_Bold_clicked(bool checked);

   void on_chkBox_ClearButton_clicked(bool checked);

private:
   Ui::Widget *ui;
};

#endif // WIDGET_H
