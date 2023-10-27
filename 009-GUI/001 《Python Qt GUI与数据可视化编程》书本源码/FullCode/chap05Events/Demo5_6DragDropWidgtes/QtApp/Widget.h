#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>


namespace Ui {
class Widget;
}

class Widget : public QWidget
{
   Q_OBJECT

protected:
   bool eventFilter(QObject *watched, QEvent *event);

public:
   explicit Widget(QWidget *parent = 0);
   ~Widget();

private slots:
   void on_radio_Source_clicked();

   void on_chkBox_AcceptDrops_clicked(bool checked);

private:
   Ui::Widget *ui;
};

#endif // WIDGET_H
