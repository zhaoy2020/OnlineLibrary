#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>

#include <QEvent>

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

private:
   Ui::Widget *ui;
};

#endif // WIDGET_H
