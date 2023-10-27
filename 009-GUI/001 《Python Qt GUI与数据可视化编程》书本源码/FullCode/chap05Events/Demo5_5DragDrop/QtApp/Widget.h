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
   void dragEnterEvent(QDragEnterEvent *event);

   void dropEvent(QDropEvent *event);

public:
   explicit Widget(QWidget *parent = 0);
   ~Widget();

private slots:

private:
   Ui::Widget *ui;
};

#endif // WIDGET_H
