#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>

#include <QResizeEvent>

#include <QCloseEvent>

#include <QKeyEvent>

#include <QPaintEvent>

namespace Ui {
class Widget;
}

class Widget : public QWidget
{
   Q_OBJECT

protected:
   void paintEvent(QPaintEvent *event);

   void resizeEvent(QResizeEvent *event);

   void closeEvent(QCloseEvent *event);

//   void keyReleaseEvent(QKeyEvent *event);

   void keyPressEvent(QKeyEvent *event);

   void showEvent(QShowEvent *event);

   void hideEvent(QHideEvent *event);

   void mousePressEvent(QMouseEvent *event);

public:
   explicit Widget(QWidget *parent = 0);
   ~Widget();

private slots:

private:
   Ui::Widget *ui;
};

#endif // WIDGET_H
