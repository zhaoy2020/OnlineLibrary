#ifndef QWGRAPHICSVIEW_H
#define QWGRAPHICSVIEW_H

#include <QObject>
#include <QGraphicsView>
#include "QWGraphicsView.h"

class QWGraphicsView : public QGraphicsView
{
    Q_OBJECT

protected:
    void mouseMoveEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *event);

public:
    explicit QWGraphicsView(QWidget *parent = 0);

signals:
    void mouseMove(QPoint point); //鼠标移动
    void mouseClicked(QPoint point); //鼠标单击
    void mouseDoubleClick(QPoint point); //双击事件
    void keyPress(QKeyEvent *event); //按键事件
};

#endif // QWGRAPHICSVIEW_H
