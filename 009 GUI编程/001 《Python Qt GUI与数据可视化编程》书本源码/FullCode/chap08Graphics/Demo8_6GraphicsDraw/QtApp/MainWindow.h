#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include    "QWGraphicsView.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void    do_mouseMove(QPoint point); //鼠标移动

    void    do_mouseClicked(QPoint point); //鼠标单击

    void    do_mouseDoubleClick(QPoint point); //鼠标双击

    void    do_keyPress(QKeyEvent *event); //按键


    void on_actItem_Rect_triggered();

    void on_actItem_Ellipse_triggered();

    void on_actItem_Polygon_triggered();

    void on_actEdit_Delete_triggered();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
