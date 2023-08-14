#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include    <QtCharts>  //必须这么设置
QT_CHARTS_USE_NAMESPACE     //必须这么设置

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
    void    do_chartView_mouseMove(QPoint point); //鼠标移动事件，自定义槽函数

    void on_actOpen_triggered();

//    void on_btnY_SetRange_clicked();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
