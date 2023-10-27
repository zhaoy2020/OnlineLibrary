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
    void do_LegendMarkerClicked(); //图例单击槽函数,自定义槽函数

    void do_chartView_mouseMove(QPoint point); //鼠标移动事件，自定义槽函数

    void do_series_hovered(const QPointF &point, bool state);

    void do_series_clicked(const QPointF &point);

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
