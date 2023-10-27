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

    void do_seriesMag_hovered(const QPointF &point, bool state);

    void do_seriesPhase_hovered(const QPointF &point, bool state);


    void on_actZoomReset_triggered(); //工具栏按钮，原始大小

    void on_actZoomIn_triggered(); //工具栏按钮，放大

    void on_actZoomOut_triggered(); //工具栏按钮，缩小

    void on_actOpen_triggered();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
