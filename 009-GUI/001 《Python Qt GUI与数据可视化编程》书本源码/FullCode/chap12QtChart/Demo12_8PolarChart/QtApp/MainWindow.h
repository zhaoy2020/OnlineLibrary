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

private:
    QPolarChart *chart;//

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void do_series_hovered(const QPointF &point, bool state);

    void on_actZoomReset_triggered(); //工具栏按钮，原始大小

    void on_actZoomIn_triggered(); //工具栏按钮，放大

    void on_actZoomOut_triggered(); //工具栏按钮，缩小


private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
