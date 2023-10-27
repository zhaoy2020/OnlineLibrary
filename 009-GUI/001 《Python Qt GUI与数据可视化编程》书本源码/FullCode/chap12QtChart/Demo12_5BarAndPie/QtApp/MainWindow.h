#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include    <QStandardItemModel>
#include    <QItemSelectionModel>
#include    <QBarSet>


#include    <QtCharts>  //必须这么设置
QT_CHARTS_USE_NAMESPACE

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
//三个分数列的数据发生变化，重新计算平均分
    void do_calcuAverage(QStandardItem *item);

    void do_PieSliceHighlight(bool show);

    void do_PieSeriesHovered(QPieSlice *slice, bool state);

    void do_legendMarkerClicked();

    void do_barHovered(bool status, int index, QBarSet *barset);

    void do_barClicked(int index, QBarSet *barset);


private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
