#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include    <QMainWindow>

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
 //当前选择单元格发生变化
    void do_currentChanged(const QModelIndex &current, const QModelIndex &previous);

    void on_actOpen_triggered(); //打开文件

    void on_actAppend_triggered(); //添加行

    void on_actDelete_triggered();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
