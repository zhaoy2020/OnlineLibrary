#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include    <QMainWindow>

#include    "QWFormDoc.h"

#include    <QMdiSubWindow>

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
    void on_actDoc_Open_triggered(); //打开文件

    void on_actDoc_New_triggered(); //新建文件

    void on_mdiArea_subWindowActivated(QMdiSubWindow *arg1); //子窗口被激活

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
