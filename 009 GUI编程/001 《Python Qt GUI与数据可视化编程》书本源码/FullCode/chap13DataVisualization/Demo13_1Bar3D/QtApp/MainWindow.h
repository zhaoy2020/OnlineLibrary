#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include    <QtDataVisualization>

using namespace QtDataVisualization;


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
   void on_actBar_ChangeValue_triggered();

   void on_actData_Add_triggered();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
