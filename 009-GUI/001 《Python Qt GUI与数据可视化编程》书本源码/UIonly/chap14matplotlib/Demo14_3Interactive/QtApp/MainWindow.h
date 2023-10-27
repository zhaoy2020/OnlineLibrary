#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include    <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

private:

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:

   void on_actTightLayout_triggered();

   void on_actSetCursor_triggered();

   void on_actScatterAgain_triggered();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
