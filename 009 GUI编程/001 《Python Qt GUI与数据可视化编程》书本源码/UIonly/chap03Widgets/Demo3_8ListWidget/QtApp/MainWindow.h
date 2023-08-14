#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include    <QListWidgetItem>

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
   void on_actList_Ini_triggered();

   void on_actList_Insert_triggered();

   void on_actList_Delete_triggered();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
