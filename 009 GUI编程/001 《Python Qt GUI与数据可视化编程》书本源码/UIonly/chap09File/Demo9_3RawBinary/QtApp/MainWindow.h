#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

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
   void on_btnFile_clicked();

   void on_btnInt32_Write_clicked();

   void on_btnInt32_Read_clicked();

   void on_actSaveALL_triggered();

   void on_actReadALL_triggered();

private:
   Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
