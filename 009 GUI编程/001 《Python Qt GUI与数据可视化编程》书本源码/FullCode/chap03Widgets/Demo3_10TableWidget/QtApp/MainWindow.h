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
   void on_btnSetHeader_clicked();

   void on_tableInfo_currentCellChanged(int currentRow, int currentColumn, int previousRow, int previousColumn);

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
