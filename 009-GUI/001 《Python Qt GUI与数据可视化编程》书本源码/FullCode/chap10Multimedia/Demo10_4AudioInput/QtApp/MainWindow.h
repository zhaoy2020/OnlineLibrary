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
//自定义槽函数
//1. 使用内置的IODevice, pull mode
   void  do_IO_readyRead();

//2. 接收音频输入数据的流设备, push mode
   void  do_updateBlockInfo(qint64 blockSize,int maxValue, int minValue);

   void  do_stateChanged(QAudio::State state);

   void on_actStart_triggered();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
