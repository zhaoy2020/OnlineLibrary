#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include    <QMainWindow>

#include    <QCameraInfo>
#include    <QCamera>

#include    <QCameraImageCapture>


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
//QCamera的槽函数
    void    do_cameraStateChanged(QCamera::State state);

//QCameraImageCapture的槽函数
    void    do_imageReady(bool ready);
    void    do_imageCaptured(int id, const QImage &preview);
    void    do_imageSaved(int id, const QString &fileName);


    void on_actStartCamera_triggered();

    void on_actStopCamera_triggered();

    void on_actCapture_triggered();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
