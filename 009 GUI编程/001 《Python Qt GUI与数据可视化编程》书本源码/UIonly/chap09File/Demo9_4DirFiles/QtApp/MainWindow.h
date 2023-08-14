#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <QFileSystemWatcher>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
   Q_OBJECT

public slots:
    void do_directoryChanged(const QString path);

    void do_fileChanged(const QString path);

public:
   explicit MainWindow(QWidget *parent = 0);
   ~MainWindow();

private slots:
   void on_btnOpenFile_clicked();

   void on_btnOpenDir_clicked();

   void on_btnFile_exists_clicked();

   void on_btnFile_copy_clicked();

private:
   Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
