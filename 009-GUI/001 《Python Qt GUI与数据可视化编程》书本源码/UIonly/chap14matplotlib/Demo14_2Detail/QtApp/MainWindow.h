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

   void on_chkBoxX_minorTick_clicked(bool checked);

   void on_chkBoxX_minorTickBottom_clicked(bool checked);

   void on_chkBoxX_minorTickTop_clicked(bool checked);

   void on_chkBoxY_majorTickLeft_clicked(bool checked);

   void on_chkBoxY_majorTickRight_clicked(bool checked);

   void on_chkBoxAxisY_TickLabLeft_clicked(bool checked);

   void on_chkBoxAxisY_TickLabRight_clicked(bool checked);

   void on_chkBoxY_minorTickLeft_clicked(bool checked);

   void on_chkBoxY_minorTickRight_clicked(bool checked);

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
