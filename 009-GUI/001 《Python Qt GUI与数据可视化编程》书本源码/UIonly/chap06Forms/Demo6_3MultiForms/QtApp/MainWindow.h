#ifndef QWMAINWINDOW_H
#define QWMAINWINDOW_H

#include <QMainWindow>

#include    "QWFormDoc.h"
#include    "QWFormTable.h"

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
    void do_docFileChanged(QString shortName);//

    void on_tabWidget_tabCloseRequested(int index);

    void on_actWidgetInsite_triggered();

    void on_actWindowInsite_triggered();

    void on_actWindow_triggered();

    void on_actWidget_triggered();

    void on_tabWidget_currentChanged(int index);

private:
    Ui::MainWindow *ui;
};

#endif // QWMAINWINDOW_H
