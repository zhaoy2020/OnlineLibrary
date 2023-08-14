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
   void on_actTree_AddFolder_triggered();

   void on_treeFiles_currentItemChanged(QTreeWidgetItem *current, QTreeWidgetItem *previous);

   void on_dockWidget_visibilityChanged(bool visible);

   void on_dockWidget_topLevelChanged(bool topLevel);

private:
    Ui::MainWindow *ui;

};

#endif // MAINWINDOW_H
