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
// QTableView的SelectionModel的当前单元格发生了变化，进行处理
    void do_currentChanged(const QModelIndex &current, const QModelIndex &previous);

// QTableView的SelectionModel的行发生了变化，进行处理
    void do_currentRowChanged(const QModelIndex &current, const QModelIndex &previous);

    void on_actOpenDB_triggered();

    void on_actRecAppend_triggered();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
