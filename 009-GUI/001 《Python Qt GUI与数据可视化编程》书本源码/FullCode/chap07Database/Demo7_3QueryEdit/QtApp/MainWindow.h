#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include    <QLabel>
#include    <QString>

#include    <QtSql>
#include    <QDataWidgetMapper>

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
// QTableView的SelectionModel的行发生了变化，进行处理
    void do_currentRowChanged(const QModelIndex &current, const QModelIndex &previous);

    void on_actOpenDB_triggered();

    void on_actRecInsert_triggered();

    void on_tableView_doubleClicked(const QModelIndex &index);

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
