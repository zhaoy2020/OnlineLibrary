#ifndef QFORMTABLE_H
#define QFORMTABLE_H

#include <QMainWindow>

namespace Ui {
class QWFormTable;
}

class QWFormTable : public QMainWindow
{
    Q_OBJECT

public:
    explicit QWFormTable(QWidget *parent = 0);
    ~QWFormTable();

private slots:
    void on_actSetSize_triggered();

    void on_actSetHeader_triggered();

private:
    Ui::QWFormTable *ui;
};

#endif // QFORMTABLE_H
