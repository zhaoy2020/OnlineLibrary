#ifndef DIALOG_H
#define DIALOG_H

#include <QDialog>

namespace Ui {
class Dialog;
}

class Dialog : public QDialog
{
    Q_OBJECT

public:
    explicit Dialog(QWidget *parent = 0);
    ~Dialog();

private slots:
    void on_btnOpen_clicked();

    void on_btnOpenMulti_clicked();

    void on_btnSelDir_clicked();

    void on_btnSave_clicked();

private:
    Ui::Dialog *ui;
};

#endif // DIALOG_H
