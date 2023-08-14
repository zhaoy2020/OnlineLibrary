#ifndef QWDIALOGLOCATE_H
#define QWDIALOGLOCATE_H

#include <QDialog>


namespace Ui {
class QWDialogLocate;
}

class QWDialogLocate : public QDialog
{
    Q_OBJECT

public:
    explicit QWDialogLocate(QWidget *parent = 0);
    ~QWDialogLocate();

private:
    Ui::QWDialogLocate *ui;
};

#endif // QWDIALOGLOCATE_H
