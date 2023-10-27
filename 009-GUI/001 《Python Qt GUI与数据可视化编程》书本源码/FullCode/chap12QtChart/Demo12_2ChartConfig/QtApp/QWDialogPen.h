#ifndef QWDIALOGPEN_H
#define QWDIALOGPEN_H

#include <QDialog>

namespace Ui {
class QWDialogPen;
}

class QWDialogPen : public QDialog
{ //QPen属性设置对话框
    Q_OBJECT

public:
    explicit QWDialogPen(QWidget *parent = 0);
    ~QWDialogPen();


private:
    Ui::QWDialogPen *ui;
};

#endif // QWDIALOGPEN_H
