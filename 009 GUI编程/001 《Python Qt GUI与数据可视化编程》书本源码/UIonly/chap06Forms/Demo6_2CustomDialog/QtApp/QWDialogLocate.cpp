#include "QWDialogLocate.h"
#include "ui_QWDialogLocate.h"

#include    <QCloseEvent>

void QWDialogLocate::closeEvent(QCloseEvent *event)
{ //窗口关闭 event,释放信号使 actTab_Locate 能用

}

void QWDialogLocate::showEvent(QShowEvent *event)
{//窗口显示 event,释放信号使 actTab_Locate 不能用

}

QWDialogLocate::QWDialogLocate(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::QWDialogLocate)
{
    ui->setupUi(this);
}

QWDialogLocate::~QWDialogLocate()
{
    delete ui;
}

void QWDialogLocate::setSpinRange(int rowCount, int colCount)
{//设置SpinBox输入最大值

}

void QWDialogLocate::do_setSpinValue(int rowNo, int colNo)
{//响应主窗口信号，更新spinBox的值

}

void QWDialogLocate::on_btnSetText_clicked()
{//定位到单元格，并设置字符串

}
