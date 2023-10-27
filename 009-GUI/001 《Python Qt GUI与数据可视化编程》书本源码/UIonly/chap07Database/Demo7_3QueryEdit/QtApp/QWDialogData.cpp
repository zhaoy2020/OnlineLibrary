#include "QWDialogData.h"
#include "ui_QWDialogData.h"

#include    <QFileDialog>

QWDialogData::QWDialogData(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::QWDialogData)
{
    ui->setupUi(this);
}

QWDialogData::~QWDialogData()
{
    delete ui;
}

void QWDialogData::setUpdateRecord(QSqlRecord &recData)
{ //编辑记录，更新记录数据到界面
}

void QWDialogData::setInsertRecord(QSqlRecord &recData)
{//插入记录，无需更新界面显示，但是要存储recData的字段结构
}

QSqlRecord QWDialogData::getRecordData()
{ //"确定"按钮后，界面数据保存到记录mRecord
}

void QWDialogData::on_btnClearPhoto_clicked()
{ //清除照片
}

void QWDialogData::on_btnSetPhoto_clicked()
{//设置照片
}
