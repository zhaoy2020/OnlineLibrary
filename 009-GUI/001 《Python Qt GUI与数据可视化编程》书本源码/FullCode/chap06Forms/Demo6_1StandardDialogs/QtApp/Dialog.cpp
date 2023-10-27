#include "Dialog.h"
#include "ui_Dialog.h"


Dialog::Dialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Dialog)
{
    ui->setupUi(this);
}

Dialog::~Dialog()
{
    delete ui;
}

void Dialog::on_btnOpen_clicked()
{ //选择单个文件
}

void Dialog::on_btnOpenMulti_clicked()
{ //选择多个文件
}

void Dialog::on_btnSelDir_clicked()
{ //选择文件夹
}

void Dialog::on_btnSave_clicked()
{//保存文件
}

