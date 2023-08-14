#include "Widget.h"
#include "ui_Widget.h"

void Widget::defense(QString weapon)
{
    QUrl   url=QUrl::fromLocalFile(appPath+"/sound/"+weapon);
    player->setSource(url);
    player->play();
}

void Widget::attack(QString weapon)
{
    QString filename=appPath+"/sound/"+weapon;
    QSound::play(filename);
}

Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
{
    ui->setupUi(this);
    appPath=QCoreApplication::applicationDirPath();//  无“/”
    player=new QSoundEffect(this);
    player->setLoopCount(2);
}

Widget::~Widget()
{
    delete ui;
}

void Widget::on_btnEffect_File_clicked()
{
   QString filename=appPath+"/sound/Ak47.wav";
   QUrl   url=QUrl::fromLocalFile(filename);
   player->setSource(url);
   player->play();
}

void Widget::on_btnEffect_Resource_clicked()
{
   QUrl   url=QUrl::fromLocalFile(":/Wave/sound/blast.wav");
   player->setSource(url);
   player->play();
}

void Widget::on_btnSound_File_clicked()
{
   QString filename=appPath+"/sound/fire2.wav";
   QSound::play(filename);
}

void Widget::on_btnSound_Resource_clicked()
{//资源文件
   QSound::play(":/Wave/sound/machinegun.wav");
}
