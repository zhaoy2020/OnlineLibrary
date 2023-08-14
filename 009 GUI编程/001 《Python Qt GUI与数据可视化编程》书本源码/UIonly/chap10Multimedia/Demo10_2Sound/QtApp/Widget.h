#ifndef WIDGET_H
#define WIDGET_H

#include    <QWidget>
#include    <QSoundEffect>
#include    <QSound>

namespace Ui {
class Widget;
}

class Widget : public QWidget
{
    Q_OBJECT

private:
    QSoundEffect    *player;
    QString     appPath;

    void    defense(QString weapon);
    void    attack(QString weapon);

public:
    explicit Widget(QWidget *parent = 0);
    ~Widget();

private slots:
    void on_btnEffect_File_clicked();

    void on_btnEffect_Resource_clicked();

    void on_btnSound_File_clicked();

    void on_btnSound_Resource_clicked();

private:
    Ui::Widget *ui;
};

#endif // WIDGET_H
