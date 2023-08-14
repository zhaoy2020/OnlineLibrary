/********************************************************************************
** Form generated from reading UI file 'Widget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_H
#define UI_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Widget
{
public:
    QVBoxLayout *verticalLayout_3;
    QGroupBox *groupBox_3;
    QVBoxLayout *verticalLayout;
    QPushButton *btnEffect_File;
    QPushButton *btnEffect_Resource;
    QGroupBox *groupBox_4;
    QVBoxLayout *verticalLayout_2;
    QPushButton *btnSound_File;
    QPushButton *btnSound_Resource;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName(QString::fromUtf8("Widget"));
        Widget->resize(273, 228);
        Widget->setStyleSheet(QString::fromUtf8("QPushButton{min-height:30px;\n"
"}"));
        verticalLayout_3 = new QVBoxLayout(Widget);
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setContentsMargins(11, 11, 11, 11);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        groupBox_3 = new QGroupBox(Widget);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        verticalLayout = new QVBoxLayout(groupBox_3);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        btnEffect_File = new QPushButton(groupBox_3);
        btnEffect_File->setObjectName(QString::fromUtf8("btnEffect_File"));

        verticalLayout->addWidget(btnEffect_File);

        btnEffect_Resource = new QPushButton(groupBox_3);
        btnEffect_Resource->setObjectName(QString::fromUtf8("btnEffect_Resource"));

        verticalLayout->addWidget(btnEffect_Resource);


        verticalLayout_3->addWidget(groupBox_3);

        groupBox_4 = new QGroupBox(Widget);
        groupBox_4->setObjectName(QString::fromUtf8("groupBox_4"));
        verticalLayout_2 = new QVBoxLayout(groupBox_4);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        btnSound_File = new QPushButton(groupBox_4);
        btnSound_File->setObjectName(QString::fromUtf8("btnSound_File"));

        verticalLayout_2->addWidget(btnSound_File);

        btnSound_Resource = new QPushButton(groupBox_4);
        btnSound_Resource->setObjectName(QString::fromUtf8("btnSound_Resource"));

        verticalLayout_2->addWidget(btnSound_Resource);


        verticalLayout_3->addWidget(groupBox_4);


        retranslateUi(Widget);

        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QApplication::translate("Widget", "QSoundEffect\346\222\255\346\224\276\345\243\260\351\237\263", nullptr));
        groupBox_3->setTitle(QApplication::translate("Widget", "QSoundEffect\346\222\255\346\224\276\351\237\263\351\242\221", nullptr));
        btnEffect_File->setText(QApplication::translate("Widget", "\346\222\255\346\224\276\346\226\207\344\273\266", nullptr));
        btnEffect_Resource->setText(QApplication::translate("Widget", "\346\222\255\346\224\276\350\265\204\346\272\220\346\226\207\344\273\266", nullptr));
        groupBox_4->setTitle(QApplication::translate("Widget", "QSound\346\222\255\346\224\276\351\237\263\351\242\221", nullptr));
        btnSound_File->setText(QApplication::translate("Widget", "\346\222\255\346\224\276\346\226\207\344\273\266", nullptr));
        btnSound_Resource->setText(QApplication::translate("Widget", "\346\222\255\346\224\276\350\265\204\346\272\220\346\226\207\344\273\266", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
