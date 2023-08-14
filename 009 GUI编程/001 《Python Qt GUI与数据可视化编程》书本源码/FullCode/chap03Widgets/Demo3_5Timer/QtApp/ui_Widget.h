/********************************************************************************
** Form generated from reading UI file 'Widget.ui'
**
** Created by: Qt User Interface Compiler version 5.10.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_H
#define UI_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLCDNumber>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Widget
{
public:
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox;
    QGridLayout *gridLayout;
    QSpinBox *spinBoxIntv;
    QPushButton *btnStart;
    QPushButton *btnSetIntv;
    QPushButton *btnStop;
    QGroupBox *groupBox_2;
    QHBoxLayout *horizontalLayout;
    QLCDNumber *LCDHour;
    QLCDNumber *LCDMin;
    QLCDNumber *LCDSec;
    QLabel *LabElapsedTime;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName(QStringLiteral("Widget"));
        Widget->resize(296, 223);
        QFont font;
        font.setPointSize(10);
        Widget->setFont(font);
        verticalLayout = new QVBoxLayout(Widget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(5, 5, 5, 5);
        groupBox = new QGroupBox(Widget);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        gridLayout = new QGridLayout(groupBox);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(-1, 5, -1, 5);
        spinBoxIntv = new QSpinBox(groupBox);
        spinBoxIntv->setObjectName(QStringLiteral("spinBoxIntv"));
        spinBoxIntv->setMaximum(999999);
        spinBoxIntv->setValue(1000);

        gridLayout->addWidget(spinBoxIntv, 1, 1, 1, 1);

        btnStart = new QPushButton(groupBox);
        btnStart->setObjectName(QStringLiteral("btnStart"));

        gridLayout->addWidget(btnStart, 0, 0, 1, 1);

        btnSetIntv = new QPushButton(groupBox);
        btnSetIntv->setObjectName(QStringLiteral("btnSetIntv"));

        gridLayout->addWidget(btnSetIntv, 1, 0, 1, 1);

        btnStop = new QPushButton(groupBox);
        btnStop->setObjectName(QStringLiteral("btnStop"));
        btnStop->setEnabled(false);

        gridLayout->addWidget(btnStop, 0, 1, 1, 1);


        verticalLayout->addWidget(groupBox);

        groupBox_2 = new QGroupBox(Widget);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        horizontalLayout = new QHBoxLayout(groupBox_2);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        LCDHour = new QLCDNumber(groupBox_2);
        LCDHour->setObjectName(QStringLiteral("LCDHour"));
        LCDHour->setSmallDecimalPoint(false);
        LCDHour->setDigitCount(2);
        LCDHour->setSegmentStyle(QLCDNumber::Filled);
        LCDHour->setProperty("value", QVariant(10));
        LCDHour->setProperty("intValue", QVariant(10));

        horizontalLayout->addWidget(LCDHour);

        LCDMin = new QLCDNumber(groupBox_2);
        LCDMin->setObjectName(QStringLiteral("LCDMin"));
        LCDMin->setDigitCount(2);
        LCDMin->setProperty("intValue", QVariant(26));

        horizontalLayout->addWidget(LCDMin);

        LCDSec = new QLCDNumber(groupBox_2);
        LCDSec->setObjectName(QStringLiteral("LCDSec"));
        LCDSec->setDigitCount(2);
        LCDSec->setProperty("intValue", QVariant(35));

        horizontalLayout->addWidget(LCDSec);


        verticalLayout->addWidget(groupBox_2);

        LabElapsedTime = new QLabel(Widget);
        LabElapsedTime->setObjectName(QStringLiteral("LabElapsedTime"));
        LabElapsedTime->setMaximumSize(QSize(16777215, 20));

        verticalLayout->addWidget(LabElapsedTime);


        retranslateUi(Widget);

        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QApplication::translate("Widget", "Demo3_5 \345\256\232\346\227\266\345\231\250", nullptr));
        groupBox->setTitle(QApplication::translate("Widget", "\345\256\232\346\227\266\345\231\250", nullptr));
        spinBoxIntv->setSuffix(QApplication::translate("Widget", " ms", nullptr));
        btnStart->setText(QApplication::translate("Widget", "\345\274\200\345\247\213", nullptr));
        btnSetIntv->setText(QApplication::translate("Widget", "\350\256\276\347\275\256\345\221\250\346\234\237", nullptr));
        btnStop->setText(QApplication::translate("Widget", "\345\201\234\346\255\242", nullptr));
        groupBox_2->setTitle(QApplication::translate("Widget", "\345\275\223\345\211\215\346\227\266\351\227\264\357\274\210\345\260\217\346\227\266\357\274\232\345\210\206\357\274\232\347\247\222\357\274\211", nullptr));
        LabElapsedTime->setText(QApplication::translate("Widget", "\346\265\201\351\200\235\347\232\204\346\227\266\351\227\264", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
