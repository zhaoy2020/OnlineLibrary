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
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
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
    QLabel *label;
    QLineEdit *editCount;
    QLabel *label_2;
    QLineEdit *editPrice;
    QPushButton *btnCalculate;
    QLabel *label_3;
    QLineEdit *editTotal;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout_2;
    QLabel *label_6;
    QSpinBox *spinCount;
    QLabel *label_5;
    QDoubleSpinBox *spinPrice;
    QLabel *label_4;
    QDoubleSpinBox *spinTotal;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName(QStringLiteral("Widget"));
        Widget->resize(332, 200);
        verticalLayout = new QVBoxLayout(Widget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        groupBox = new QGroupBox(Widget);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        gridLayout = new QGridLayout(groupBox);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        label = new QLabel(groupBox);
        label->setObjectName(QStringLiteral("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        editCount = new QLineEdit(groupBox);
        editCount->setObjectName(QStringLiteral("editCount"));

        gridLayout->addWidget(editCount, 0, 1, 1, 1);

        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout->addWidget(label_2, 0, 2, 1, 1);

        editPrice = new QLineEdit(groupBox);
        editPrice->setObjectName(QStringLiteral("editPrice"));

        gridLayout->addWidget(editPrice, 0, 3, 1, 1);

        btnCalculate = new QPushButton(groupBox);
        btnCalculate->setObjectName(QStringLiteral("btnCalculate"));

        gridLayout->addWidget(btnCalculate, 1, 0, 1, 2);

        label_3 = new QLabel(groupBox);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout->addWidget(label_3, 1, 2, 1, 1);

        editTotal = new QLineEdit(groupBox);
        editTotal->setObjectName(QStringLiteral("editTotal"));

        gridLayout->addWidget(editTotal, 1, 3, 1, 1);


        verticalLayout->addWidget(groupBox);

        groupBox_2 = new QGroupBox(Widget);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        gridLayout_2 = new QGridLayout(groupBox_2);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        label_6 = new QLabel(groupBox_2);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setAlignment(Qt::AlignCenter);

        gridLayout_2->addWidget(label_6, 0, 0, 1, 1);

        spinCount = new QSpinBox(groupBox_2);
        spinCount->setObjectName(QStringLiteral("spinCount"));
        spinCount->setFrame(true);
        spinCount->setButtonSymbols(QAbstractSpinBox::UpDownArrows);
        spinCount->setProperty("showGroupSeparator", QVariant(false));
        spinCount->setMaximum(1000);
        spinCount->setValue(6);

        gridLayout_2->addWidget(spinCount, 0, 1, 1, 1);

        label_5 = new QLabel(groupBox_2);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setAlignment(Qt::AlignCenter);

        gridLayout_2->addWidget(label_5, 0, 2, 1, 1);

        spinPrice = new QDoubleSpinBox(groupBox_2);
        spinPrice->setObjectName(QStringLiteral("spinPrice"));
        spinPrice->setMinimumSize(QSize(90, 0));
        spinPrice->setMaximum(10000);
        spinPrice->setValue(11.23);

        gridLayout_2->addWidget(spinPrice, 0, 3, 1, 1);

        label_4 = new QLabel(groupBox_2);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setAlignment(Qt::AlignCenter);

        gridLayout_2->addWidget(label_4, 1, 2, 1, 1);

        spinTotal = new QDoubleSpinBox(groupBox_2);
        spinTotal->setObjectName(QStringLiteral("spinTotal"));
        spinTotal->setReadOnly(true);
        spinTotal->setMaximum(100000);

        gridLayout_2->addWidget(spinTotal, 1, 3, 1, 1);


        verticalLayout->addWidget(groupBox_2);


        retranslateUi(Widget);

        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QApplication::translate("Widget", "Demo3-1\357\274\214\346\225\260\345\200\274\350\276\223\345\205\245", nullptr));
        groupBox->setTitle(QApplication::translate("Widget", "QLineEdit\350\276\223\345\205\245\345\222\214\346\230\276\347\244\272\346\225\260\345\200\274", nullptr));
        label->setText(QApplication::translate("Widget", "\346\225\260 \351\207\217", nullptr));
        editCount->setText(QApplication::translate("Widget", "16", nullptr));
        label_2->setText(QApplication::translate("Widget", "\345\215\225 \344\273\267", nullptr));
        editPrice->setText(QApplication::translate("Widget", "12.37", nullptr));
        btnCalculate->setText(QApplication::translate("Widget", "\350\256\241\347\256\227\346\200\273\344\273\267", nullptr));
        label_3->setText(QApplication::translate("Widget", "\346\200\273 \344\273\267", nullptr));
        groupBox_2->setTitle(QApplication::translate("Widget", "SpinBox\350\276\223\345\205\245\345\222\214\346\230\276\347\244\272", nullptr));
        label_6->setText(QApplication::translate("Widget", "\346\225\260 \351\207\217", nullptr));
        spinCount->setSuffix(QApplication::translate("Widget", " kg", nullptr));
        label_5->setText(QApplication::translate("Widget", "\345\215\225 \344\273\267", nullptr));
        spinPrice->setPrefix(QApplication::translate("Widget", "$ ", nullptr));
        label_4->setText(QApplication::translate("Widget", "\350\207\252\345\212\250\350\256\241\347\256\227\346\200\273\344\273\267", nullptr));
        spinTotal->setPrefix(QApplication::translate("Widget", "$ ", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
