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
#include <QtWidgets/QLabel>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Widget
{
public:
    QVBoxLayout *verticalLayout;
    QLabel *LabHover;
    QLabel *LabDBClick;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName(QString::fromUtf8("Widget"));
        Widget->resize(265, 154);
        verticalLayout = new QVBoxLayout(Widget);
        verticalLayout->setSpacing(20);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        LabHover = new QLabel(Widget);
        LabHover->setObjectName(QString::fromUtf8("LabHover"));
        QFont font;
        font.setFamily(QString::fromUtf8("Times New Roman"));
        font.setPointSize(12);
        font.setBold(true);
        font.setWeight(75);
        LabHover->setFont(font);
        LabHover->setAutoFillBackground(false);
        LabHover->setStyleSheet(QString::fromUtf8(""));
        LabHover->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(LabHover);

        LabDBClick = new QLabel(Widget);
        LabDBClick->setObjectName(QString::fromUtf8("LabDBClick"));
        LabDBClick->setFont(font);
        LabDBClick->setAutoFillBackground(false);
        LabDBClick->setStyleSheet(QString::fromUtf8(""));
        LabDBClick->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(LabDBClick);


        retranslateUi(Widget);

        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QApplication::translate("Widget", "Demo5_4\357\274\214\344\272\213\344\273\266\350\277\207\346\273\244\345\231\250", nullptr));
        LabHover->setText(QApplication::translate("Widget", "\351\235\240\350\277\221\346\210\221\357\274\214\347\202\271\345\207\273\346\210\221", nullptr));
        LabDBClick->setText(QApplication::translate("Widget", "\345\217\257\345\217\214\345\207\273\347\232\204\346\240\207\347\255\276", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
