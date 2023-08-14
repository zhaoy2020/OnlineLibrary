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
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Widget
{
public:
    QPushButton *btnTest;
    QPushButton *btnMove;
    QLabel *LabMove;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName(QStringLiteral("Widget"));
        Widget->resize(481, 348);
        QFont font;
        font.setPointSize(10);
        Widget->setFont(font);
        Widget->setCursor(QCursor(Qt::CrossCursor));
        btnTest = new QPushButton(Widget);
        btnTest->setObjectName(QStringLiteral("btnTest"));
        btnTest->setGeometry(QRect(170, 105, 146, 51));
        btnMove = new QPushButton(Widget);
        btnMove->setObjectName(QStringLiteral("btnMove"));
        btnMove->setGeometry(QRect(65, 220, 141, 51));
        LabMove = new QLabel(Widget);
        LabMove->setObjectName(QStringLiteral("LabMove"));
        LabMove->setGeometry(QRect(55, 35, 196, 41));
        QPalette palette;
        QBrush brush(QColor(255, 0, 0, 255));
        brush.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::WindowText, brush);
        palette.setBrush(QPalette::Inactive, QPalette::WindowText, brush);
        QBrush brush1(QColor(120, 120, 120, 255));
        brush1.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Disabled, QPalette::WindowText, brush1);
        LabMove->setPalette(palette);
        QFont font1;
        font1.setPointSize(12);
        font1.setBold(true);
        font1.setWeight(75);
        LabMove->setFont(font1);

        retranslateUi(Widget);

        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QApplication::translate("Widget", "Demo5_1, \347\211\271\345\256\232\344\272\213\344\273\266\347\232\204\345\244\204\347\220\206", nullptr));
        btnTest->setText(QApplication::translate("Widget", "Button at Center\n"
"resizeEvent\344\272\213\344\273\266", nullptr));
        btnMove->setText(QApplication::translate("Widget", "Movable Button\n"
"W,S,A,D\351\224\256\347\247\273\345\212\250", nullptr));
        LabMove->setText(QApplication::translate("Widget", "Ctrl+\351\274\240\346\240\207\345\267\246\351\224\256 \345\271\266\347\247\273\345\212\250", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
