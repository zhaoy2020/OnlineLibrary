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
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Widget
{
public:
    QLabel *LabPic;
    QPlainTextEdit *plainTextEdit;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName(QStringLiteral("Widget"));
        Widget->resize(468, 462);
        Widget->setMaximumSize(QSize(680, 480));
        QFont font;
        font.setPointSize(10);
        Widget->setFont(font);
        Widget->setAcceptDrops(true);
        LabPic = new QLabel(Widget);
        LabPic->setObjectName(QStringLiteral("LabPic"));
        LabPic->setGeometry(QRect(9, 155, 450, 298));
        LabPic->setAcceptDrops(true);
        LabPic->setPixmap(QPixmap(QString::fromUtf8("../../demo5_1SpecificEvents/QtApp/images/sea1.jpg")));
        LabPic->setScaledContents(true);
        LabPic->setAlignment(Qt::AlignCenter);
        plainTextEdit = new QPlainTextEdit(Widget);
        plainTextEdit->setObjectName(QStringLiteral("plainTextEdit"));
        plainTextEdit->setGeometry(QRect(9, 9, 450, 140));
        plainTextEdit->setMinimumSize(QSize(0, 140));

        retranslateUi(Widget);

        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QApplication::translate("Widget", "Demo5_4\357\274\214\345\244\226\351\203\250\346\226\207\344\273\266\346\213\226\346\224\276", nullptr));
        LabPic->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
