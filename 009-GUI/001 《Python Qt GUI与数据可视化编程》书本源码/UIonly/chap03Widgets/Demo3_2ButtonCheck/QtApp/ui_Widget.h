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
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Widget
{
public:
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QPushButton *btnAlign_Left;
    QPushButton *btnAlign_Center;
    QPushButton *btnAlign_Right;
    QFrame *line;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *btnFont_Bold;
    QPushButton *btnFont_Italic;
    QPushButton *btnFont_UnderLine;
    QFrame *line_2;
    QHBoxLayout *horizontalLayout_3;
    QCheckBox *chkBox_Readonly;
    QCheckBox *chkbox_Enable;
    QCheckBox *chkBox_ClearButton;
    QLineEdit *editInput;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName(QStringLiteral("Widget"));
        Widget->resize(310, 189);
        verticalLayout = new QVBoxLayout(Widget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        btnAlign_Left = new QPushButton(Widget);
        btnAlign_Left->setObjectName(QStringLiteral("btnAlign_Left"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/508.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnAlign_Left->setIcon(icon);
        btnAlign_Left->setCheckable(true);
        btnAlign_Left->setChecked(true);
        btnAlign_Left->setAutoExclusive(true);
        btnAlign_Left->setFlat(true);

        horizontalLayout->addWidget(btnAlign_Left);

        btnAlign_Center = new QPushButton(Widget);
        btnAlign_Center->setObjectName(QStringLiteral("btnAlign_Center"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/510.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnAlign_Center->setIcon(icon1);
        btnAlign_Center->setCheckable(true);
        btnAlign_Center->setChecked(false);
        btnAlign_Center->setAutoExclusive(true);
        btnAlign_Center->setFlat(true);

        horizontalLayout->addWidget(btnAlign_Center);

        btnAlign_Right = new QPushButton(Widget);
        btnAlign_Right->setObjectName(QStringLiteral("btnAlign_Right"));
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/512.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnAlign_Right->setIcon(icon2);
        btnAlign_Right->setCheckable(true);
        btnAlign_Right->setChecked(false);
        btnAlign_Right->setAutoExclusive(true);
        btnAlign_Right->setFlat(true);

        horizontalLayout->addWidget(btnAlign_Right);


        verticalLayout->addLayout(horizontalLayout);

        line = new QFrame(Widget);
        line->setObjectName(QStringLiteral("line"));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        verticalLayout->addWidget(line);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        btnFont_Bold = new QPushButton(Widget);
        btnFont_Bold->setObjectName(QStringLiteral("btnFont_Bold"));
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/images/500.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnFont_Bold->setIcon(icon3);
        btnFont_Bold->setCheckable(true);

        horizontalLayout_2->addWidget(btnFont_Bold);

        btnFont_Italic = new QPushButton(Widget);
        btnFont_Italic->setObjectName(QStringLiteral("btnFont_Italic"));
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/icons/images/502.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnFont_Italic->setIcon(icon4);
        btnFont_Italic->setCheckable(true);

        horizontalLayout_2->addWidget(btnFont_Italic);

        btnFont_UnderLine = new QPushButton(Widget);
        btnFont_UnderLine->setObjectName(QStringLiteral("btnFont_UnderLine"));
        QIcon icon5;
        icon5.addFile(QStringLiteral(":/icons/images/504.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnFont_UnderLine->setIcon(icon5);
        btnFont_UnderLine->setCheckable(true);

        horizontalLayout_2->addWidget(btnFont_UnderLine);


        verticalLayout->addLayout(horizontalLayout_2);

        line_2 = new QFrame(Widget);
        line_2->setObjectName(QStringLiteral("line_2"));
        line_2->setFrameShape(QFrame::HLine);
        line_2->setFrameShadow(QFrame::Sunken);

        verticalLayout->addWidget(line_2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        chkBox_Readonly = new QCheckBox(Widget);
        chkBox_Readonly->setObjectName(QStringLiteral("chkBox_Readonly"));

        horizontalLayout_3->addWidget(chkBox_Readonly);

        chkbox_Enable = new QCheckBox(Widget);
        chkbox_Enable->setObjectName(QStringLiteral("chkbox_Enable"));
        chkbox_Enable->setChecked(true);

        horizontalLayout_3->addWidget(chkbox_Enable);

        chkBox_ClearButton = new QCheckBox(Widget);
        chkBox_ClearButton->setObjectName(QStringLiteral("chkBox_ClearButton"));

        horizontalLayout_3->addWidget(chkBox_ClearButton);


        verticalLayout->addLayout(horizontalLayout_3);

        editInput = new QLineEdit(Widget);
        editInput->setObjectName(QStringLiteral("editInput"));
        editInput->setMinimumSize(QSize(0, 50));
        QFont font;
        font.setPointSize(14);
        editInput->setFont(font);

        verticalLayout->addWidget(editInput);


        retranslateUi(Widget);

        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QApplication::translate("Widget", "Demo3-2\357\274\214\346\214\211\351\222\256\347\232\204\344\275\277\347\224\250", nullptr));
        btnAlign_Left->setText(QApplication::translate("Widget", "\345\261\205\345\267\246", nullptr));
        btnAlign_Center->setText(QApplication::translate("Widget", "\345\261\205\344\270\255", nullptr));
        btnAlign_Right->setText(QApplication::translate("Widget", "\345\261\205\345\217\263", nullptr));
        btnFont_Bold->setText(QApplication::translate("Widget", "\347\262\227\344\275\223", nullptr));
        btnFont_Italic->setText(QApplication::translate("Widget", "\346\226\234\344\275\223", nullptr));
        btnFont_UnderLine->setText(QApplication::translate("Widget", "\344\270\213\345\210\222\347\272\277", nullptr));
        chkBox_Readonly->setText(QApplication::translate("Widget", "Readonly", nullptr));
        chkbox_Enable->setText(QApplication::translate("Widget", "Enabled", nullptr));
        chkBox_ClearButton->setText(QApplication::translate("Widget", "ClearButtonEnabled", nullptr));
        editInput->setText(QApplication::translate("Widget", "\346\265\213\350\257\225\346\230\276\347\244\272\346\226\207\346\234\254", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
