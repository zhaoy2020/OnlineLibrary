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
#include <QtWidgets/QFrame>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Widget
{
public:
    QFrame *frame;
    QHBoxLayout *horizontalLayout;
    QToolButton *toolButton_2;
    QToolButton *toolButton;
    QToolButton *toolButton_3;
    QToolButton *toolButton_4;
    QSpacerItem *horizontalSpacer;
    QToolButton *toolButton_5;
    QLabel *label;
    QLabel *label_2;
    QLineEdit *editRequired;
    QLineEdit *editName;
    QLineEdit *lineEdit_3;
    QLabel *label_3;
    QPushButton *pushButton;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName(QStringLiteral("Widget"));
        Widget->resize(584, 370);
        QFont font;
        font.setFamily(QString::fromUtf8("\345\256\213\344\275\223"));
        font.setPointSize(12);
        font.setBold(false);
        font.setItalic(false);
        font.setWeight(50);
        Widget->setFont(font);
        Widget->setStyleSheet(QLatin1String("QLineEdit{\n"
"	     border-width: 2px;\n"
"	     border-style: solid;\n"
"	     border-color: gray;\n"
"	     border-radius: 10px;\n"
"	     padding: 0px 10px; \n"
"}\n"
"\n"
"QLineEdit:hover{\n"
"	     background-color: lime;\n"
"	     color: blue;\n"
"}\n"
"\n"
""));
        frame = new QFrame(Widget);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setGeometry(QRect(0, 0, 425, 87));
        frame->setMaximumSize(QSize(16777215, 90));
        frame->setStyleSheet(QStringLiteral(""));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        horizontalLayout = new QHBoxLayout(frame);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        toolButton_2 = new QToolButton(frame);
        toolButton_2->setObjectName(QStringLiteral("toolButton_2"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/video_chat.ico"), QSize(), QIcon::Normal, QIcon::Off);
        toolButton_2->setIcon(icon);
        toolButton_2->setIconSize(QSize(45, 45));
        toolButton_2->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);

        horizontalLayout->addWidget(toolButton_2);

        toolButton = new QToolButton(frame);
        toolButton->setObjectName(QStringLiteral("toolButton"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/user_group.ico"), QSize(), QIcon::Normal, QIcon::Off);
        toolButton->setIcon(icon1);
        toolButton->setIconSize(QSize(45, 45));
        toolButton->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);

        horizontalLayout->addWidget(toolButton);

        toolButton_3 = new QToolButton(frame);
        toolButton_3->setObjectName(QStringLiteral("toolButton_3"));
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/voice_chat.ico"), QSize(), QIcon::Normal, QIcon::Off);
        toolButton_3->setIcon(icon2);
        toolButton_3->setIconSize(QSize(45, 45));
        toolButton_3->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);

        horizontalLayout->addWidget(toolButton_3);

        toolButton_4 = new QToolButton(frame);
        toolButton_4->setObjectName(QStringLiteral("toolButton_4"));
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/images/find.ico"), QSize(), QIcon::Normal, QIcon::Off);
        toolButton_4->setIcon(icon3);
        toolButton_4->setIconSize(QSize(45, 45));
        toolButton_4->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);

        horizontalLayout->addWidget(toolButton_4);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        toolButton_5 = new QToolButton(frame);
        toolButton_5->setObjectName(QStringLiteral("toolButton_5"));
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/icons/images/app.ico"), QSize(), QIcon::Normal, QIcon::Off);
        toolButton_5->setIcon(icon4);
        toolButton_5->setIconSize(QSize(45, 45));
        toolButton_5->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);

        horizontalLayout->addWidget(toolButton_5);

        label = new QLabel(Widget);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(75, 145, 64, 16));
        label_2 = new QLabel(Widget);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(55, 250, 96, 16));
        editRequired = new QLineEdit(Widget);
        editRequired->setObjectName(QStringLiteral("editRequired"));
        editRequired->setGeometry(QRect(155, 140, 116, 36));
        editName = new QLineEdit(Widget);
        editName->setObjectName(QStringLiteral("editName"));
        editName->setGeometry(QRect(155, 190, 116, 36));
        editName->setStyleSheet(QStringLiteral(""));
        editName->setReadOnly(false);
        lineEdit_3 = new QLineEdit(Widget);
        lineEdit_3->setObjectName(QStringLiteral("lineEdit_3"));
        lineEdit_3->setGeometry(QRect(155, 240, 116, 41));
        lineEdit_3->setReadOnly(true);
        label_3 = new QLabel(Widget);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(75, 195, 64, 16));
        pushButton = new QPushButton(Widget);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        pushButton->setGeometry(QRect(375, 135, 64, 66));

        retranslateUi(Widget);

        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QApplication::translate("Widget", "Widget Application", nullptr));
        toolButton_2->setText(QApplication::translate("Widget", "\350\247\206\351\242\221\344\272\244\346\265\201", nullptr));
        toolButton->setText(QApplication::translate("Widget", "\345\260\217\347\273\204\350\256\250\350\256\272", nullptr));
        toolButton_3->setText(QApplication::translate("Widget", "\351\237\263\351\207\217\346\216\247\345\210\266", nullptr));
        toolButton_4->setText(QApplication::translate("Widget", "\344\272\272\345\221\230\346\237\245\346\211\276", nullptr));
        toolButton_5->setText(QApplication::translate("Widget", "\347\273\237\350\256\241\346\230\276\347\244\272", nullptr));
        label->setText(QApplication::translate("Widget", "\345\277\205\345\241\253\345\255\227\346\256\265", nullptr));
        label_2->setText(QApplication::translate("Widget", "\345\217\252\350\257\273\345\255\227\346\256\265\345\255\227\346\256\265", nullptr));
        editName->setText(QApplication::translate("Widget", "\344\275\240\345\245\275\347\232\204", nullptr));
        label_3->setText(QApplication::translate("Widget", "\346\231\256\351\200\232\345\255\227\346\256\265", nullptr));
        pushButton->setText(QApplication::translate("Widget", "OK", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
