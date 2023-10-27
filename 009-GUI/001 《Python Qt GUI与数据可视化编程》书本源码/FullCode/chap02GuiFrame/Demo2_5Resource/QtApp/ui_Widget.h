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
#include <QtGui/QIcon>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Widget
{
public:
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox_Age;
    QGridLayout *gridLayout;
    QLabel *label;
    QSlider *sliderSetAge;
    QLabel *label_2;
    QLineEdit *editAgeInt;
    QLabel *label_3;
    QLineEdit *editAgeStr;
    QGroupBox *groupBox_Name;
    QGridLayout *gridLayout_2;
    QLabel *label_5;
    QLineEdit *editNameInput;
    QPushButton *btnSetName;
    QLabel *label_4;
    QLineEdit *editNameHello;
    QFrame *frame_Button;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer;
    QSpacerItem *horizontalSpacer_4;
    QSpacerItem *horizontalSpacer_2;
    QPushButton *btnClose;
    QSpacerItem *horizontalSpacer_3;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName(QString::fromUtf8("Widget"));
        Widget->resize(359, 257);
        verticalLayout = new QVBoxLayout(Widget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(-1, -1, -1, 5);
        groupBox_Age = new QGroupBox(Widget);
        groupBox_Age->setObjectName(QString::fromUtf8("groupBox_Age"));
        gridLayout = new QGridLayout(groupBox_Age);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(groupBox_Age);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        sliderSetAge = new QSlider(groupBox_Age);
        sliderSetAge->setObjectName(QString::fromUtf8("sliderSetAge"));
        sliderSetAge->setMaximum(100);
        sliderSetAge->setValue(50);
        sliderSetAge->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(sliderSetAge, 0, 1, 1, 1);

        label_2 = new QLabel(groupBox_Age);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        editAgeInt = new QLineEdit(groupBox_Age);
        editAgeInt->setObjectName(QString::fromUtf8("editAgeInt"));

        gridLayout->addWidget(editAgeInt, 1, 1, 1, 1);

        label_3 = new QLabel(groupBox_Age);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout->addWidget(label_3, 2, 0, 1, 1);

        editAgeStr = new QLineEdit(groupBox_Age);
        editAgeStr->setObjectName(QString::fromUtf8("editAgeStr"));

        gridLayout->addWidget(editAgeStr, 2, 1, 1, 1);


        verticalLayout->addWidget(groupBox_Age);

        groupBox_Name = new QGroupBox(Widget);
        groupBox_Name->setObjectName(QString::fromUtf8("groupBox_Name"));
        gridLayout_2 = new QGridLayout(groupBox_Name);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        label_5 = new QLabel(groupBox_Name);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setAlignment(Qt::AlignCenter);

        gridLayout_2->addWidget(label_5, 0, 0, 1, 1);

        editNameInput = new QLineEdit(groupBox_Name);
        editNameInput->setObjectName(QString::fromUtf8("editNameInput"));

        gridLayout_2->addWidget(editNameInput, 0, 1, 1, 1);

        btnSetName = new QPushButton(groupBox_Name);
        btnSetName->setObjectName(QString::fromUtf8("btnSetName"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/images/322.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnSetName->setIcon(icon);

        gridLayout_2->addWidget(btnSetName, 0, 2, 1, 1);

        label_4 = new QLabel(groupBox_Name);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout_2->addWidget(label_4, 1, 0, 1, 1);

        editNameHello = new QLineEdit(groupBox_Name);
        editNameHello->setObjectName(QString::fromUtf8("editNameHello"));

        gridLayout_2->addWidget(editNameHello, 1, 1, 1, 2);


        verticalLayout->addWidget(groupBox_Name);

        frame_Button = new QFrame(Widget);
        frame_Button->setObjectName(QString::fromUtf8("frame_Button"));
        frame_Button->setMaximumSize(QSize(16777215, 45));
        frame_Button->setFrameShape(QFrame::NoFrame);
        frame_Button->setFrameShadow(QFrame::Raised);
        horizontalLayout = new QHBoxLayout(frame_Button);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, -1, -1, 5);
        horizontalSpacer = new QSpacerItem(83, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_4);

        horizontalSpacer_2 = new QSpacerItem(84, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);

        btnClose = new QPushButton(frame_Button);
        btnClose->setObjectName(QString::fromUtf8("btnClose"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnClose->setIcon(icon1);

        horizontalLayout->addWidget(btnClose);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_3);


        verticalLayout->addWidget(frame_Button);


        retranslateUi(Widget);
        QObject::connect(btnClose, SIGNAL(clicked()), Widget, SLOT(close()));

        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QApplication::translate("Widget", "Demo2-5\357\274\214 \350\265\204\346\272\220\346\226\207\344\273\266\343\200\201\350\207\252\345\256\232\344\271\211\344\277\241\345\217\267", nullptr));
        groupBox_Age->setTitle(QApplication::translate("Widget", "\345\271\264\351\276\204\350\256\276\347\275\256", nullptr));
        label->setText(QApplication::translate("Widget", "\350\256\276\347\275\256\345\271\264\351\276\204(0~100)", nullptr));
        label_2->setText(QApplication::translate("Widget", "ageChanged(int)\345\223\215\345\272\224", nullptr));
        label_3->setText(QApplication::translate("Widget", "ageChanged(str)\345\223\215\345\272\224", nullptr));
        groupBox_Name->setTitle(QApplication::translate("Widget", "\345\247\223\345\220\215\350\256\276\347\275\256", nullptr));
        label_5->setText(QApplication::translate("Widget", "\350\276\223\345\205\245\345\247\223\345\220\215", nullptr));
        editNameInput->setText(QApplication::translate("Widget", "Mike", nullptr));
        btnSetName->setText(QApplication::translate("Widget", "\350\256\276\347\275\256\345\247\223\345\220\215", nullptr));
        label_4->setText(QApplication::translate("Widget", "nameChanged(str)\345\223\215\345\272\224", nullptr));
        btnClose->setText(QApplication::translate("Widget", "\345\205\263\351\227\255", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
