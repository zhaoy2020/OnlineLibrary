/********************************************************************************
** Form generated from reading UI file 'Widget.ui'
**
** Created by: Qt User Interface Compiler version 5.10.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_H
#define UI_WIDGET_H

#include <QtCore/QDate>
#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCalendarWidget>
#include <QtWidgets/QDateEdit>
#include <QtWidgets/QDateTimeEdit>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTimeEdit>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Widget
{
public:
    QHBoxLayout *horizontalLayout;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout;
    QPushButton *btnGetTime;
    QLabel *label;
    QTimeEdit *timeEdit;
    QLineEdit *editTime;
    QPushButton *btnSetTime;
    QLabel *label_2;
    QDateEdit *dateEdit;
    QLineEdit *editDate;
    QPushButton *btnSetDate;
    QLabel *label_3;
    QDateTimeEdit *dateTimeEdit;
    QLineEdit *editDateTime;
    QPushButton *btnSetDateTime;
    QLabel *LabDateTime;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout_3;
    QLineEdit *editCalendar;
    QCalendarWidget *calendarWidget;
    QLabel *label_5;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName(QStringLiteral("Widget"));
        Widget->resize(688, 283);
        QFont font;
        font.setPointSize(10);
        Widget->setFont(font);
        horizontalLayout = new QHBoxLayout(Widget);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        groupBox_2 = new QGroupBox(Widget);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        gridLayout = new QGridLayout(groupBox_2);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        btnGetTime = new QPushButton(groupBox_2);
        btnGetTime->setObjectName(QStringLiteral("btnGetTime"));

        gridLayout->addWidget(btnGetTime, 0, 1, 1, 2);

        label = new QLabel(groupBox_2);
        label->setObjectName(QStringLiteral("label"));

        gridLayout->addWidget(label, 1, 0, 1, 1);

        timeEdit = new QTimeEdit(groupBox_2);
        timeEdit->setObjectName(QStringLiteral("timeEdit"));
        timeEdit->setCalendarPopup(true);
        timeEdit->setTime(QTime(15, 30, 55));

        gridLayout->addWidget(timeEdit, 1, 1, 1, 2);

        editTime = new QLineEdit(groupBox_2);
        editTime->setObjectName(QStringLiteral("editTime"));

        gridLayout->addWidget(editTime, 1, 3, 1, 1);

        btnSetTime = new QPushButton(groupBox_2);
        btnSetTime->setObjectName(QStringLiteral("btnSetTime"));

        gridLayout->addWidget(btnSetTime, 2, 3, 1, 1);

        label_2 = new QLabel(groupBox_2);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout->addWidget(label_2, 3, 0, 1, 1);

        dateEdit = new QDateEdit(groupBox_2);
        dateEdit->setObjectName(QStringLiteral("dateEdit"));
        dateEdit->setCurrentSection(QDateTimeEdit::YearSection);
        dateEdit->setCalendarPopup(true);
        dateEdit->setCurrentSectionIndex(0);
        dateEdit->setDate(QDate(2016, 11, 21));

        gridLayout->addWidget(dateEdit, 3, 2, 1, 1);

        editDate = new QLineEdit(groupBox_2);
        editDate->setObjectName(QStringLiteral("editDate"));

        gridLayout->addWidget(editDate, 3, 3, 1, 1);

        btnSetDate = new QPushButton(groupBox_2);
        btnSetDate->setObjectName(QStringLiteral("btnSetDate"));

        gridLayout->addWidget(btnSetDate, 4, 3, 1, 1);

        label_3 = new QLabel(groupBox_2);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout->addWidget(label_3, 5, 0, 1, 2);

        dateTimeEdit = new QDateTimeEdit(groupBox_2);
        dateTimeEdit->setObjectName(QStringLiteral("dateTimeEdit"));
        dateTimeEdit->setDateTime(QDateTime(QDate(2018, 10, 9), QTime(8, 21, 28)));
        dateTimeEdit->setDate(QDate(2018, 10, 9));
        dateTimeEdit->setTime(QTime(8, 21, 28));
        dateTimeEdit->setMaximumDateTime(QDateTime(QDate(3000, 12, 31), QTime(23, 59, 59)));
        dateTimeEdit->setMinimumDateTime(QDateTime(QDate(1763, 9, 14), QTime(0, 0, 0)));
        dateTimeEdit->setMinimumDate(QDate(1763, 9, 14));
        dateTimeEdit->setCurrentSection(QDateTimeEdit::YearSection);
        dateTimeEdit->setCalendarPopup(true);
        dateTimeEdit->setCurrentSectionIndex(0);
        dateTimeEdit->setTimeSpec(Qt::LocalTime);

        gridLayout->addWidget(dateTimeEdit, 5, 2, 1, 1);

        editDateTime = new QLineEdit(groupBox_2);
        editDateTime->setObjectName(QStringLiteral("editDateTime"));
        editDateTime->setMinimumSize(QSize(150, 0));

        gridLayout->addWidget(editDateTime, 5, 3, 1, 1);

        btnSetDateTime = new QPushButton(groupBox_2);
        btnSetDateTime->setObjectName(QStringLiteral("btnSetDateTime"));

        gridLayout->addWidget(btnSetDateTime, 6, 3, 1, 1);

        LabDateTime = new QLabel(groupBox_2);
        LabDateTime->setObjectName(QStringLiteral("LabDateTime"));
        LabDateTime->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(LabDateTime, 0, 3, 1, 1);


        horizontalLayout->addWidget(groupBox_2);

        groupBox_3 = new QGroupBox(Widget);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        gridLayout_3 = new QGridLayout(groupBox_3);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        editCalendar = new QLineEdit(groupBox_3);
        editCalendar->setObjectName(QStringLiteral("editCalendar"));

        gridLayout_3->addWidget(editCalendar, 0, 1, 1, 1);

        calendarWidget = new QCalendarWidget(groupBox_3);
        calendarWidget->setObjectName(QStringLiteral("calendarWidget"));
        calendarWidget->setGridVisible(false);

        gridLayout_3->addWidget(calendarWidget, 1, 0, 1, 2);

        label_5 = new QLabel(groupBox_3);
        label_5->setObjectName(QStringLiteral("label_5"));

        gridLayout_3->addWidget(label_5, 0, 0, 1, 1);


        horizontalLayout->addWidget(groupBox_3);


        retranslateUi(Widget);

        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QApplication::translate("Widget", "Demo3_4 \346\227\245\346\234\237\346\227\266\351\227\264", nullptr));
        groupBox_2->setTitle(QApplication::translate("Widget", "\346\227\245\346\234\237\346\227\266\351\227\264", nullptr));
        btnGetTime->setText(QApplication::translate("Widget", "\350\257\273\345\217\226\345\275\223\345\211\215\346\227\245\346\234\237\346\227\266\351\227\264", nullptr));
        label->setText(QApplication::translate("Widget", "\346\227\266  \351\227\264", nullptr));
        timeEdit->setDisplayFormat(QApplication::translate("Widget", "HH:mm:ss", nullptr));
        editTime->setInputMask(QApplication::translate("Widget", "99:99:99", nullptr));
        btnSetTime->setText(QApplication::translate("Widget", "\350\256\276\347\275\256\346\227\266\351\227\264", nullptr));
        label_2->setText(QApplication::translate("Widget", "\346\227\245  \346\234\237", nullptr));
        dateEdit->setDisplayFormat(QApplication::translate("Widget", "yyyy\345\271\264M\346\234\210d\346\227\245", nullptr));
        editDate->setInputMask(QApplication::translate("Widget", "9999-99-99", nullptr));
        btnSetDate->setText(QApplication::translate("Widget", "\350\256\276\347\275\256\346\227\245\346\234\237", nullptr));
        label_3->setText(QApplication::translate("Widget", "\346\227\245\346\234\237\346\227\266\351\227\264", nullptr));
        dateTimeEdit->setDisplayFormat(QApplication::translate("Widget", "yyyy-MM-dd HH:mm:ss", nullptr));
        editDateTime->setInputMask(QApplication::translate("Widget", "9999-99-99 99:99:99", nullptr));
        btnSetDateTime->setText(QApplication::translate("Widget", "\350\256\276\347\275\256\346\227\245\346\234\237\346\227\266\351\227\264", nullptr));
        LabDateTime->setText(QApplication::translate("Widget", "\345\255\227\347\254\246\344\270\262\346\230\276\347\244\272", nullptr));
        groupBox_3->setTitle(QApplication::translate("Widget", "\346\227\245\345\216\206\351\200\211\346\213\251", nullptr));
        label_5->setText(QApplication::translate("Widget", "\351\200\211\346\213\251\347\232\204\346\227\245\346\234\237\357\274\232", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
