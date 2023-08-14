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
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QScrollBar>
#include <QtWidgets/QSlider>
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
    QSlider *slider;
    QLabel *label_2;
    QScrollBar *scrollBar;
    QLabel *label_3;
    QProgressBar *progressBar;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout_2;
    QCheckBox *chkBox_Visible;
    QCheckBox *chkBox_Inverted;
    QRadioButton *radio_Percent;
    QRadioButton *radio_Value;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName(QStringLiteral("Widget"));
        Widget->resize(374, 228);
        QFont font;
        font.setPointSize(10);
        Widget->setFont(font);
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
        gridLayout->setVerticalSpacing(12);
        label = new QLabel(groupBox);
        label->setObjectName(QStringLiteral("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        slider = new QSlider(groupBox);
        slider->setObjectName(QStringLiteral("slider"));
        slider->setMaximum(200);
        slider->setValue(23);
        slider->setOrientation(Qt::Horizontal);
        slider->setTickPosition(QSlider::NoTicks);

        gridLayout->addWidget(slider, 0, 1, 1, 1);

        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        scrollBar = new QScrollBar(groupBox);
        scrollBar->setObjectName(QStringLiteral("scrollBar"));
        scrollBar->setMaximum(200);
        scrollBar->setValue(23);
        scrollBar->setOrientation(Qt::Horizontal);
        scrollBar->setInvertedAppearance(false);

        gridLayout->addWidget(scrollBar, 1, 1, 1, 1);

        label_3 = new QLabel(groupBox);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout->addWidget(label_3, 2, 0, 1, 1);

        progressBar = new QProgressBar(groupBox);
        progressBar->setObjectName(QStringLiteral("progressBar"));
        progressBar->setMaximum(200);
        progressBar->setValue(24);
        progressBar->setTextVisible(true);
        progressBar->setInvertedAppearance(false);
        progressBar->setTextDirection(QProgressBar::TopToBottom);

        gridLayout->addWidget(progressBar, 2, 1, 1, 1);


        verticalLayout->addWidget(groupBox);

        groupBox_2 = new QGroupBox(Widget);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        gridLayout_2 = new QGridLayout(groupBox_2);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        chkBox_Visible = new QCheckBox(groupBox_2);
        chkBox_Visible->setObjectName(QStringLiteral("chkBox_Visible"));
        chkBox_Visible->setChecked(true);

        gridLayout_2->addWidget(chkBox_Visible, 0, 0, 1, 1);

        chkBox_Inverted = new QCheckBox(groupBox_2);
        chkBox_Inverted->setObjectName(QStringLiteral("chkBox_Inverted"));

        gridLayout_2->addWidget(chkBox_Inverted, 0, 1, 1, 1);

        radio_Percent = new QRadioButton(groupBox_2);
        radio_Percent->setObjectName(QStringLiteral("radio_Percent"));
        radio_Percent->setChecked(true);

        gridLayout_2->addWidget(radio_Percent, 1, 0, 1, 1);

        radio_Value = new QRadioButton(groupBox_2);
        radio_Value->setObjectName(QStringLiteral("radio_Value"));
        radio_Value->setChecked(false);

        gridLayout_2->addWidget(radio_Value, 1, 1, 1, 1);


        verticalLayout->addWidget(groupBox_2);


        retranslateUi(Widget);

        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QApplication::translate("Widget", "Demo3_3 progressBar", nullptr));
        groupBox->setTitle(QString());
        label->setText(QApplication::translate("Widget", "Slider", nullptr));
        label_2->setText(QApplication::translate("Widget", "ScrollBar", nullptr));
        label_3->setText(QApplication::translate("Widget", "ProgressBar", nullptr));
        groupBox_2->setTitle(QApplication::translate("Widget", "ProgressBar\350\256\276\347\275\256", nullptr));
        chkBox_Visible->setText(QApplication::translate("Widget", "textVisible", nullptr));
        chkBox_Inverted->setText(QApplication::translate("Widget", "InvertedAppearance", nullptr));
        radio_Percent->setText(QApplication::translate("Widget", "\346\230\276\347\244\272\346\240\274\345\274\217--\347\231\276\345\210\206\346\257\224", nullptr));
        radio_Value->setText(QApplication::translate("Widget", "\346\230\276\347\244\272\346\240\274\345\274\217--\345\275\223\345\211\215\345\200\274", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
