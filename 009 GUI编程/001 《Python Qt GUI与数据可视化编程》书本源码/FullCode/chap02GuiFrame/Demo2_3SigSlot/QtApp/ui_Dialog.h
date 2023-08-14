/********************************************************************************
** Form generated from reading UI file 'Dialog.ui'
**
** Created by: Qt User Interface Compiler version 5.12.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DIALOG_H
#define UI_DIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_Dialog
{
public:
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox1;
    QHBoxLayout *horizontalLayout_2;
    QCheckBox *chkBoxUnder;
    QCheckBox *chkBoxItalic;
    QCheckBox *chkBoxBold;
    QGroupBox *groupBox2;
    QHBoxLayout *horizontalLayout_3;
    QRadioButton *radioBlack;
    QRadioButton *radioRed;
    QRadioButton *radioBlue;
    QPlainTextEdit *textEdit;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer_2;
    QPushButton *btnClear;
    QSpacerItem *horizontalSpacer;
    QPushButton *btnOK;
    QPushButton *btnClose;

    void setupUi(QDialog *Dialog)
    {
        if (Dialog->objectName().isEmpty())
            Dialog->setObjectName(QString::fromUtf8("Dialog"));
        Dialog->resize(352, 263);
        QFont font;
        font.setFamily(QString::fromUtf8("\345\256\213\344\275\223"));
        font.setPointSize(11);
        font.setBold(true);
        font.setWeight(75);
        Dialog->setFont(font);
        verticalLayout = new QVBoxLayout(Dialog);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(-1, -1, -1, 9);
        groupBox1 = new QGroupBox(Dialog);
        groupBox1->setObjectName(QString::fromUtf8("groupBox1"));
        horizontalLayout_2 = new QHBoxLayout(groupBox1);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        chkBoxUnder = new QCheckBox(groupBox1);
        chkBoxUnder->setObjectName(QString::fromUtf8("chkBoxUnder"));

        horizontalLayout_2->addWidget(chkBoxUnder);

        chkBoxItalic = new QCheckBox(groupBox1);
        chkBoxItalic->setObjectName(QString::fromUtf8("chkBoxItalic"));

        horizontalLayout_2->addWidget(chkBoxItalic);

        chkBoxBold = new QCheckBox(groupBox1);
        chkBoxBold->setObjectName(QString::fromUtf8("chkBoxBold"));
        chkBoxBold->setChecked(true);

        horizontalLayout_2->addWidget(chkBoxBold);


        verticalLayout->addWidget(groupBox1);

        groupBox2 = new QGroupBox(Dialog);
        groupBox2->setObjectName(QString::fromUtf8("groupBox2"));
        horizontalLayout_3 = new QHBoxLayout(groupBox2);
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        radioBlack = new QRadioButton(groupBox2);
        radioBlack->setObjectName(QString::fromUtf8("radioBlack"));
        radioBlack->setChecked(true);

        horizontalLayout_3->addWidget(radioBlack);

        radioRed = new QRadioButton(groupBox2);
        radioRed->setObjectName(QString::fromUtf8("radioRed"));

        horizontalLayout_3->addWidget(radioRed);

        radioBlue = new QRadioButton(groupBox2);
        radioBlue->setObjectName(QString::fromUtf8("radioBlue"));
        radioBlue->setChecked(false);

        horizontalLayout_3->addWidget(radioBlue);


        verticalLayout->addWidget(groupBox2);

        textEdit = new QPlainTextEdit(Dialog);
        textEdit->setObjectName(QString::fromUtf8("textEdit"));
        QFont font1;
        font1.setPointSize(20);
        font1.setBold(true);
        font1.setWeight(75);
        textEdit->setFont(font1);

        verticalLayout->addWidget(textEdit);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, 10, -1, 10);
        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);

        btnClear = new QPushButton(Dialog);
        btnClear->setObjectName(QString::fromUtf8("btnClear"));

        horizontalLayout->addWidget(btnClear);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        btnOK = new QPushButton(Dialog);
        btnOK->setObjectName(QString::fromUtf8("btnOK"));

        horizontalLayout->addWidget(btnOK);

        btnClose = new QPushButton(Dialog);
        btnClose->setObjectName(QString::fromUtf8("btnClose"));

        horizontalLayout->addWidget(btnClose);


        verticalLayout->addLayout(horizontalLayout);

        QWidget::setTabOrder(chkBoxUnder, chkBoxItalic);
        QWidget::setTabOrder(chkBoxItalic, chkBoxBold);
        QWidget::setTabOrder(chkBoxBold, radioBlack);
        QWidget::setTabOrder(radioBlack, radioRed);
        QWidget::setTabOrder(radioRed, radioBlue);
        QWidget::setTabOrder(radioBlue, textEdit);
        QWidget::setTabOrder(textEdit, btnClear);
        QWidget::setTabOrder(btnClear, btnOK);
        QWidget::setTabOrder(btnOK, btnClose);

        retranslateUi(Dialog);
        QObject::connect(btnOK, SIGNAL(clicked()), Dialog, SLOT(accept()));
        QObject::connect(btnClose, SIGNAL(clicked()), Dialog, SLOT(close()));

        QMetaObject::connectSlotsByName(Dialog);
    } // setupUi

    void retranslateUi(QDialog *Dialog)
    {
        Dialog->setWindowTitle(QApplication::translate("Dialog", " Demo2-3\344\277\241\345\217\267\344\270\216\346\247\275", nullptr));
        groupBox1->setTitle(QString());
        chkBoxUnder->setText(QApplication::translate("Dialog", "Underline", nullptr));
        chkBoxItalic->setText(QApplication::translate("Dialog", "Italic", nullptr));
        chkBoxBold->setText(QApplication::translate("Dialog", "Bold", nullptr));
        groupBox2->setTitle(QString());
        radioBlack->setText(QApplication::translate("Dialog", "Black", nullptr));
        radioRed->setText(QApplication::translate("Dialog", "Red", nullptr));
        radioBlue->setText(QApplication::translate("Dialog", "Blue", nullptr));
        textEdit->setPlainText(QApplication::translate("Dialog", "PyQt5 \347\274\226\347\250\213\346\214\207\345\215\227\n"
"Python \345\222\214 Qt", nullptr));
        btnClear->setText(QApplication::translate("Dialog", "\346\270\205\347\251\272", nullptr));
        btnOK->setText(QApplication::translate("Dialog", "\347\241\256 \345\256\232", nullptr));
        btnClose->setText(QApplication::translate("Dialog", "\351\200\200\345\207\272", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Dialog: public Ui_Dialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DIALOG_H
