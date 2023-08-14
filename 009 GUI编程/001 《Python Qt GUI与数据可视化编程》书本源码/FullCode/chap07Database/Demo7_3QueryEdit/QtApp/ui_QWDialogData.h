/********************************************************************************
** Form generated from reading UI file 'QWDialogData.ui'
**
** Created by: Qt User Interface Compiler version 5.10.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QWDIALOGDATA_H
#define UI_QWDIALOGDATA_H

#include <QtCore/QDate>
#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDateEdit>
#include <QtWidgets/QDialog>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_QWDialogData
{
public:
    QHBoxLayout *horizontalLayout_2;
    QGroupBox *groupBox_2;
    QHBoxLayout *horizontalLayout;
    QGridLayout *gridLayout;
    QLabel *label_12;
    QLineEdit *editName;
    QLabel *label_7;
    QComboBox *comboSex;
    QLabel *label_5;
    QDateEdit *editBirth;
    QComboBox *comboProvince;
    QLabel *label_6;
    QComboBox *comboDep;
    QSpinBox *spinSalary;
    QPlainTextEdit *editMemo;
    QLabel *label_9;
    QLabel *label_2;
    QSpinBox *spinEmpNo;
    QLabel *label;
    QLabel *label_3;
    QVBoxLayout *verticalLayout;
    QLabel *label_13;
    QLabel *LabPhoto;
    QSpacerItem *verticalSpacer;
    QFrame *frame;
    QVBoxLayout *verticalLayout_2;
    QPushButton *btnSetPhoto;
    QPushButton *btnClearPhoto;
    QPushButton *btnOK;
    QPushButton *btnClose;
    QSpacerItem *verticalSpacer_2;

    void setupUi(QDialog *QWDialogData)
    {
        if (QWDialogData->objectName().isEmpty())
            QWDialogData->setObjectName(QStringLiteral("QWDialogData"));
        QWDialogData->resize(499, 322);
        QFont font;
        font.setPointSize(10);
        QWDialogData->setFont(font);
        horizontalLayout_2 = new QHBoxLayout(QWDialogData);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        groupBox_2 = new QGroupBox(QWDialogData);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        horizontalLayout = new QHBoxLayout(groupBox_2);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        label_12 = new QLabel(groupBox_2);
        label_12->setObjectName(QStringLiteral("label_12"));

        gridLayout->addWidget(label_12, 6, 0, 1, 1);

        editName = new QLineEdit(groupBox_2);
        editName->setObjectName(QStringLiteral("editName"));

        gridLayout->addWidget(editName, 1, 1, 1, 1);

        label_7 = new QLabel(groupBox_2);
        label_7->setObjectName(QStringLiteral("label_7"));

        gridLayout->addWidget(label_7, 4, 0, 1, 1);

        comboSex = new QComboBox(groupBox_2);
        comboSex->addItem(QString());
        comboSex->addItem(QString());
        comboSex->setObjectName(QStringLiteral("comboSex"));

        gridLayout->addWidget(comboSex, 2, 1, 1, 1);

        label_5 = new QLabel(groupBox_2);
        label_5->setObjectName(QStringLiteral("label_5"));

        gridLayout->addWidget(label_5, 3, 0, 1, 1);

        editBirth = new QDateEdit(groupBox_2);
        editBirth->setObjectName(QStringLiteral("editBirth"));
        editBirth->setCalendarPopup(true);
        editBirth->setDate(QDate(2017, 2, 20));

        gridLayout->addWidget(editBirth, 3, 1, 1, 1);

        comboProvince = new QComboBox(groupBox_2);
        comboProvince->addItem(QString());
        comboProvince->addItem(QString());
        comboProvince->addItem(QString());
        comboProvince->addItem(QString());
        comboProvince->addItem(QString());
        comboProvince->addItem(QString());
        comboProvince->addItem(QString());
        comboProvince->addItem(QString());
        comboProvince->addItem(QString());
        comboProvince->addItem(QString());
        comboProvince->addItem(QString());
        comboProvince->addItem(QString());
        comboProvince->addItem(QString());
        comboProvince->addItem(QString());
        comboProvince->setObjectName(QStringLiteral("comboProvince"));
        comboProvince->setEditable(true);

        gridLayout->addWidget(comboProvince, 4, 1, 1, 1);

        label_6 = new QLabel(groupBox_2);
        label_6->setObjectName(QStringLiteral("label_6"));

        gridLayout->addWidget(label_6, 5, 0, 1, 1);

        comboDep = new QComboBox(groupBox_2);
        comboDep->addItem(QString());
        comboDep->addItem(QString());
        comboDep->addItem(QString());
        comboDep->addItem(QString());
        comboDep->setObjectName(QStringLiteral("comboDep"));
        comboDep->setEditable(true);

        gridLayout->addWidget(comboDep, 5, 1, 1, 1);

        spinSalary = new QSpinBox(groupBox_2);
        spinSalary->setObjectName(QStringLiteral("spinSalary"));
        spinSalary->setMinimum(1000);
        spinSalary->setMaximum(50000);
        spinSalary->setSingleStep(100);
        spinSalary->setValue(1000);

        gridLayout->addWidget(spinSalary, 6, 1, 1, 1);

        editMemo = new QPlainTextEdit(groupBox_2);
        editMemo->setObjectName(QStringLiteral("editMemo"));
        editMemo->setMaximumSize(QSize(16777215, 16777215));

        gridLayout->addWidget(editMemo, 7, 1, 1, 1);

        label_9 = new QLabel(groupBox_2);
        label_9->setObjectName(QStringLiteral("label_9"));

        gridLayout->addWidget(label_9, 7, 0, 1, 1);

        label_2 = new QLabel(groupBox_2);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        spinEmpNo = new QSpinBox(groupBox_2);
        spinEmpNo->setObjectName(QStringLiteral("spinEmpNo"));
        spinEmpNo->setMinimum(1);
        spinEmpNo->setMaximum(10000);

        gridLayout->addWidget(spinEmpNo, 0, 1, 1, 1);

        label = new QLabel(groupBox_2);
        label->setObjectName(QStringLiteral("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        label_3 = new QLabel(groupBox_2);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout->addWidget(label_3, 2, 0, 1, 1);


        horizontalLayout->addLayout(gridLayout);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        label_13 = new QLabel(groupBox_2);
        label_13->setObjectName(QStringLiteral("label_13"));

        verticalLayout->addWidget(label_13);

        LabPhoto = new QLabel(groupBox_2);
        LabPhoto->setObjectName(QStringLiteral("LabPhoto"));
        LabPhoto->setMinimumSize(QSize(150, 0));
        LabPhoto->setMaximumSize(QSize(350, 16777215));

        verticalLayout->addWidget(LabPhoto);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);


        horizontalLayout->addLayout(verticalLayout);


        horizontalLayout_2->addWidget(groupBox_2);

        frame = new QFrame(QWDialogData);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        verticalLayout_2 = new QVBoxLayout(frame);
        verticalLayout_2->setSpacing(25);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        btnSetPhoto = new QPushButton(frame);
        btnSetPhoto->setObjectName(QStringLiteral("btnSetPhoto"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/00.JPG"), QSize(), QIcon::Normal, QIcon::Off);
        btnSetPhoto->setIcon(icon);

        verticalLayout_2->addWidget(btnSetPhoto);

        btnClearPhoto = new QPushButton(frame);
        btnClearPhoto->setObjectName(QStringLiteral("btnClearPhoto"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/103.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnClearPhoto->setIcon(icon1);

        verticalLayout_2->addWidget(btnClearPhoto);

        btnOK = new QPushButton(frame);
        btnOK->setObjectName(QStringLiteral("btnOK"));
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/322.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnOK->setIcon(icon2);

        verticalLayout_2->addWidget(btnOK);

        btnClose = new QPushButton(frame);
        btnClose->setObjectName(QStringLiteral("btnClose"));
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/images/324.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnClose->setIcon(icon3);

        verticalLayout_2->addWidget(btnClose);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer_2);


        horizontalLayout_2->addWidget(frame);

#ifndef QT_NO_SHORTCUT
        label_12->setBuddy(spinSalary);
        label_7->setBuddy(comboProvince);
        label_5->setBuddy(editBirth);
        label_6->setBuddy(comboDep);
        label_9->setBuddy(editMemo);
        label_2->setBuddy(editName);
        label->setBuddy(spinEmpNo);
        label_3->setBuddy(comboSex);
#endif // QT_NO_SHORTCUT

        retranslateUi(QWDialogData);
        QObject::connect(btnClose, SIGNAL(clicked()), QWDialogData, SLOT(reject()));
        QObject::connect(btnOK, SIGNAL(clicked()), QWDialogData, SLOT(accept()));

        QMetaObject::connectSlotsByName(QWDialogData);
    } // setupUi

    void retranslateUi(QDialog *QWDialogData)
    {
        QWDialogData->setWindowTitle(QApplication::translate("QWDialogData", "\347\274\226\350\276\221\350\256\260\345\275\225", nullptr));
        groupBox_2->setTitle(QString());
        label_12->setText(QApplication::translate("QWDialogData", "\345\267\245  \350\265\204", nullptr));
        label_7->setText(QApplication::translate("QWDialogData", "\345\207\272\347\224\237\347\234\201\344\273\275", nullptr));
        comboSex->setItemText(0, QApplication::translate("QWDialogData", "\347\224\267", nullptr));
        comboSex->setItemText(1, QApplication::translate("QWDialogData", "\345\245\263", nullptr));

        label_5->setText(QApplication::translate("QWDialogData", "\345\207\272\347\224\237\346\227\245\346\234\237", nullptr));
        editBirth->setDisplayFormat(QApplication::translate("QWDialogData", "yyyy-MM-dd", nullptr));
        comboProvince->setItemText(0, QApplication::translate("QWDialogData", "\345\214\227\344\272\254", nullptr));
        comboProvince->setItemText(1, QApplication::translate("QWDialogData", "\344\270\212\346\265\267", nullptr));
        comboProvince->setItemText(2, QApplication::translate("QWDialogData", "\345\244\251\346\264\245", nullptr));
        comboProvince->setItemText(3, QApplication::translate("QWDialogData", "\351\207\215\345\272\206", nullptr));
        comboProvince->setItemText(4, QApplication::translate("QWDialogData", "\345\256\211\345\276\275", nullptr));
        comboProvince->setItemText(5, QApplication::translate("QWDialogData", "\346\262\263\345\214\227", nullptr));
        comboProvince->setItemText(6, QApplication::translate("QWDialogData", "\346\262\263\345\215\227", nullptr));
        comboProvince->setItemText(7, QApplication::translate("QWDialogData", "\346\271\226\345\214\227", nullptr));
        comboProvince->setItemText(8, QApplication::translate("QWDialogData", "\346\271\226\345\215\227", nullptr));
        comboProvince->setItemText(9, QApplication::translate("QWDialogData", "\345\261\261\350\245\277", nullptr));
        comboProvince->setItemText(10, QApplication::translate("QWDialogData", "\345\261\261\344\270\234", nullptr));
        comboProvince->setItemText(11, QApplication::translate("QWDialogData", "\351\231\225\350\245\277", nullptr));
        comboProvince->setItemText(12, QApplication::translate("QWDialogData", "\346\261\237\350\213\217", nullptr));
        comboProvince->setItemText(13, QApplication::translate("QWDialogData", "\346\261\237\350\245\277", nullptr));

        comboProvince->setCurrentText(QApplication::translate("QWDialogData", "\345\214\227\344\272\254", nullptr));
        label_6->setText(QApplication::translate("QWDialogData", "\351\203\250  \351\227\250", nullptr));
        comboDep->setItemText(0, QApplication::translate("QWDialogData", "\351\224\200\345\224\256\351\203\250", nullptr));
        comboDep->setItemText(1, QApplication::translate("QWDialogData", "\346\212\200\346\234\257\351\203\250", nullptr));
        comboDep->setItemText(2, QApplication::translate("QWDialogData", "\347\224\237\344\272\247\351\203\250", nullptr));
        comboDep->setItemText(3, QApplication::translate("QWDialogData", "\350\241\214\346\224\277\351\203\250", nullptr));

        comboDep->setCurrentText(QApplication::translate("QWDialogData", "\351\224\200\345\224\256\351\203\250", nullptr));
        spinSalary->setPrefix(QString());
        label_9->setText(QApplication::translate("QWDialogData", "\345\244\207   \346\263\250", nullptr));
        label_2->setText(QApplication::translate("QWDialogData", "\345\247\223  \345\220\215", nullptr));
        label->setText(QApplication::translate("QWDialogData", "\345\267\245  \345\217\267", nullptr));
        label_3->setText(QApplication::translate("QWDialogData", "\346\200\247  \345\210\253", nullptr));
        label_13->setText(QApplication::translate("QWDialogData", "\347\205\247  \347\211\207", nullptr));
        LabPhoto->setText(QApplication::translate("QWDialogData", "LabPhoto", nullptr));
        btnSetPhoto->setText(QApplication::translate("QWDialogData", "\345\257\274\345\205\245\347\205\247\347\211\207", nullptr));
        btnClearPhoto->setText(QApplication::translate("QWDialogData", "\346\270\205\351\231\244\347\205\247\347\211\207", nullptr));
        btnOK->setText(QApplication::translate("QWDialogData", "\347\241\256  \345\256\232", nullptr));
        btnClose->setText(QApplication::translate("QWDialogData", "\345\217\226  \346\266\210", nullptr));
    } // retranslateUi

};

namespace Ui {
    class QWDialogData: public Ui_QWDialogData {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QWDIALOGDATA_H
