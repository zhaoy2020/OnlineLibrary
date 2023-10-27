/********************************************************************************
** Form generated from reading UI file 'QWDialogSize.ui'
**
** Created by: Qt User Interface Compiler version 5.10.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QWDIALOGSIZE_H
#define UI_QWDIALOGSIZE_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_QWDialogSize
{
public:
    QHBoxLayout *horizontalLayout;
    QGroupBox *groupBox;
    QGridLayout *gridLayout;
    QLabel *label_2;
    QSpinBox *spin_ColCount;
    QLabel *label;
    QSpinBox *spin_RwoCount;
    QFrame *frame;
    QVBoxLayout *verticalLayout;
    QPushButton *btnOK;
    QPushButton *btnCancel;

    void setupUi(QDialog *QWDialogSize)
    {
        if (QWDialogSize->objectName().isEmpty())
            QWDialogSize->setObjectName(QStringLiteral("QWDialogSize"));
        QWDialogSize->setWindowModality(Qt::NonModal);
        QWDialogSize->resize(266, 132);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(QWDialogSize->sizePolicy().hasHeightForWidth());
        QWDialogSize->setSizePolicy(sizePolicy);
        QFont font;
        font.setPointSize(10);
        QWDialogSize->setFont(font);
        QWDialogSize->setSizeGripEnabled(false);
        QWDialogSize->setModal(false);
        horizontalLayout = new QHBoxLayout(QWDialogSize);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        groupBox = new QGroupBox(QWDialogSize);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        gridLayout = new QGridLayout(groupBox);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setVerticalSpacing(15);
        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout->addWidget(label_2, 0, 0, 1, 1);

        spin_ColCount = new QSpinBox(groupBox);
        spin_ColCount->setObjectName(QStringLiteral("spin_ColCount"));
        spin_ColCount->setMinimum(1);
        spin_ColCount->setMaximum(500);
        spin_ColCount->setValue(6);

        gridLayout->addWidget(spin_ColCount, 0, 1, 1, 1);

        label = new QLabel(groupBox);
        label->setObjectName(QStringLiteral("label"));

        gridLayout->addWidget(label, 1, 0, 1, 1);

        spin_RwoCount = new QSpinBox(groupBox);
        spin_RwoCount->setObjectName(QStringLiteral("spin_RwoCount"));
        spin_RwoCount->setMinimum(1);
        spin_RwoCount->setMaximum(500);
        spin_RwoCount->setValue(10);

        gridLayout->addWidget(spin_RwoCount, 1, 1, 1, 1);


        horizontalLayout->addWidget(groupBox);

        frame = new QFrame(QWDialogSize);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setMaximumSize(QSize(90, 16777215));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        verticalLayout = new QVBoxLayout(frame);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        btnOK = new QPushButton(frame);
        btnOK->setObjectName(QStringLiteral("btnOK"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/704.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnOK->setIcon(icon);

        verticalLayout->addWidget(btnOK);

        btnCancel = new QPushButton(frame);
        btnCancel->setObjectName(QStringLiteral("btnCancel"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/706.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnCancel->setIcon(icon1);

        verticalLayout->addWidget(btnCancel);


        horizontalLayout->addWidget(frame);


        retranslateUi(QWDialogSize);
        QObject::connect(btnOK, SIGNAL(clicked()), QWDialogSize, SLOT(accept()));
        QObject::connect(btnCancel, SIGNAL(clicked()), QWDialogSize, SLOT(reject()));

        QMetaObject::connectSlotsByName(QWDialogSize);
    } // setupUi

    void retranslateUi(QDialog *QWDialogSize)
    {
        QWDialogSize->setWindowTitle(QApplication::translate("QWDialogSize", "\350\256\276\347\275\256\350\241\250\346\240\274\350\241\214\346\225\260\345\222\214\345\210\227\346\225\260", nullptr));
        groupBox->setTitle(QApplication::translate("QWDialogSize", "\350\256\276\347\275\256\350\241\250\346\240\274\350\241\214\346\225\260\345\222\214\345\210\227\346\225\260", nullptr));
        label_2->setText(QApplication::translate("QWDialogSize", "\345\210\227  \346\225\260", nullptr));
        label->setText(QApplication::translate("QWDialogSize", "\350\241\214  \346\225\260", nullptr));
        btnOK->setText(QApplication::translate("QWDialogSize", "\347\241\256\345\256\232", nullptr));
        btnCancel->setText(QApplication::translate("QWDialogSize", "\345\217\226\346\266\210", nullptr));
    } // retranslateUi

};

namespace Ui {
    class QWDialogSize: public Ui_QWDialogSize {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QWDIALOGSIZE_H
