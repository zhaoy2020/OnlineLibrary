/********************************************************************************
** Form generated from reading UI file 'QWFormDoc.ui'
**
** Created by: Qt User Interface Compiler version 5.10.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QWFORMDOC_H
#define UI_QWFORMDOC_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_QWFormDoc
{
public:
    QVBoxLayout *verticalLayout;
    QPlainTextEdit *plainTextEdit;

    void setupUi(QWidget *QWFormDoc)
    {
        if (QWFormDoc->objectName().isEmpty())
            QWFormDoc->setObjectName(QStringLiteral("QWFormDoc"));
        QWFormDoc->resize(666, 401);
        QFont font;
        font.setPointSize(10);
        QWFormDoc->setFont(font);
        verticalLayout = new QVBoxLayout(QWFormDoc);
        verticalLayout->setSpacing(3);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(3, 3, 3, 3);
        plainTextEdit = new QPlainTextEdit(QWFormDoc);
        plainTextEdit->setObjectName(QStringLiteral("plainTextEdit"));
        QFont font1;
        font1.setFamily(QString::fromUtf8("\345\256\213\344\275\223"));
        plainTextEdit->setFont(font1);

        verticalLayout->addWidget(plainTextEdit);


        retranslateUi(QWFormDoc);

        QMetaObject::connectSlotsByName(QWFormDoc);
    } // setupUi

    void retranslateUi(QWidget *QWFormDoc)
    {
        QWFormDoc->setWindowTitle(QApplication::translate("QWFormDoc", "new document", nullptr));
    } // retranslateUi

};

namespace Ui {
    class QWFormDoc: public Ui_QWFormDoc {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QWFORMDOC_H
